import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import wandb
import torch

from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from nanochat.dataloader import tokenizing_data_loader_with_state
from nanochat.common import autodetect_device_type, DummyWandb, compute_init, get_base_dir
from nanochat.gpt import GPT, GPTConfig

# -----------------------------------------------------------------------------
# User settings
run = "dummy" # wandb
device_type = "" # cuda|mps|cpu, autodetect by default
# Model architecture
depth = 4
max_seq_len = 512
# Training horizon. Only one of the 3 will be used, on this order of precendence
num_iterations = -1
target_flops = -1
target_param_data_ratio = 20
# Optimization
device_batch_size = 1
total_batch_size = 524_288 # Total desired batch size (in tokens)
embedding_lr = 0.2 # Learning rate for the embedding parameters (Adam)
unembedding_lr = 0.004 # Learning rate for the unembedding parameters (Adam)
weight_decay = 0.0 # Weight decay for the embedding/unembedding parameters (Adam)
matrix_lr = 0.02 # lr for the matrix parameters (Muon)
grad_clip = 1.0 
warmup_ratio = 0.0
warmdown_ratio = 0.2
final_lr_frac = 0.0 # final lr fraction of the initial LR
resume_from_step = -1 # resume training from this step of the optimization (-1 = disable)
# Evaluation
eval_every = 250 
eval_tokens = 20 * 524_288 # num of tokens to calculate the val loss on
core_metric_every = 2_000
core_metric_per_task = 500 # Examples per task in estimating the core metric
sample_every = 2_000
save_every = -1 # every how many steps to save the model checkpoint (-1 = disable, save only at the end)
model_tag = "" # optionally override the model tag for the output checkpoint dir name
# TODO: allow CLI to override the configs or config from config file
user_config = dict()
for k,v in globals().items():
    if k.startswith("_") and isinstance(k, (int, float, bool, str)):
        user_config[k] = v 
# -----------------------------------------------------------------------------

# NOTE: Currently this is single GPU setup

# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
device = compute_init()
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# wandb logging init
use_dummy_wandb = run == "dummy" 
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=run, config=user_config)

# Tokenizer
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

# Model kwargs are 
num_layers = depth
model_dim = depth * 64 # aspect ratio of 64
num_heads = max(1, (model_dim + 127) // 128) # head dim 128
num_kv_heads = num_heads # 1:1 -> GQA disabled
print(f"num_layers: {num_layers}")
print(f"model_dim: {model_dim}")
print(f"num_heads: {num_heads}")
print(f"num_kv_heads: {num_kv_heads}")

# figure out if we need gradient accumulation to reach the desired total batch size
tokens_per_fwdbwd = device_batch_size * max_seq_len # tokens per iteration
assert total_batch_size % tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // tokens_per_fwdbwd
print(f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# -----------------------------------------------------------------------------
# Initialize the Model

# Create a new model with random weights
model_config_kwargs = dict(
    sequence_len=max_seq_len, 
    vocab_size=vocab_size, 
    n_layer=num_layers, 
    n_head=num_heads, 
    n_kv_head=num_kv_heads, 
    n_embd=model_dim)
with torch.device("meta"):
    model_config = GPTConfig(**model_config_kwargs)
    model : GPT = GPT(model_config)
model.to_empty(device=device)
model.init_weights()

# If we are resuming overwrite the models parameters with the ones from the checkpoint
base_dir = get_base_dir()
output_dirname = model_tag if model_tag else f"d{depth}"
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = resume_from_step != -1
if resuming:
    print(f"Resuming optimization from step {resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, resume_from_step, device, load_optimizer=True)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # free up this memory after copy

orig_model = model # original uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change)
model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params:,}")
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token}")

# Calculate the number of iterations
assert num_iterations > 0 or target_flops > 0 or target_param_data_ratio > 0
if num_iterations > 0:
    print(f"Using user-provided number of iterations: {num_iterations}")
elif target_flops > 0:
    # calculate the number of iterations from the target flops
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
    print(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif target_param_data_ratio > 0:
    target_tokens = num_params * target_param_data_ratio
    num_iterations = target_tokens // total_batch_size
    print(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No target horizon specified")
total_tokens = total_batch_size * num_iterations
print(f"Total number of training tokens: {total_tokens:,}")
print(f"Tokens : Params ratio: {total_batch_size * num_iterations / num_params:.2f}") # Chinchilla is ~20
print(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (Muon for Linear layers, AdamW for embedding and lm_head)
optimizers = model.setup_optimizers(unembedding_lr=unembedding_lr, embedding_lr=embedding_lr, matrix_lr=matrix_lr, weight_decay=weight_decay)
adamw_optimizer, muon_optimizer = optimizers

if resuming:
    for opt, dat in zip(optimizers, optimizer_data):
        opt.load_state_dict(dat)
    del optimizer_data # free memory

# -----------------------------------------------------------------------------
# Initialize the DataLoader
tokens_dir = os.path.join(base_dir, "tokenized_data")
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_data_loader_with_state(B=device_batch_size, T=max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
build_val_loader = lambda: tokenizing_data_loader_with_state(device_batch_size, max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
