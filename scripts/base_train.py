from contextlib import nullcontext
import os
import time
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import wandb
import torch

from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from nanochat.dataloader import tokenizing_data_loader_with_state
from nanochat.common import autodetect_device_type, DummyWandb, compute_init, get_base_dir
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.report import get_report
from nanochat.gpt import GPT, GPTConfig
from scripts.base_eval import evaluate_model

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
device_batch_size = 12
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
eval_tokens = 512 # 20 * 524_288 # num of tokens to calculate the val loss on
core_metric_every = -1 # 2_000
core_metric_max_per_task = 500 # Examples per task in estimating the core metric
sample_every = 2_000
save_every = -1 # every how many steps to save the model checkpoint (-1 = disable, save only at the end)
model_tag = "" # optionally override the model tag for the output checkpoint dir name
# TODO: allow CLI to override the configs or config from config file
user_config = dict()
for k,v in list(globals().items()):
    if not k.startswith("_") and isinstance(v, (int, float, bool, str)):
        user_config[k] = v
# -----------------------------------------------------------------------------

# NOTE: Currently this is single GPU setup

# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
device = compute_init(device_type=device_type)
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
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
def build_val_loader():
    return tokenizing_data_loader_with_state(B=device_batch_size, T=max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Set up hyperparameter schedulers

# LR scheduler
def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac

# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# -----------------------------------------------------------------------------
# Loop state (variables updated by the training loop)

if not resuming:
    step = 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # wall clock time
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# -----------------------------------------------------------------------------
# Training Loop

while True:
    last_step = step == num_iterations # loop runs num_iterations + 1 for eval/save at the end
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while eval the val bpb 
    if last_step or (step > 0 and step % eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # once in a while eval the CORE metric
    # Use the original uncompiled model because the inputs keep changing shape
    results = {}
    if core_metric_every > 0 and (last_step or (step > 0 and step % core_metric_every == 0)):
        model.eval()
        with autocast_ctx:
            results = evaluate_model(orig_model, tokenizer, device, max_per_task=core_metric_max_per_task)
        print(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()
    
    # once in a while sample from the model
    # use the original uncompiled model, because the inputs keep changing shape
    if last_step or (step > 0 and step % sample_every == 0):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer)
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint ath the end of the run or every save_every steps
    if last_step or (step > 0 and step != resume_from_step and save_every > 0 and step % save_every == 0):
        save_checkpoint(
            checkpoint_dir, 
            step,
            orig_model.state_dict(),
            [opt.state_dict() for opt in optimizers],
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": device_batch_size,
                "max_seq_len": max_seq_len,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
        )

    # termination condition
    if last_step:
        break

    # -----------------------------------------------------------------------------
    # single training step
    t0 = time.time()
    train_loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss_accum += loss.detach()
        loss = loss / grad_accum_steps # normalize gradient with grad accumulation
        loss.backward()
        x, y, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward pass
    train_loss = train_loss_accum / grad_accum_steps

    # gradient clipping
    grad_clip_enabled = grad_clip > 0.0
    if grad_clip_enabled:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip)
        grad_norm = grad_norm_tensor.item() # GPU tensor -> CPU float 
    # step the optimizer
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    t1 = time.time()
    dt = t1 - t0
    # -----------------------------------------------------------------------------

    # logging
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12  # bfloat16 H100 SXM and without 2:4 sparsity
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100 # in %
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    print_grad_norm = f" grad norm: {grad_norm:.4f} |" if grad_clip_enabled else ""
    print(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} |{print_grad_norm} lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m")
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
        }
        if grad_clip_enabled:
            log_data["train/grad_norm"] = grad_norm
        wandb_run.log(log_data)

    # state update
    step += 1


print(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print(f"Total training time: {total_training_time/60:.2f}m")
print(f"Minimum validation bpb: {min_val_bpb:.4f}")

# Log to report
get_report().log(section="Base model training", data=[
    user_config, # CLI args
    { # stats about the training setup
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Params ratio": total_batch_size * num_iterations / num_params,
        "DDP world size": 1,
        "warmup_ratio": warmup_ratio,
        "warmdown_ratio": warmdown_ratio,
        "final_lr_frac": final_lr_frac,
    },
    { # stats about training outcomes
        "Minimum validation bpb": min_val_bpb,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

# cleanup
wandb_run.finish() # wandb run finish