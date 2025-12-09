import wandb

from nanochat.common import autodetect_device_type

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
enembedding_lr = 0.004 # Learning rate for the unembedding parameters (Adam)
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
# TODO: allow CLI to override the configs
# TODO: allow override from config file
# -----------------------------------------------------------------------------

# NOTE: Currently this is single GPU setup

# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type

# wandb logging init
use_dummy_wandb = run == "dummy" 
