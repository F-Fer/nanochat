# nanochat

## Quick Start

### Tokenizer Training

To train a new BPE tokenizer on your dataset:

```bash
uv run python -m scripts.tok_train
```

### Training

To start a base model training run:

```bash
uv run python -m scripts.base_train --run "my-first-run" --depth 4
```

You can customize the training horizon using iterations, target FLOPs, or parameter-to-data ratios.

## Project Structure

- `nanochat/`: Core implementation.
  - `gpt.py`: GPT architecture definition.
  - `muon.py`: Muon optimizer implementation.
  - `dataloader.py`: Distributed sharded data loading.
  - `tokenizer.py`: Python wrapper for the Rust BPE tokenizer.
- `rustbpe/`: High-performance Rust BPE implementation.
- `scripts/`: Entry point scripts for training and evaluation.
  - `base_train.py`: Main training loop.
  - `tok_train.py`: Tokenizer training script.
  - `base_eval.py`: Evaluation utilities.
- `data/`: Local storage for checkpoints, tokenized data, and reports.
