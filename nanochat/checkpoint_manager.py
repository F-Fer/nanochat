"""
Utilities for saving and loading model/optim/state checkpoints.
"""
import os
import json
import logging
import torch

from nanochat.common import setup_default_logging

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data):
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Save the model state parameters
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    torch.save(model_data, model_path)
    logger.info(f"Saved model parameters to: {model_path}")
    # Save the metadata dict as json
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)
    logger.info(f"Saved model metadata to {meta_path}")
    if optimizer_data is not None:
        os.makedirs(checkpoint_dir, exists_ok=True)
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}.pt")
        torch.save(optimizer_data, optimizer_path)
        logger.info(f"Saved optimizer state to {optimizer_path}")

def load_checkpoint(checkpoint_dir, step, device, load_optim=False):
    # Load the model state
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    # Load optimizer state if requested
    optimizer_data = None
    if load_optim:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)
    # Load metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data

