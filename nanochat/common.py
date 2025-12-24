import os
import re
import logging
import urllib.request
from dotenv import load_dotenv
import torch
import torch.distributed as dist

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    def format(self, record):
        # Add color to the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        # Format the message
        message = super().format(record)
        # Add color to specific parts of the message
        if levelname == 'INFO':
            # Highlight numbers and percentages
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
        return message

def setup_default_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler]
    )

setup_default_logging()
logger = logging.getLogger(__name__)

def get_base_dir():
    load_dotenv()
    # co-locate nanochat intermediates with other cached data in ~/.cache (by default)
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir

def download_file(url, filename, postprocess_fn=None):
    """
    Downloads a file from a URL to a local path in the base directory.
    """
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)

    if os.path.exists(file_path):
        return file_path
    
    # Download the content as bytes
    print(f"Downloading {url}...")
    with urllib.request.urlopen(url) as response:
        content = response.read() # bytes

    # Write to local file
    with open(file_path, 'wb') as f:
        f.write(content)
    print(f"Downloaded to {file_path}")

    # Run the postprocess function if provided
    if postprocess_fn is not None:
        postprocess_fn(file_path)

    return file_path

def autodetect_device_type():
    # prefer to use CUDA if available, otherwise use MPS, otherwise fallback on CPU
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    print(f"Autodetected device type: {device_type}")
    return device_type

def is_ddp():
    return int(os.environ.get('RANK', -1)) != -1

def get_dist_info():
    """
    Returns:
        ddp: bool
        ddp_rank: int
        ddp_local_rank: int
        ddp_world_size: int
    """
    if is_ddp():
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE']) # Must be set by torchrun
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1 # single GPU


def compute_init(device_type="cuda"):
    """
    Basic initialization
    Args:
        device_type = 'cuda' | 'cpu' | 'mps'
    Returns:
        ddp: bool
        ddp_rank: int
        ddp_local_rank: int
        ddp_world_size: int
        device: torch.device
    """
    assert device_type in ["cuda", "mps", "cpu"]
    if device_type == "cuda":
        assert torch.cuda.is_available(), "Your PyTorch installation is not configured for CUDA but device_type is 'cuda'"
    if device_type == "mps":
        assert torch.backends.mps.is_available(), "Your PyTorch installation is not configured for MPS but device_type is 'mps'"
    
    # Reproducability
    torch.manual_seed(42)

    # Precision
    if device_type == "cuda":
        torch.set_float32_matmul_precision("high")

    # Distributed setup: Distributed Data Parallel (DDP), optional, and requires CUDA
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)  # make "cuda" default to this device
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type) # mps/cpu
    
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device

def print0(s="",**kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

class DummyWandb:
    """Useful if we wish to not use wandb but have all the same signatures"""
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass

def compute_cleanup():
    """Companion function to compute_init, to clean things up before script exit"""
    if is_ddp():
        dist.destroy_process_group()