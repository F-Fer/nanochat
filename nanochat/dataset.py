import os
from pyarrow.parquet import pq

from nanochat.common import get_base_dir

# The URL on the internet where the data is hosted and downloaded from on demand
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822 # the last datashard is shard_01822.parquet
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_dir")
os.makedirs(DATA_DIR, exist_ok=True)

def index_to_filename(index: int):
    assert index <= MAX_SHARD, f"The shard index must be less than {MAX_SHARD + 1}"
    return f"shard_{index:05d}.parquet"

# --------------------------------------------
# Helper functions for other modules

def list_parquet_files(data_dir=None):
    """Looks into the data_dir and returns the full path to all parquet files."""
    data_dir = data_dir if data_dir is not None else DATA_DIR
    parquet_files = []
    for f in os.listdir(data_dir):
        if f.endswith(".parquet") and not f.endswith(".tmp"):
            parquet_files.append(f)
    parquet_files = sorted(parquet_files)
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

def parquet_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups.
    - split can be "train" or "val". The last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) >= 2, f"At least 2 parquet files are needed, found {len(parquet_paths)}"
    if split == "train":
        parquet_paths = parquet_paths[:-1]
    else: 
        parquet_paths = parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

# --------------------------------------------
