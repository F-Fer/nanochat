import os
import argparse
import time

import torch

from nanochat.dataset import parquet_iter_batched
from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer
from nanochat.report import get_report

# Parse command line args
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
parser.add_argument('--max_chars', type=int, default=10_000_000_000, help='Maximum characters to train on (default: 10B)')
parser.add_argument('--doc_cap', type=int, default=10_000, help='Maximum characters per document (default: 10,000)')
parser.add_argument('--vocab_size', type=int, default=65536, help='Vocabulary size (default: 65536 = 2^16)')
args = parser.parse_args()
print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")

# -----------------------------------------------------------------------------
# Text iterator

def text_iterator():
    """
    1. Flatten the batches into a single iterator
    2. Crop every document to args.doc_cap chars
    3. Break when we have seen args.max_chars chars
    """
    n_chars = 0
    for batch in parquet_iter_batched(split="train"):
        for doc in batch:
            doc_text = doc
            if len(doc_text) > args.doc_cap:
                doc_text = doc_text[:args.doc_cap]
            n_chars += len(doc_text)
            yield doc_text
            if n_chars > args.max_chars:
                print(f"Processed {n_chars:,} chars")
                return
    print(f"Processed {n_chars:,} chars")

text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train tokenizer

t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
train_time = (t1 - t0)
print(f"Training time: {train_time:.2f}s")

# -----------------------------------------------------------------------------
# Save tokenizer to disk

base_dir = get_base_dir()
save_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(save_dir)

# -----------------------------------------------------------------------------
# Sanity check

test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text

# -----------------------------------------------------------------------------
# Create cache mapping from token id to number of bytes of that token for efficient 
# evaluation of bits per byte.
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[token_id]
    if token_str in special_set:
        token_bytes.append(0) # we dont include special tokens
    else:
        id_bytes = len(token_str.encode("utf-8")) # number of bytes taht make up this token
        token_bytes.append(id_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32)
token_bytes_path = os.path.join(save_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token bytes to {token_bytes_path}")

# Log to report
token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
get_report().log(section="Tokenizer training", data=[
    vars(args), # argparse command line arguments
    {"train_time": train_time},
    {"num_special_tokens": len(special_set)},
    {
        "token_bytes_min": int(token_bytes_nonzero.min().item()),
        "token_bytes_max": int(token_bytes_nonzero.max().item()),
        "token_bytes_mean": token_bytes_nonzero.mean().item(),
        "token_bytes_std": token_bytes_nonzero.std().item(),
    }
])