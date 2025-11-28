import os
import argparse
import time

from nanochat.dataset import parquet_iter_batched
from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer

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
print(f"Training time: {(t1 - t0):.2f}s")

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
print(encoded)
decoded = tokenizer.decode(encoded)
assert decoded == test_text
