
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768

def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))

def apply_rotary_emb(x, cos, sin):
    """
    Apply rotary embeddings to an input.
    x is of shape (batch_size, num_heads, seq_len, head_dim)
    cos and sin are of shape (1, 1, seq_len, head_dim//2)
    """
    assert x.ndim == 4 # Lets say x is of shape (1, 6, 1024, 128)
    d = x.shape[3] // 2 # Split x into two halves along the head_dim
    x1, x2 = x[..., :d], x[..., d:] # x1 and x2 would be (1, 6, 1024, 64)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3) # re-assemble to (1, 6, 1024, 128)
    return out

class CausalSelfAttention(nn.Module):
    pass
