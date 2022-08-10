import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from torch.nn import functional as F


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_pdrop: float = 0.0,
        resid_pdrop: float = 0.0,
        block_size: int = 1024,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        # output projection
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (embed_dim)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.embed_dim, dim=2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    """a Transformer block"""

    def __init__(
        self, embed_dim: int, num_heads: int, resid_pdrop: float, attn_pdrop: float, block_size: int
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            block_size=block_size,
        )
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(embed_dim, 4 * embed_dim),
                c_proj=nn.Linear(4 * embed_dim, embed_dim),
                act=NewGELU(),
                dropout=nn.Dropout(resid_pdrop),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))  # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x
