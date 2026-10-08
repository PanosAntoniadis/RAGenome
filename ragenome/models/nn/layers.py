"""
Building blocks of the RAGenome network.

  * Rotary position embeddings (RoPE) with explicit positions, used by both attention layers.
  * Packed: layout of a batch of packed sequences.
  * LocalWindowAttention: intra-species local attention over packed sequences, implemented with FlashAttention.
  * LocalTransformer: a stack of pre-norm (local-attention, feed-forward) layers.
  * CrossAttention: inter-species cross-attention of the query over the packed retrieved sequences.
  * Level: local attention -> in_proj -> (nested block) -> out_proj (+ skip) -> local attention.
"""

from typing import NamedTuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.amp import autocast

try:
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func

    _FLASH_AVAILABLE = True
except ImportError:
    _FLASH_AVAILABLE = False


# Rotary embeddings, reduced from https://github.com/lucidrains/rotary-embedding-torch (MIT license)
# to what RAGenome uses: the position of every token is given explicitly (its alignment column) and
# the first `dim` features of every attention head are rotated.


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    return rearrange(torch.stack((-x2, x1), dim=-1), "... d r -> ... (d r)")


@autocast("cuda", enabled=False)
def apply_rotary_emb(freqs, t):
    """Rotate the first `freqs.shape[-1]` features of `t`; `freqs` broadcasts against them."""
    dtype = t.dtype
    rot_dim = freqs.shape[-1]
    t_rot, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t_rot = t_rot * freqs.cos() + rotate_half(t_rot) * freqs.sin()
    return torch.cat((t_rot, t_pass), dim=-1).type(dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000.0):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        # Fixed frequencies. Kept as a (non-trainable) Parameter so that the checkpoint keys
        # stay `...rotary_emb.freqs`.
        self.freqs = nn.Parameter(freqs, requires_grad=False)

    @autocast("cuda", enabled=False)
    def forward(self, positions):
        """positions: integer tensor of any shape -> rotation angles of shape (*positions.shape, dim)."""
        return (positions.float()[..., None] * self.freqs).repeat_interleave(2, dim=-1)


class Packed(NamedTuple):
    """
    Layout of a batch of packed sequences.
    [species_0 | ... | species_{N-1} | query], with the retrieved part padded at the tail.
    """

    pos_ids: Tensor  # (B, L) RoPE position of every token (alignment column)
    doc_ids: Tensor  # (B, L) index of the sequence a token belongs to, -1 = padding
    cu_seqlens: Tensor  # int32 boundaries of all sequences of the batch (FlashAttention varlen)
    max_seqlen: int  # length of the longest sequence
    n_retrieved: int  # length of the (padded) retrieved part of every row


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn: nn.Module, bias: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(dim, bias=bias)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x


class LocalWindowAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, rotary_emb_dim, window_size):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.rotary_emb = RotaryEmbedding(rotary_emb_dim)
        self.window_size = window_size

    def forward(self, x, packed: Packed):
        assert _FLASH_AVAILABLE, "flash_attn is required (pip install flash-attn)"
        B, L, D = x.shape

        # Gather real tokens before projecting/rotating so that all matmuls run only on
        # real tokens.
        flat_mask = (packed.doc_ids >= 0).view(-1)
        x_flat = x.reshape(B * L, D)[flat_mask]

        q, (k, v) = self.to_q(x_flat), self.to_kv(x_flat).chunk(2, dim=-1)
        q, k, v = (rearrange(t, "n (h d) -> n h d", h=self.heads) for t in (q, k, v))

        freqs = self.rotary_emb(packed.pos_ids.reshape(-1)[flat_mask]).unsqueeze(1)
        q, k = apply_rotary_emb(freqs, q), apply_rotary_emb(freqs, k)

        # Drop zero-length segments from cu_seqlens.
        cu_seqlens = torch.unique_consecutive(packed.cu_seqlens)
        half_w = self.window_size // 2

        out = _flash_attn_varlen_func(
            q, k, v,
            cu_seqlens, cu_seqlens,
            packed.max_seqlen, packed.max_seqlen,
            window_size=(half_w, half_w),
        )  # (total_real, H, D_h)
        out = self.to_out(out.reshape(-1, self.heads * self.dim_head))

        out_buf = torch.zeros(B * L, D, device=x.device, dtype=out.dtype)
        out_buf[flat_mask] = out
        return out_buf.view(B, L, D)


def FeedForward(dim, mult=4):
    return nn.Sequential(
        nn.Linear(dim, dim * mult, bias=False),
        nn.GELU(),
        nn.Identity(),
        nn.Linear(dim * mult, dim, bias=False),
    )


class LocalTransformer(nn.Module):
    def __init__(self, dim, *, depth, sliding_window, heads, dim_head, rotary_emb_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNormResidual(
                    dim,
                    LocalWindowAttention(dim, heads, dim_head, rotary_emb_dim, sliding_window),
                    bias=False,
                ),
                PreNormResidual(dim, FeedForward(dim)),
            ])
            for _ in range(depth)
        ])

    def forward(self, x, packed: Packed):
        for attn, ff in self.layers:
            x = attn(x, packed=packed)
            x = ff(x)
        return x


class CrossAttention(nn.Module):
    """
    Inter-species cross-attention block: the query species attends to the packed retrieved
    sequences of the other species.
    """

    def __init__(self, dim: int, heads: int, dim_head: int, depth: int, rotary_emb_dim: int):
        super().__init__()
        self.depth = depth
        self.heads = heads
        inner_dim = heads * dim_head

        self.to_q   = nn.ModuleList([nn.Linear(dim, inner_dim, bias=False) for _ in range(depth)])
        self.to_k   = nn.ModuleList([nn.Linear(dim, inner_dim, bias=False) for _ in range(depth)])
        self.to_v   = nn.ModuleList([nn.Linear(dim, inner_dim, bias=False) for _ in range(depth)])
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, dim, bias=False) for _ in range(depth)])
        self.norm   = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])

        self.rotary_emb = RotaryEmbedding(rotary_emb_dim)

    def forward(self, x: torch.Tensor, packed: Packed) -> torch.Tensor:
        assert _FLASH_AVAILABLE, "flash_attn is required (pip install flash-attn)"
        n_kv = packed.n_retrieved
        kv, query = x[:, :n_kv], x[:, n_kv:]  # (B, n_kv, D) packed retrieved tokens, (B, L, D)
        B, L, _ = query.shape

        # Real retrieved tokens of every item. Items without any retrieved token cannot 
        # go through FlashAttention: they get no cross-attention update.
        kv_real = packed.doc_ids[:, :n_kv] >= 0
        kv_lengths = kv_real.sum(dim=1)
        nonempty = kv_lengths > 0
        n_active = int(nonempty.sum())

        if n_active:
            # K/V never change across depth iterations (only the query is updated), so the real
            # KV tokens and their rotary angles are gathered once.
            kv_flat = kv[kv_real]  # (total_real_kv, D)
            k_freqs = self.rotary_emb(packed.pos_ids[:, :n_kv][kv_real]).unsqueeze(1)
            q_freqs = self.rotary_emb(torch.arange(L, device=x.device)).unsqueeze(1)  # (L, 1, r)
            cu_seqlens_q = torch.arange(n_active + 1, device=x.device, dtype=torch.int32) * L
            cu_seqlens_k = F.pad(kv_lengths[nonempty].cumsum(0), (1, 0)).to(torch.int32)
            max_seqlen_k = int(kv_lengths.max())

        for i in range(self.depth):
            out = torch.zeros_like(query)
            if n_active:
                q = rearrange(self.to_q[i](query[nonempty]), "b l (h d) -> b l h d", h=self.heads)
                k = rearrange(self.to_k[i](kv_flat), "n (h d) -> n h d", h=self.heads)
                v = rearrange(self.to_v[i](kv_flat), "n (h d) -> n h d", h=self.heads)
                q, k = apply_rotary_emb(q_freqs, q), apply_rotary_emb(k_freqs, k)

                attn = _flash_attn_varlen_func(
                    q.reshape(n_active * L, *q.shape[2:]), k, v,
                    cu_seqlens_q, cu_seqlens_k, L, max_seqlen_k,
                )  # (n_active * L, H, Dh)
                out[nonempty] = self.to_out[i](attn.reshape(n_active, L, -1)).to(out.dtype)
            query = self.norm[i](query + out)
        return torch.cat([kv, query], dim=1)


class Level(nn.Module):
    """
    local attention -> in_proj -> valley -> out_proj (+ skip connection) -> local attention
    (intra-species local attention; the valley holds the inter-species cross-attention).

    The valley is either the `CrossAttention` block or another `Level`.
    """

    def __init__(self, dim, local_depth, valley, norm_out, **attn_kwargs):
        super().__init__()
        self.pre_transformer = LocalTransformer(dim, depth=local_depth, **attn_kwargs)
        self.post_transformer = LocalTransformer(dim, depth=local_depth, **attn_kwargs)
        self.in_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.valley = valley
        self.norm_out = nn.LayerNorm(dim) if norm_out else nn.Identity()

    def forward(self, x: torch.Tensor, packed: Packed) -> torch.Tensor:
        x = self.pre_transformer(x, packed)
        x = self.out_proj(self.valley(self.in_proj(x), packed)) + x
        return self.norm_out(self.post_transformer(x, packed))
