"""RoPE primitives and frequency precomputation.

Implements the standard rotary positional embedding used by causal LMs and the
raw-survivor RFC pipeline. This module is the canonical RoPE primitive used by
all downstream cache logic.
"""

from __future__ import annotations

import torch


def precompute_rope_freqs(max_position: int, head_dim: int, base: float = 10000.0, device: torch.device | None = None):
    device = torch.device('cpu') if device is None else device
    if head_dim % 2 != 0:
        raise ValueError('head_dim must be even for RoPE')
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    positions = torch.arange(max_position, dtype=torch.float32, device=device).unsqueeze(1)
    return positions * inv_freq.view(1, -1)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply standard RoPE to x using the provided precomputed frequencies.

    The positions tensor can be scalar, length-N, or broadcastable across the batch
    dimensions of x. This supports both the existing single-position use cases and
    the batched case where each survivor has a different logical position.
    """
    if x.shape[-1] % 2 != 0:
        raise ValueError('RoPE requires an even last dimension')

    x = x.to(device=freqs.device if freqs.device.type != 'cpu' else x.device)
    positions = positions.to(device=x.device, dtype=torch.long)
    if positions.ndim == 0:
        positions = positions.unsqueeze(0)

    angle = freqs[positions]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos = torch.cos(angle).to(dtype=x.dtype)
    sin = torch.sin(angle).to(dtype=x.dtype)

    rot_even = x_even * cos - x_odd * sin
    rot_odd = x_even * sin + x_odd * cos
    return torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)
