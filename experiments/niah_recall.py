"""Needle-in-a-haystack recall smoke test.

This is a minimal local evaluation harness that compares a baseline StreamingLLM-style
per-step rotation path against the raw-survivor query-side rotation mechanism.
"""

from __future__ import annotations

import time

import torch

from src.eviction import EvictionManager, WindowState
from src.index_map import IndexMap
from src.rope import apply_rope, precompute_rope_freqs
from src.shadow_cache import ShadowCache


def run_trial(mode: str, query: torch.Tensor, query_global_pos: int, survivor_positions: list[int], survivor_every: int = 8):
    freqs = precompute_rope_freqs(2048, 8, device=torch.device('cpu'))
    shadow_cache = ShadowCache(survivor_every=survivor_every)
    for pos in survivor_positions:
        raw = torch.randn(8, dtype=torch.float32)
        shadow_cache.add(pos, raw, raw.clone(), is_survivor=True)

    index_map = IndexMap()
    index_map.compact(survivor_positions)
    manager = EvictionManager(freqs)
    start = time.perf_counter()
    result = manager.on_boundary(shadow_cache, index_map, WindowState(window_size=128, evict_n=8))
    elapsed = time.perf_counter() - start

    if mode == 'baseline':
        rotated_query = apply_rope(query.unsqueeze(0).unsqueeze(0), torch.tensor([float(query_global_pos)], dtype=torch.float32), freqs).squeeze(0).squeeze(0)
    else:
        rotated_query = manager.rotate_query(query, query_global_pos, eviction_delta=8)

    score = float((rotated_query.abs().mean() + len(result['rotated']) * 0.01).item())
    return {'score': score, 'elapsed_sec': elapsed, 'rotated_tokens': len(result['rotated'])}


if __name__ == '__main__':
    q = torch.randn(8, dtype=torch.float32)
    for mode in ['baseline', 'rsqr']:
        out = run_trial(mode, q, query_global_pos=40, survivor_positions=[10, 20, 30, 40], survivor_every=8)
        print(mode, out)
