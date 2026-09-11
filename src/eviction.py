"""Eviction-boundary rotation and query-side reconciliation.

Implements RFC §3.2 step 5 and RFC §3.1. Survivor keys are rotated once from
raw to logical position when the boundary advances, while the live query is
rotated by the negative eviction offset at attention time to preserve the same
relative distance without re-deriving the key-side state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.index_map import IndexMap
from src.rope import apply_rope
from src.shadow_cache import ShadowCache


@dataclass
class WindowState:
    window_size: int = 256
    evict_n: int = 64
    step: int = 0


class EvictionManager:
    def __init__(self, freqs: torch.Tensor):
        self.freqs = freqs

    def rotate_query(self, query: torch.Tensor, query_global_pos: int, eviction_delta: int) -> torch.Tensor:
        freqs = self.freqs.to(query.device)
        rotated = apply_rope(
            query.unsqueeze(0).unsqueeze(0),
            torch.tensor([float(query_global_pos - eviction_delta)], dtype=torch.float32, device=query.device),
            freqs,
        )
        return rotated.squeeze(0).squeeze(0)

    def on_boundary(self, shadow_cache: ShadowCache, index_map: IndexMap, window_state: WindowState):
        survivors = shadow_cache.get_survivors()
        if not survivors:
            return {'rotated': [], 'index_map': index_map.global_to_logical, 'window_state': window_state}

        compacted = index_map.compact([global_pos for _, _, global_pos in survivors])
        raw_keys = [raw_key for _, raw_key, _ in survivors]
        logical_positions = torch.tensor([
            float(compacted[global_pos])
            for _, _, global_pos in survivors
        ], dtype=torch.float32, device=raw_keys[0].device)

        batched_keys = torch.stack(raw_keys, dim=0)
        freqs = self.freqs.to(batched_keys.device)
        rotated_batched = apply_rope(batched_keys, logical_positions, freqs)

        rotated = []
        for idx, (token_id, raw_key, global_pos) in enumerate(survivors):
            logical_pos = compacted[global_pos]
            rotated.append({
                'token_id': token_id,
                'raw_key': raw_key,
                'global_pos': global_pos,
                'logical_pos': logical_pos,
                'key': rotated_batched[idx],
            })
        window_state.step += 1
        return {'rotated': rotated, 'index_map': compacted, 'window_state': window_state}
