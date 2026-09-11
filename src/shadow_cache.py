"""Raw-shadow survivor bookkeeping.

Implements RFC §3.2 steps 1-2: flagged survivors carry a raw (unrotated) key
alongside their regular in-window rotated key, while non-survivors remain
ordinary window entries and are evicted without a shadow copy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ShadowEntry:
    token_id: int
    raw_key: torch.Tensor
    rotated_key: torch.Tensor
    is_survivor: bool
    global_position: int


class ShadowCache:
    def __init__(self, survivor_every: int = 8):
        self.survivor_every = survivor_every
        self.entries: list[ShadowEntry] = []
        self._next_global_position = 0

    def add(self, token_id: int, raw_key: torch.Tensor, rotated_key: torch.Tensor, is_survivor: bool):
        global_position = self._next_global_position
        self._next_global_position += 1
        self.entries.append(
            ShadowEntry(
                token_id=int(token_id),
                raw_key=raw_key.detach().clone(),
                rotated_key=rotated_key.detach().clone(),
                is_survivor=bool(is_survivor),
                global_position=global_position,
            )
        )

    def get_survivors(self) -> list[tuple[int, torch.Tensor, int]]:
        return [
            (entry.token_id, entry.raw_key, entry.global_position)
            for entry in self.entries
            if entry.is_survivor
        ]
