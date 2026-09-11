"""Logical-position compaction for surviving tokens.

Implements RFC §3.2 step 4: after eviction, surviving global positions are
compacted into a contiguous logical index space with only integer bookkeeping.
"""

from __future__ import annotations


class IndexMap:
    def __init__(self):
        self.global_to_logical: dict[int, int] = {}

    def compact(self, survivor_global_positions: list[int]) -> dict[int, int]:
        sorted_positions = sorted(int(pos) for pos in survivor_global_positions)
        mapping = {global_pos: logical_pos for logical_pos, global_pos in enumerate(sorted_positions)}
        self.global_to_logical = mapping
        return mapping

    def logical_position(self, global_position: int) -> int:
        return self.global_to_logical[int(global_position)]
