import torch

from src.eviction import EvictionManager, WindowState
from src.index_map import IndexMap
from src.rope import apply_rope, precompute_rope_freqs
from src.shadow_cache import ShadowCache


def test_eviction_path_matches_fresh_rotation_for_survivor():
    freqs = precompute_rope_freqs(2048, 8, device=torch.device('cpu'))
    raw_key = torch.randn(8, dtype=torch.float32)
    shadow_cache = ShadowCache(survivor_every=8)
    shadow_cache.add(token_id=100, raw_key=raw_key, rotated_key=raw_key.clone(), is_survivor=True)

    index_map = IndexMap()
    index_map.compact([100])
    manager = EvictionManager(freqs)
    boundary_out = manager.on_boundary(shadow_cache, index_map, WindowState(window_size=128, evict_n=8))
    actual = boundary_out['rotated'][0]['key']

    expected = apply_rope(raw_key.unsqueeze(0).unsqueeze(0), torch.tensor([0.0], dtype=torch.float32), freqs).squeeze(0).squeeze(0)
    max_abs_diff = (actual - expected).abs().max().item()
    assert max_abs_diff < 1e-6, max_abs_diff


def test_query_rotation_uses_negative_eviction_offset():
    freqs = precompute_rope_freqs(2048, 8, device=torch.device('cpu'))
    query = torch.randn(8, dtype=torch.float32)
    manager = EvictionManager(freqs)
    rotated = manager.rotate_query(query, query_global_pos=40, eviction_delta=8)
    expected = apply_rope(query.unsqueeze(0).unsqueeze(0), torch.tensor([32.0], dtype=torch.float32), freqs).squeeze(0).squeeze(0)
    assert (rotated - expected).abs().max().item() < 1e-6


def test_eviction_pipeline_matches_direct_rope_across_positions_and_compactions():
    freqs = precompute_rope_freqs(2048, 8, device=torch.device('cpu'))
    survivor_sets = [
        [100, 108, 116, 124],
        [200, 208, 216, 224],
        [400, 408, 416, 424],
    ]

    for survivor_positions in survivor_sets:
        shadow_cache = ShadowCache(survivor_every=8)
        for token_id in survivor_positions:
            raw_key = torch.randn(8, dtype=torch.float32)
            shadow_cache.add(token_id=token_id, raw_key=raw_key, rotated_key=raw_key.clone(), is_survivor=True)

        index_map = IndexMap()
        index_map.compact(survivor_positions)
        manager = EvictionManager(freqs)
        boundary = manager.on_boundary(
            shadow_cache,
            index_map,
            WindowState(window_size=256, evict_n=8),
        )

        for entry in boundary['rotated']:
            logical_pos = entry['logical_pos']
            raw_key = entry['raw_key']
            expected = apply_rope(
                raw_key.unsqueeze(0).unsqueeze(0),
                torch.tensor([float(logical_pos)], dtype=torch.float32),
                freqs,
            ).squeeze(0).squeeze(0)
            actual = entry['key']
            max_abs_diff = (actual - expected).abs().max().item()
            assert max_abs_diff < 1e-6, (survivor_positions, logical_pos, max_abs_diff)
