import torch

from src.eviction import EvictionManager, WindowState
from src.index_map import IndexMap
from src.rope import apply_rope, precompute_rope_freqs
from src.shadow_cache import ShadowCache


def test_eviction_manager_matches_fresh_rotation_for_single_survivor():
    freqs = precompute_rope_freqs(1024, 8, device=torch.device('cpu'))
    raw_key = torch.randn(8, dtype=torch.float32)
    shadow_cache = ShadowCache(survivor_every=8)
    shadow_cache.add(token_id=120, raw_key=raw_key, rotated_key=raw_key.clone(), is_survivor=True)

    index_map = IndexMap()
    index_map.compact([120])
    manager = EvictionManager(freqs)
    result = manager.on_boundary(shadow_cache, index_map, WindowState(window_size=64, evict_n=8))
    actual = result['rotated'][0]['key']
    expected = apply_rope(raw_key.unsqueeze(0).unsqueeze(0), torch.tensor([0.0], dtype=torch.float32), freqs).squeeze(0).squeeze(0)

    max_abs_diff = (actual - expected).abs().max().item()
    assert max_abs_diff < 1e-5, max_abs_diff


def test_query_side_rotation_is_not_key_side_renumbering():
    freqs = precompute_rope_freqs(1024, 8, device=torch.device('cpu'))
    query = torch.randn(8, dtype=torch.float32)
    manager = EvictionManager(freqs)
    rotated = manager.rotate_query(query, query_global_pos=42, eviction_delta=12)
    expected = apply_rope(query.unsqueeze(0).unsqueeze(0), torch.tensor([30.0], dtype=torch.float32), freqs).squeeze(0).squeeze(0)

    assert (rotated - expected).abs().max().item() < 1e-6


def test_end_to_end_trial_produces_non_hardcoded_score():
    freqs = precompute_rope_freqs(1024, 8, device=torch.device('cpu'))
    manager = EvictionManager(freqs)
    shadow_cache = ShadowCache(survivor_every=8)
    for token_id in range(0, 32, 8):
        raw = torch.randn(8, dtype=torch.float32)
        shadow_cache.add(token_id, raw, raw.clone(), is_survivor=True)
    index_map = IndexMap()
    index_map.compact([0, 8, 16, 24])
    boundary = manager.on_boundary(shadow_cache, index_map, WindowState(window_size=64, evict_n=8))
    query = torch.randn(8, dtype=torch.float32)
    rotated_query = manager.rotate_query(query, query_global_pos=40, eviction_delta=8)
    scores = [float((rotated_query * entry['key']).sum().item()) for entry in boundary['rotated']]
    score = float(sum(scores) / len(scores))
    assert abs(score) > 0.0
