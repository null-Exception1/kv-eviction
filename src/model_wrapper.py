"""HF model wrapper for the raw-survivor eviction pipeline.

This is a lightweight adapter around a causal LM so KV writes and queries can be
routed through the raw-survivor cache path. The baseline mode keeps StreamingLLM-style
continuous key rotation; the RSQR mode uses the cached raw key plus query-side rotation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.eviction import EvictionManager, WindowState
from src.index_map import IndexMap
from src.rope import apply_rope, precompute_rope_freqs
from src.shadow_cache import ShadowCache


@dataclass
class ModelWrapperConfig:
    model_name: str = 'Qwen/Qwen2.5-0.5B-Instruct'
    torch_dtype: str = 'float32'
    window_size: int = 256
    survivor_every: int = 8
    mode: str = 'rsqr'
    precompute_ahead: bool = False


class StreamingLLMBaseline:
    def __init__(self, freqs: torch.Tensor):
        self.freqs = freqs

    def rotate_key(self, raw_key: torch.Tensor, pos: int):
        pos_tensor = torch.tensor([float(pos)], device=raw_key.device, dtype=torch.float32)
        return apply_rope(raw_key.unsqueeze(0).unsqueeze(0), pos_tensor, self.freqs).squeeze(0).squeeze(0)


class RSQRModelWrapper:
    def __init__(self, config: ModelWrapperConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=getattr(torch, config.torch_dtype),
            device_map='cpu',
        )
        self.freqs = precompute_rope_freqs(4096, 64, device=torch.device('cpu'))
        self.shadow_cache = ShadowCache(config.survivor_every)
        self.index_map = IndexMap()
        self.eviction_manager = EvictionManager(self.freqs)
        self.baseline = StreamingLLMBaseline(self.freqs)

    def score_query(self, query: torch.Tensor, query_global_pos: int, eviction_delta: int):
        return self.eviction_manager.rotate_query(query, query_global_pos, eviction_delta)

    def forward_with_policy(self, token_ids: torch.Tensor, query: torch.Tensor, query_global_pos: int):
        if self.config.mode == 'baseline':
            return self.baseline.rotate_key(query, query_global_pos)
        return self.score_query(query, query_global_pos, 8)

    def run_trial(self, token_ids: list[int], query: torch.Tensor, query_global_pos: int, eviction_delta: int | None = None):
        eviction_delta = self.config.survivor_every if eviction_delta is None else eviction_delta
        self.shadow_cache = ShadowCache(self.config.survivor_every)
        self.index_map = IndexMap()

        for idx, token_id in enumerate(token_ids):
            if idx % self.config.survivor_every == 0:
                raw = torch.randn(8, dtype=torch.float32)
                self.shadow_cache.add(token_id, raw, raw.clone(), is_survivor=True)

        result = self.eviction_manager.on_boundary(
            self.shadow_cache,
            self.index_map,
            WindowState(window_size=self.config.window_size, evict_n=eviction_delta),
        )

        rotated_query = self.score_query(query, query_global_pos, eviction_delta)
        scores = [float((rotated_query * entry['key']).sum().item()) for entry in result['rotated']]
        score = float(sum(scores) / len(scores)) if scores else 0.0
        return {'score': score, 'result': result}
