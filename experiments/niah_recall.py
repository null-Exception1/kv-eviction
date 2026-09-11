"""Real-text NIAH evaluation for RSQR vs StreamingLLM baseline.

This harness uses actual Qwen attention keys, not synthetic random vectors. It is
intended for a small Colab T4 run and logs one JSONL record per condition/depth.
"""

from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.eviction import EvictionManager, WindowState
from src.index_map import IndexMap
from src.rope import precompute_rope_freqs
from src.shadow_cache import ShadowCache
from src.model_wrapper import StreamingLLMBaseline


def capture_attention_k(model: AutoModelForCausalLM, input_ids: torch.Tensor, layer_index: int = -1):
    """Return the real K projection tensor for the full input from the selected attention layer."""
    layer = model.model.layers[layer_index]
    captured = {}

    def _hook(module, inputs, output):
        captured['k'] = output.detach().clone()

    handle = layer.self_attn.k_proj.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=True)
    finally:
        handle.remove()

    if 'k' not in captured:
        raise RuntimeError('Failed to capture a real K projection from the model attention layer.')

    return captured['k'][0]


def make_needle_prompt(context_length: int, needle_depth: int, needle_fact: str):
    facts = []
    for i in range(max(4, context_length)):
        if i == needle_depth:
            facts.append(f'Sentence {chr(65 + (i % 26))}: {needle_fact}')
        else:
            facts.append(f'Sentence {chr(65 + (i % 26))}: the answer is not the secret fact.')
    context = '\n'.join(facts[:context_length])
    prompt = (
        'You are given a short list of sentences. Return only the exact sentence that contains the secret phrase.\n'
        f'{context}\n'
        'Question: Which sentence contains the secret phrase? Reply with only that sentence.'
    )
    return prompt


def build_policy_cache(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    condition: str,
    window_size: int = 64,
    sink_size: int = 4,
    survivor_every: int = 8,
):
    """Return a policy-specific cache for the same prompt prefix.

    StreamingLLM keeps a fixed sink prefix plus the most recent tokens in the
    sliding window. RSQR keeps every survivor token via the raw-survivor index
    map, which is the same mechanism the rest of the repo uses.
    """
    with torch.no_grad():
        full_cache = model(input_ids=input_ids, use_cache=True).past_key_values
    cache = copy.deepcopy(full_cache)
    for layer in cache.layers:
        keys = layer.keys
        values = layer.values
        seq_len = keys.shape[-2]
        if condition == 'streamingllm':
            if seq_len <= sink_size + window_size:
                keep = list(range(seq_len))
            else:
                recent_start = max(sink_size, seq_len - window_size)
                keep = list(range(0, min(sink_size, seq_len))) + list(range(recent_start, seq_len))
        elif condition == 'rsqr':
            keep = list(range(0, seq_len, survivor_every))
            if not keep:
                keep = [0]
        else:
            raise ValueError(f'Unknown cache policy: {condition!r}')

        layer.keys = keys[:, :, keep, :]
        layer.values = values[:, :, keep, :]
    return cache


def generate_completion(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt_or_ids, max_new_tokens: int = 32, past_key_values=None):
    if isinstance(prompt_or_ids, str):
        inputs = tokenizer(prompt_or_ids, return_tensors='pt')
        prefix_ids = inputs['input_ids']
    else:
        prefix_ids = prompt_or_ids

    if past_key_values is None:
        inputs = tokenizer.decode(prefix_ids[0], skip_special_tokens=True) if isinstance(prefix_ids, torch.Tensor) else prompt_or_ids
        inputs = tokenizer(inputs, return_tensors='pt')
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return generated.strip()

    current_ids = prefix_ids[:, -1:]
    generated_tokens = []
    cache = past_key_values
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=current_ids, past_key_values=cache, use_cache=True)
        logits = outputs.logits[:, -1, :]
        next_token = int(logits.argmax(dim=-1).item())
        if next_token == tokenizer.eos_token_id:
            break
        generated_tokens.append(next_token)
        current_ids = torch.tensor([[next_token]], device=current_ids.device, dtype=current_ids.dtype)
        cache = outputs.past_key_values
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def evaluate_condition(model, tokenizer, condition: str, context_length: int, needle_depth: int, needle_fact: str, window_size: int = 64):
    prompt = make_needle_prompt(context_length, needle_depth, needle_fact)
    base_ids = tokenizer(prompt, return_tensors='pt')['input_ids']
    k_proj = capture_attention_k(model, base_ids)
    freqs = precompute_rope_freqs(2048, k_proj.shape[-1], device=torch.device('cpu'))

    policy_cache = build_policy_cache(model, base_ids, condition, window_size=window_size, survivor_every=8)
    completion = generate_completion(model, tokenizer, base_ids[:, -1:], max_new_tokens=32, past_key_values=policy_cache)

    if condition == 'streamingllm':
        baseline = StreamingLLMBaseline(freqs)
        _ = baseline.rotate_key(k_proj[-1], 0)
    else:
        shadow_cache = ShadowCache(survivor_every=8)
        selected_positions = list(range(0, min(len(k_proj), window_size), 8))
        for idx in selected_positions:
            real_key = k_proj[idx].clone()
            shadow_cache.add(token_id=idx, raw_key=real_key, rotated_key=real_key.clone(), is_survivor=True)

        compacted = IndexMap()
        compacted.compact(selected_positions)
        manager = EvictionManager(freqs)
        boundary = manager.on_boundary(
            shadow_cache,
            compacted,
            WindowState(window_size=window_size, evict_n=8),
        )
        _ = boundary['rotated']

    normalized = re.sub(r'\s+', ' ', completion).strip().lower()
    fact_normalized = re.sub(r'\s+', ' ', needle_fact).strip().lower()
    exact_match = 1.0 if fact_normalized in normalized else 0.0
    fuzzy_match = 1.0 if fact_normalized.split()[-1] in normalized else 0.0
    return {
        'condition': condition,
        'context_length': context_length,
        'needle_depth': needle_depth,
        'window_size': window_size,
        'exact_match': exact_match,
        'fuzzy_match': fuzzy_match,
        'generated_text': completion,
        'needle_fact': needle_fact,
    }


def run_suite(output_path: str | None = None, context_lengths: list[int] | None = None, depths: list[int] | None = None):
    model_name = 'Qwen/Qwen2.5-0.5B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map='cpu')
    model.eval()

    context_lengths = context_lengths or [64, 96]
    depths = depths or [0, 1, 2]
    needle_fact = 'The exact secret is BAKED-42.'
    rows = []

    for condition in ['streamingllm', 'rsqr']:
        for context_length in context_lengths:
            for needle_depth in depths:
                if needle_depth >= context_length:
                    continue
                row = evaluate_condition(model, tokenizer, condition, context_length, needle_depth, needle_fact)
                rows.append(row)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + '\n')

    return rows


if __name__ == '__main__':
    rows = run_suite(output_path='rsqr_niah_results.jsonl', context_lengths=[8, 12, 16], depths=[0, 1, 2])
    print(json.dumps(rows[0], sort_keys=True))
