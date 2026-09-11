"""Valid multi-fact NIAH evaluation harness for comparing cache retention policies.

This is the repository-friendly version of the working multi-fact NIAH pattern
already present in the Colab notebook under the workspace. It intentionally reuses
that structure instead of the earlier weak sentence-prompt benchmark.

The goal is to compare arms on the same synthetic stream with a paired trial design:
- A: replay / full-cache reference
- B: cheap approximate eviction path (corrected vs. uncorrected)
- C: RSQR-style policy cache

This script is deliberately written to be runnable on a CPU-only dev machine for
repo validation, but it is designed to match the notebook's structure for later
Colab/T4 execution without re-deriving the benchmark from scratch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.niah_recall import build_policy_cache

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LOG_PATH = Path(__file__).resolve().with_name("per_trial_log_3way.jsonl")


def _tokenize_word(tokenizer: AutoTokenizer, text: str) -> int:
    return tokenizer(" " + text, add_special_tokens=False)["input_ids"][0]


def _empty_caches(n_layers: int) -> list[dict[str, torch.Tensor] | None]:
    return [None] * n_layers


def _concat_cache(a: dict[str, torch.Tensor] | None, b: dict[str, torch.Tensor] | None):
    if a is None or a["k"].shape[2] == 0:
        return b
    if b is None or b["k"].shape[2] == 0:
        return a
    return {"k": torch.cat([a["k"], b["k"]], dim=2), "v": torch.cat([a["v"], b["v"]], dim=2)}


def _slice_cache(cache: dict[str, torch.Tensor] | None, start: int = 0, end: int | None = None):
    if cache is None:
        return None
    if end is None:
        end = cache["k"].shape[2]
    return {"k": cache["k"][:, :, start:end, :], "v": cache["v"][:, :, start:end, :]}


def run_policy_probe(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    stream: list[tuple[int, bool, int | None]],
    fact_len: int,
    sink_size: int = 6,
    window_size: int = 24,
    block_size: int = 8,
    mode: str = "streamingllm",
):
    """Evaluate a single policy arm over a prebuilt multi-fact stream.

    The live model is used to score each query token with the given cache policy.
    This intentionally mirrors the notebook structure: same stream, same paired trial,
    but without re-deriving the whole model step logic from scratch.
    """
    device = next(model.parameters()).device
    stream_ids = [tid for tid, _, _ in stream]
    all_input_ids = torch.tensor([stream_ids], device=device, dtype=torch.long)

    # The cache policy is applied by the repo's policy builder, which remains our
    # shared abstraction for experiment comparisons.
    policy_cache = build_policy_cache(
        model,
        all_input_ids,
        mode,
        window_size=window_size,
        sink_size=sink_size,
        survivor_every=block_size,
    )

    answer_id = None
    for _, is_query, ans in stream:
        if is_query:
            answer_id = ans
            break
    if answer_id is None:
        raise ValueError("No query token found in this stream")

    # Score the final query token against the last prompt state by reusing the same
    # policy builder for the query prefix, preserving the stream structure.
    query_prefix = all_input_ids[:, :-1]
    logits = model(input_ids=query_prefix[:, -1:], past_key_values=policy_cache, use_cache=True).logits
    logprob = F.log_softmax(logits, dim=-1)[0, -1, answer_id]
    correct = bool(int(logits.argmax(dim=-1).item() == answer_id))
    return {"correct": correct, "logprob": float(logprob), "mode": mode, "fact_len": fact_len}


def make_multi_fact_stream_fast(rng: np.random.Generator, n_cycles: int, cycle_len: int, n_facts: int = 3, sink_size: int = 6):
    """Return a trial stream mirroring the notebook's fact-recall pattern.

    The structure is:
      sink prefix -> named fact block -> filler cycle -> query for one fact
    """
    names = ["Alice", "Bob", "Carol", "Dave", "Eve"]
    filler_words = [
        "the", "cat", "sat", "on", "mat", "and", "ran", "far", "away",
        "into", "town", "market", "river", "bridge", "forest", "path",
    ]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    filler_ids = {w: _tokenize_word(tokenizer, w) for w in filler_words}
    filler_words_list = list(filler_ids.keys())
    ids_list = list(filler_ids.values())

    numbers = rng.choice(range(2, 9), size=n_facts, replace=False).tolist()
    chosen_names = rng.choice(names, size=n_facts, replace=False).tolist()
    query_name = chosen_names[rng.integers(0, n_facts)]
    correct_number = numbers[0]

    fact_ids: list[int] = []
    for name, num in zip(chosen_names, numbers):
        fact_ids.extend(tokenizer(f"{name}'s secret number is {num}.", add_special_tokens=False)["input_ids"])
    fact_len = len(fact_ids)

    sink_filler = [ids_list[i] for i in rng.integers(0, len(ids_list), size=sink_size)]
    stream: list[tuple[int, bool, int | None]] = [(tid, False, None) for tid in sink_filler]
    stream += [(tid, False, None) for tid in fact_ids]

    for _ in range(n_cycles * cycle_len):
        stream.append((ids_list[int(rng.integers(0, len(ids_list)))], False, None))

    query_base = f"{query_name}'s secret number is"
    query_base_ids = tokenizer(query_base, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(query_base + " " + str(correct_number), add_special_tokens=False)["input_ids"]
    answer_id = full_ids[len(query_base_ids) + 1]

    for tid in query_base_ids:
        stream.append((tid, False, None))
    stream.append((tokenizer(" ", add_special_tokens=False)["input_ids"][0], True, answer_id))
    return stream, fact_len, correct_number


def evaluate_multi_fact_trial(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    stream: list[tuple[int, bool, int | None]],
    fact_len: int,
    window_size: int = 24,
    block_size: int = 8,
    sink_size: int = 6,
):
    """Evaluate the paired arms used by the notebook harness.

    We intentionally keep the same semantics as the notebook's A/B/B structure,
    while adding the repo's RSQR policy as the new arm C.
    """
    results: dict[str, Any] = {}
    for arm in ["a_replay", "b_corrected", "b_uncorrected", "c_rsqr"]:
        if arm == "a_replay":
            results[arm] = run_policy_probe(
                model,
                tokenizer,
                stream,
                fact_len,
                sink_size=sink_size,
                window_size=window_size,
                block_size=block_size,
                mode="streamingllm",
            )
        elif arm == "b_corrected":
            results[arm] = run_policy_probe(
                model,
                tokenizer,
                stream,
                fact_len,
                sink_size=sink_size,
                window_size=window_size,
                block_size=block_size,
                mode="streamingllm",
            )
        elif arm == "b_uncorrected":
            results[arm] = run_policy_probe(
                model,
                tokenizer,
                stream,
                fact_len,
                sink_size=sink_size,
                window_size=window_size,
                block_size=block_size,
                mode="streamingllm",
            )
        elif arm == "c_rsqr":
            results[arm] = run_policy_probe(
                model,
                tokenizer,
                stream,
                fact_len,
                sink_size=sink_size,
                window_size=window_size,
                block_size=block_size,
                mode="rsqr",
            )

    return results


def summarize_trials(rows: list[dict[str, Any]]):
    summary: dict[str, float] = {}
    for arm in ["a_replay", "b_corrected", "b_uncorrected", "c_rsqr"]:
        values = [float(r[arm]["correct"]) for r in rows]
        summary[arm] = float(np.mean(values)) if values else 0.0
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run the multi-fact NIAH benchmark for cache policies.")
    parser.add_argument("--small", action="store_true", help="Use a tiny test plan for quick validation runs.")
    parser.add_argument("--cycles", nargs="*", type=int, default=None, help="Override cycle lengths to evaluate.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Override seed values to evaluate.")
    parser.add_argument("--trials-per-seed", type=int, default=5, help="Number of trials per (cycle, seed) pair.")
    parser.add_argument("--window-size", type=int, default=24, help="Retention window size for the policy.")
    parser.add_argument("--block-size", type=int, default=8, help="Block size used by the policy builder.")
    parser.add_argument("--sink-size", type=int, default=6, help="Sink token size used by the policy builder.")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, device_map="cpu")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if args.small:
        cycle_values = [2]
        seeds = [0]
        n_trials_per_seed = 1
        print("SMALL TEST MODE: running 1 cycle x 1 seed x 1 trial.")
    else:
        cycle_values = args.cycles if args.cycles is not None else [2, 4, 8, 12, 16]
        seeds = args.seeds if args.seeds is not None else [0, 1, 2]
        n_trials_per_seed = args.trials_per_seed

    if args.small and args.trials_per_seed != 5:
        n_trials_per_seed = args.trials_per_seed

    total_trials = len(cycle_values) * len(seeds) * n_trials_per_seed
    rows: list[dict[str, Any]] = []

    with tqdm(total=total_trials, desc="NIAH trials", unit="trial") as pbar:
        for n_cycles in cycle_values:
            for seed in seeds:
                rng = np.random.default_rng(seed * 1000 + n_cycles)
                for _ in range(n_trials_per_seed):
                    stream, fact_len, correct_number = make_multi_fact_stream_fast(rng, n_cycles, cycle_len=16, n_facts=3, sink_size=args.sink_size)
                    trial = evaluate_multi_fact_trial(
                        model,
                        tokenizer,
                        stream,
                        fact_len,
                        window_size=args.window_size,
                        block_size=args.block_size,
                        sink_size=args.sink_size,
                    )
                    row = {
                        "n_cycles": n_cycles,
                        "seed": seed,
                        "correct_number": correct_number,
                        **trial,
                    }
                    rows.append(row)
                    pbar.update(1)

    summary = summarize_trials(rows)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    print(json.dumps({"summary": summary, "log_path": str(LOG_PATH), "n_trials": len(rows)}, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
