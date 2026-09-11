# Progress Log — Raw-Survivor Query-Side Rotation (RFC v4)

Assumes RFC v4 + blog origin post already in context. This logs findings since RFC v4's §5 (7 open questions) and next steps. Env: fp32, HF model, `apply_rope`/`FREQS`/`N_KV_HEADS`/`HEAD_DIM` already defined in notebook.

## Status of §5 open questions

**#1 batched vs continuous latency** — OPEN. RAP (arXiv 2602.02599): RoPE is <1% of inference latency even unfused; reframes question from "rotation FLOP cost" to "kernel-launch overhead." No latency measured yet.

**#2 bump severity vs window/Δ** — OPEN. Untouched. Needs real eviction loop (Phase 3+).

**#3 drift scaling past 10 iters** — CLOSED. 100-iter sweep at P=300, evict_n=8, 5 trials, fp32: diff **oscillates** in 8e-6–3e-5 band, no monotonic growth, no compounding. Matches RoPE angular periodicity, not accumulation. Contradicts original §2 framing of "accumulates error" — it's bounded, not growing. Caveat: only tested at evict_n=8; not re-verified at larger evict_n (e.g. 300) where floor may sit higher per #7.

**#4 memory accounting** — OPEN. No code written. Lead only: ShadowKV (2410.21465) — pre-RoPE keys compress 6x better than post-RoPE, unconfirmed compatibility with rotate-on-exit.

**#5 precompute-ahead saturation point** — OPEN. Blocked on new sub-question: CUDA graphs require static shapes; eviction batch size is variable by design (survivor density × Δ). Untested whether fixed-shape-bucket padding (vLLM-style) is viable. This should be resolved/tested *before* saturation-point measurement, not after.

**#6 attention-score sensitivity to magnitude floor** — CLOSED. Real K/Q, softmax vs 6 distractors, 10 trials at worst-case cell (P=450, evict_n=300, tensor diff ~1.8e-5 mean/4e-5 max). Result: softmax prob diff maxes at 1.28e-6, mean ~4e-7 — **same order as fp32 rounding noise** (diagonal P==evict_n cells sit at 2.4e-7–4.8e-7 purely from float noise). Effect does not survive into attention. Caveat: synthetic randn keys/queries, single step, no real model activations, no cross-layer/multi-step accumulation tested.

**#7 isolate P from evict_n** — CLOSED. Full P×evict_n grid (0–450 step 50, 5 draws/cell). Confirmed: **evict_n magnitude drives the floor, not P, not target position (P−evict_n)**. Proof: same |target_pos| gives wildly different diffs depending on P/evict_n split (e.g. |target|=100: 0.0 vs 1.49e-05). evict_n=0 always exactly 0.0 regardless of P. High variance at fixed evict_n (single-digit e-6 to 4e-5) — order of magnitude trustworthy, exact curve shape not yet.

**AnchorAttention scoping note (resolved):** confirmed via notebook (`torch_dtype=torch.float32`) that all sweeps are fp32. AnchorAttention's magnitude-floor mechanism is bf16-only (disappears under fp32 in their own ablation) — **does not transfer** to this RFC's floor. The §2.1/§5#6 floor is a separate, undocumented effect, not a confirmation of AnchorAttention.

## Net result: precision track (3/7) is CLOSED and clean

Findings support a *stronger* claim than RFC v4 currently makes: compounding drift is bounded (not growing), and even at its measured ceiling (~4e-5) it's invisible past softmax. The remaining 4 open questions (#1, #2, #4, #5) all require actual running code / real hardware — no longer answerable via standalone RoPE tensor sweeps.

## Roadmap status

- Phase 0 (dtype check, P/evict_n isolation) — DONE
- Phase 1 (drift extension, attention sensitivity) — DONE
- Phase 2 (memory accounting) — NOT STARTED
- Phase 3 (minimal working implementation + NIAH recall vs StreamingLLM) — NOT STARTED, starting now
- Phase 4 (latency: bump severity sweep, batched-vs-continuous bench, CUDA graph capturability test, precompute-ahead saturation) — NOT STARTED, needs cloud GPU (see hardware note)

## Hardware note

Local: GTX 1050 Ti (Pascal, 4GB) + i7. Sufficient for Phase 3 (correctness/mechanism work, small model e.g. TinyLlama-1.1B/Llama-3.2-1B, fp32, short-medium context) — mechanism correctness doesn't depend on model size or GPU class. NOT sufficient/comparable for Phase 4 (latency benchmarking, CUDA graph capture) — needs cloud GPU (A6000/A100-class) both for graph-capture maturity and to get numbers comparable to StreamingLLM's reported A6000 curve (31→65ms/token, cache 256→4096, Llama-2-7B).

