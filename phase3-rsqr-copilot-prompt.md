# Copilot prompt — Phase 3: RSQR implementation + evaluation notebook (Colab T4)

> Paste everything below into Copilot (Chat or agent mode) in the `auto-kv-cache-eviction` repo.

---

Build a new Colab notebook, `rsqr_phase3_eval.ipynb`, that implements and evaluates the
Raw-Survivor Query-Side Rotation (RSQR) mechanism described in
`rfc_raw_survivor_query_side_rotation_v4.md`, extending the existing three-way harness in
`qwen_multi_fact_eviction.ipynb`. Target runtime: Google Colab, **T4 GPU**, free tier. Keep peak
VRAM well under 16GB — the model stays `Qwen/Qwen2.5-0.5B-Instruct`, so this is comfortable, but
avoid gratuitous batch sizes or holding multiple duplicate model copies in memory at once.

## 0. Reuse, don't rewrite

Load and reuse as much of the existing harness as possible:
- Model/testbed setup: `Qwen/Qwen2.5-0.5B-Instruct`, GQA, 24 layers, 2 KV heads, head dim 64, `rope_theta=1e6`.
- The existing `apply_rope(x, positions, freqs)` reference implementation.
- The existing synthetic multi-fact recall task generator (sink tokens → 3 named facts → filler
  tokens triggering eviction cycles → query for one fact), and its trial/seed structure
  (`n_cycles ∈ {2,4,8,12,16}`, 50 trials × 3 seeds = 150 per point).
- The existing three arms as baselines: **A full replay**, **B renumber + re-rotate (key-side)**,
  **B leave-gap**.
- The existing `per_trial_log_3way.jsonl` format — extend it, don't replace it, so old and new
  rows stay comparable (add a `scheme` field value for the new arm(s) rather than changing the
  schema).

Do not re-derive the RoPE math or the task design from scratch; pull the existing cells forward
and build on top of them.

## 1. Implement the RSQR mechanism (RFC §3) as a fourth arm

Add a new eviction/correction scheme, **C: RSQR (query-side)**, implementing RFC §3.2's survivor
lifecycle exactly:

1. **In-window:** tokens rotate normally on arrival (key-side, as today).
2. **Survivor flagging:** every Δ-th token (config param, default matches the existing
   `block_size=8` cadence unless a sweep says otherwise) is flagged in advance. For flagged tokens
   only, store a **raw (unrotated) shadow copy** alongside the normal rotated in-window K.
3. **Window-exit:** when a flagged survivor exits the window, drop its rotated copy; keep only the
   raw copy plus its **global position** (needed for index-map compaction). It is never rotated
   again until step 5.
4. **Index-map compaction:** maintain a lightweight integer map from global position → compacted
   logical position for all retained survivors. No tensor data moves when this map updates.
5. **Query-time reconciliation:** at eviction-boundary events only (not every decode step), rotate
   each raw survivor once, directly from raw to its current logical position (single hop, never a
   correction-on-correction). Rotate the **incoming query**, not the keys, to align with the
   survivors' compacted timeline per RFC §3.1's query-side identity:
   `A = softmax((R_{m-evicted} Q_m) · (R_n K_n)^T / sqrt(d))`.

Implement this as its own function/class parallel to the existing B (re-rotate) and B (leave-gap)
implementations, sharing the same cache/window/block plumbing so it drops into the existing sweep
loop as a fourth column.

Also implement the **precompute-ahead variant** (RFC §3.4) as a config flag on the RSQR arm
(`precompute_ahead: bool`): when enabled, each raw block's correction for its *next* expected
eviction boundary is computed as soon as it's cheap to batch, rather than exactly at window-exit —
still always a single hop from that block's own raw baseline, never a correction on a prior
correction's output. Keep this toggleable so both variants can be benchmarked separately.

## 2. Accuracy evaluation — extend the existing sweep, plus the pinned-fact caveat

- Run the same `n_cycles ∈ {2,4,8,12,16}` sweep, same trial/seed count, with RSQR as arm C
  alongside the existing A/B/B. Append results to the existing table format (add columns, don't
  reshape).
- **Address the repo's flagged limitation directly:** add a second, separate sweep where the three
  named facts are *not* pinned outside the eviction-eligible region — i.e., a run where the facts
  themselves can be evicted like any other token, for all four arms. Report this as a clearly
  separate table (`fact_pinned=True` vs `fact_pinned=False`), since it's a materially different
  claim than the existing pinned-fact numbers.
- Log every trial to the extended `per_trial_log` (paired design: same random stream/seed scored
  by all arms per trial), specifically so paired significance testing is possible without re-running.
- Add a **paired significance test** (McNemar's test, or paired bootstrap if you prefer) between
  every pair of arms at every `n_cycles` point, run against the trial log, and print a summary
  table of p-values or CIs alongside the accuracy table. This was flagged as unresolved in the
  existing README and costs no additional GPU time since it only needs the existing per-trial log.

## 3. Precision evaluation — extend RFC §2 / §2.1 directly

Implement all of the following as separate, clearly-labeled cells, each printing its own
conclusion inline (don't just produce numbers, state what they mean relative to the RFC's open
questions):

- **§5 item 3 — drift-scaling sanity check:** extend the existing 10-iteration compounding test to
  a `range(1, 100)` sweep at fixed `P=300, evict_n=8`, logging `max_abs_diff` per iteration count.
  Plot iteration count vs. error and check by inspection/fit whether it's linear or accelerating.
- **§5 item 7 — isolate P from evict_n:** run two separate sweeps instead of the RFC's combined
  one — (a) fix `evict_n`, vary `P` over a wide range; (b) fix `P`, vary `evict_n` over a wide
  range — each for both 1-hop and 2-hop correction vs. a fresh-rotation control, so the error floor
  can be attributed to absolute position vs. correction size vs. target position specifically.
- **Production dtype check:** repeat the §2 and §2.1 tests in `bf16` and `fp16`, not just `float32`
  (flagged as unchecked in both the RFC and README). Report whether the error floor appears at
  smaller magnitudes than in float32, as the RFC predicts it might.
- **Real activations, not just `torch.randn`:** repeat at least the core §2 compounding test using
  real K-vectors pulled from an actual forward pass over the task's filler tokens, alongside the
  synthetic version, and report whether the two differ meaningfully.
- **§5 item 6 — attention-score sensitivity:** this is the load-bearing gap. Take K-tensor diffs in
  the 1e-5–1e-4 range (as produced by the §2.1 sweep at `P∈[800,1000]`) and measure whether they
  actually move softmax attention-score outputs or downstream recall accuracy at all, versus being
  invisible past that point. Structure this as: inject the measured drift into a real attention
  computation and compare softmax outputs with vs. without the perturbation, then separately check
  whether it correlates with any accuracy delta in the Section 2 sweep.
- Verify RSQR's own precision claim directly: confirm empirically that a survivor rotated via
  RSQR's raw→logical single hop matches a fresh direct rotation to the equivalent position to
  float32 machine-epsilon (i.e., reproduce the README §5-style check, `~4.77e-7`, but for the RSQR
  code path specifically, not just the underlying `apply_rope` primitive in isolation).

## 4. Latency evaluation — RFC §5 item 1, the "load-bearing unmeasured claim"

This is explicitly the most important open question in the RFC — prioritize it if Colab time
budget is tight.

- Implement a **continuous per-step rotation** baseline that matches StreamingLLM's reference
  behavior (rotate from raw at every decode step, `O(window_size)` per step) so there's a real
  comparison point, not just the reported literature curve.
- Measure per-token latency for: (a) the continuous baseline, (b) RSQR base (rotate only at
  eviction boundaries, batched), (c) RSQR precompute-ahead.
- Plot latency vs. cache size analogous to the cited StreamingLLM curve (cache size 256→4096) —
  even though this is a 0.5B model on a T4, not Llama-2-7B on an A6000, so state clearly that
  absolute numbers won't match the cited curve; the comparison that matters is the **shape**
  (constant/mild vs. batchy/spiky) and the aggregate compute, not absolute ms/token.
- Explicitly report **tail latency** (p95/p99 per-token time), not just mean — this is the
  variance-vs-reduction distinction the RFC draws in §3.5, and mean-only reporting would miss the
  whole point of this measurement.
- **§5 item 2 — bump severity vs. window size / survivor density:** sweep window size and Δ
  (survivor selection interval) together, not a single configuration, and report whether smaller
  windows (more frequent eviction boundaries) degrade tail latency enough to erode RSQR's
  aggregate-compute advantage.
- **§5 item 5 — precompute-ahead saturation point:** measure per-token correction cost against
  survivor batch size for the precompute-ahead variant, and report where the curve stops being
  overhead-dominated (small-batch, launch-cost-bound) and flattens toward linear-in-batch-size.
  Compare how much earlier precompute-ahead reaches that saturation point versus the passive base
  RSQR mechanism at the same config.

## 5. Memory accounting — RFC §5 item 4

- Quantify actual measured memory (not just asymptotic reasoning) for holding raw shadow copies of
  flagged survivors while still in-window, as a function of Δ and window size, for both RSQR
  variants.
- For precompute-ahead specifically, quantify the cost of holding a *chain* of precomputed
  correction states per flagged block versus the base mechanism's single raw copy, and check the
  RFC's claim that this is front-loaded-but-not-larger-in-aggregate against actual measured peak
  memory over a full session, not just at one point in time.

## 6. Structure, output, and reporting conventions

- Mirror the existing README's section numbering in the notebook's markdown headers (motivation →
  mechanism → precision → accuracy → latency → memory → open questions still remaining) so results
  map directly back to the RFC/README sections they address, for easy transcription into a Phase 3
  writeup afterward.
- Every results cell should end with a short markdown cell stating the finding in plain language
  and explicitly flagging remaining caveats — follow the existing repo's own tone of stating what
  is and isn't established, rather than overclaiming from a single run.
- Save all raw outputs (trial logs, latency traces, memory traces) as separate files (`.jsonl` /
  `.csv`) alongside the notebook, not just inline printouts, so they can be committed to the repo
  for the paired significance tests and any future re-analysis.
- Set and log all random seeds; keep the paired-per-trial design (same stream scored by every arm)
  used in the existing harness, extended to cover the new RSQR arm(s) too.
- Add a final markdown summary cell listing, for each of the RFC's §5 open questions, whether this
  notebook (a) fully resolved it, (b) partially addressed it with remaining caveats, or (c) left it
  open, and why — don't let results imply more certainty than the runs support.

## 7. Colab/T4 practicalities

- First cell: `!nvidia-smi` to confirm a T4 is attached, plus package installs pinned to versions
  compatible with the existing notebook's imports.
- Load the model once in fp16 (T4 has poor bf16 throughput; note this explicitly when running the
  bf16 precision-check cells from Section 3 — those can run in bf16 for correctness-of-math
  purposes even though fp16 is preferred for the actual inference/timing cells).
- Use `torch.cuda.synchronize()` around all latency timing to avoid measuring async-dispatch noise
  on a shared Colab GPU, and take medians over enough repeats to be robust to Colab's noisy-neighbor
  variance rather than single-shot timings.
- Add periodic `torch.cuda.empty_cache()` / explicit tensor deletion between sweep configurations
  so long sweeps don't OOM over a multi-hour Colab session; checkpoint intermediate results
  (trial log, latency CSV) to disk incrementally in case the runtime disconnects.
