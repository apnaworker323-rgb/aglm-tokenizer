# AGLM Architecture Optimization Report — Synthesis

Status: Phases 1, 2, 3, 10 **complete** (real GPU measurements, 100 stable steps each unless noted). Phase 5
**partial** (backbone-only, isolated from the output head). Phases 4, 6, 8, 9, 12, 13 **not run** — see
"What was not done" below. No architecture change has been made to production. No long training was started.

This document answers the ten questions the research brief asked the final report to answer, each grounded in
`AGLM_GPU_COMPONENT_PROFILE.md`, `AGLM_OUTPUT_HEAD_AUDIT.md`, and `AGLM_PATCH2_BOTTLENECK_REPORT.md`.

---

## 1. Why does 2× global-position compression not currently increase GPU throughput?

Because the component patching acts on (the Transformer body + local codec) is **~5–8% of step time**, while
the component patching cannot touch (the output head) is **~70–78%**. Measured directly by instrumenting
`HierarchicalLatentLM` at the same four stage boundaries for Patch-1 and Patch-2: `global_backbone` saved
0.153ms, `local_codec` saved 0.225ms, `output_head` was unchanged (16.44 vs 16.41ms, noise-level) because it
still scores every original token regardless of patch size. Full detail: `AGLM_PATCH2_BOTTLENECK_REPORT.md`.

## 2. What component consumes the largest fraction of training time?

The output head — `nn.AdaptiveLogSoftmaxWithLoss` over the full 1,551,017-class vocabulary — at **70.0%** of
step time (CUDA-event instrumentation, N=100) and **77.7%** of step time when isolated completely from the
backbone (Phase 2 ablation, independent methodology). The optimizer step is second at **17.9%**, itself mostly
a tax on the same two vocabulary-sized parameter tables (input embedding 73.1% of params, output head 24.3% of
params). Attention + FFN + normalization + embedding lookup — "the Transformer" — is **under 2.5% combined**.
Full table: `AGLM_GPU_COMPONENT_PROFILE.md`.

## 3. How much faster could training theoretically become if that bottleneck disappeared?

Two independent measured ceilings:

- **Backbone-only upper bound** (real embedding+backbone, output/loss replaced by the cheapest valid dummy
  objective, profiling only): **4.09× the full model's tok/s** (42,000.5 vs 10,270.0 tok/s).
- **A measured, not just theoretical, partial fix** (exact frequency-rank permutation applied only at the
  output-head boundary, architecture and vocabulary completely unchanged): **2.39× the full model's tok/s**
  (23,428.9 vs 9,795.7 tok/s), with peak VRAM cut 45% (2,724 → 1,486 MiB) as a side effect.

  The gap between 2.39× (achieved via calibration alone) and 4.09× (the absolute backbone ceiling) is the
  remaining headroom if the output head's *implementation* — not just its cutoff calibration — were also
  optimized (fused kernels, fewer clusters, avoiding the `.nonzero()`-driven gather/scatter pattern). Not
  pursued this session; see "Not done."

## 4. Is the 1.55M output head the limiting factor?

**Yes — but the limiting factor is a fixable calibration bug in how it's configured, not an inherent property
of needing 1.55M classes.** The adaptive-softmax cutoffs `(16384, 131072, 524288)` implicitly assume a
frequency-sorted vocabulary. AGLM's ID space is allocated by structural/lexical category, not frequency
(correlation between token ID and measured corpus frequency: **−0.0145**, statistically zero; only 3.5% overlap
between the true top-16,384-by-frequency tokens and the current head cluster's ID range). Consequently only
30.9% of real training tokens land in the cheap head cluster — a correctly-calibrated head cluster of the same
size would capture 88.7%. An exact, lossless, output-side frequency permutation (tokenizer and input IDs
completely untouched) recovers 2.39× throughput end-to-end. Full derivation, including a self-caught analysis
bug from a `uint64` negation overflow that had to be fixed before trusting the frequency numbers: see
`AGLM_OUTPUT_HEAD_AUDIT.md`.

## 5. Does attention become dominant at longer contexts?

**Not observed at production seq_len (512), and not yet directly observable in the full model at longer
context, because the current unfixed output head runs out of memory first.** Two findings:

- The real output head OOMs by seq_len=2048 (batch=1, 2,048 total tokens/step) — inside a tail-cluster
  `log_softmax` call — well before attention's O(n²) term would plausibly matter for a 4-layer, d_model=192
  model. On this 6 GiB GPU, the output head is a hard blocker to even *measuring* long-context behavior in the
  current unfixed configuration, not just a speed tax at seq_len=512.
- Isolating the backbone alone (dummy loss, no output head — same technique as the Phase 2 ablation) across
  seq_len 512→8192 shows attention's cost is negligible at 512 but visibly super-linear by 4096–8192:

  | seq_len | batch | backbone fwd ms | backbone bwd ms | peak VRAM MiB |
  |---:|---:|---:|---:|---:|
  | 512 | 2 | 3.492 | 6.760 | 867.6 |
  | 1024 | 2 | 3.549 | 8.067 | 867.6 |
  | 2048 | 2 | 6.450 | 15.237 | 868.6 |
  | 4096 | 1 | 7.530 | 17.156 | 868.6 |
  | 8192 | 1 | 17.952 | 41.695 | 1,046.6 |

  Note the 2048(batch=2) vs 4096(batch=1) pair: identical total token count (4,096) per step, but the *longer*
  sequence shape costs more (7.53ms vs 6.45ms forward) — the signature of attention's per-token cost growing
  with sequence length even at fixed total-token throughput, not just fixed batch size.

Bottom line: attention is not dominant today, and the backbone-only scan shows it *will* become a real cost at
longer contexts on this model — but the honest answer to "at what length does it overtake the output head in
the real model" is **not yet measurable** until the output head's OOM ceiling is pushed back (i.e., after the
frequency-remap fix, which also cuts output-head VRAM by 45%). This is now the top follow-up experiment.

## 6. Should we optimize Transformer, Mamba, FFN, or output head first?

**Output head first, unambiguously.** This is Case A from the brief's own decision tree: output/loss consumes
the large majority of step time. Per the brief's explicit instruction, Mamba/FFN/attention optimization was not
pursued — there is nothing to gain there yet that would be visible above the output head's cost. The measured
2.39× fix (frequency remap) is also lower-risk than any backbone architecture change: it touches zero model
parameters, zero tokenizer state, and is a pure bijective relabeling.

## 7. Does Patch-2 deserve further development?

**Not right now, and not for the reason originally suspected.** Patch-2's mechanism works exactly as designed
(exact codec, preserved sampled BPB, genuinely halved global positions) — it simply optimizes a part of the
step (backbone + local codec, ~5–8%) that is not currently the bottleneck. It is not "useless"; it is
**premature**. Recommend revisiting Patch-2 (and the sequence-length question in §5) only after the output-head
fix ships, at which point the backbone's relative share of a much smaller step roughly doubles and patching has
something real to save.

## 8. Which architecture gives the best raw MiB/s at acceptable BPB?

**Cannot fully answer yet — BPB has not been measured for any candidate this session.** Every run in this
report uses freshly-initialized weights over ~100 steps; loss values reported are not converged and are not
valid BPB comparisons (explicitly flagged in `AGLM_OUTPUT_HEAD_AUDIT.md` rather than fabricated). What can be
said: the frequency-remapped output head is mathematically an exact relabeling of the identical 1,551,017-way
categorical distribution the baseline already computes, so it should be BPB-neutral in principle — that
expectation is untested empirically and is the single most important next experiment (Phase 13, not run).

## 9. What is the projected 20TB wall-clock for each finalist?

**Linear projections from this compact prototype's measured raw MiB/s. Explicitly not a promise about any
future model size, GPU, or multi-GPU setup** — `projected_20TB_days = 20e12 / (raw_MiB/s × 2^20) / 86400`.

| candidate | raw MiB/s | tok/s | peak VRAM MiB | projected 20TB days | (years) |
|---|---:|---:|---:|---:|---:|
| production benchmark report (original, mmap) | 0.0430 | 9,260.2 | 2,723.5 | 5,133.9 | 14.06 |
| Phase 1 rerun baseline (this session, uncalibrated cutoffs) | 0.0459 | 9,795.7 | 2,724.4 | 4,808.8 | 13.17 |
| **frequency-remapped output head (measured, exact, same params)** | **0.1098** | **23,428.9** | **1,485.9** | **2,010.6** | **5.50** |
| backbone-only (dummy loss; theoretical ceiling, not a real candidate) | 0.1968 | 42,000.5 | 866.8 | 1,121.6 | 3.07 |

The frequency-remap fix alone moves the projection from ~13.2 years to ~5.5 years on this single 6GB GPU — still
obviously impractical in absolute terms (this is a 67.9M-parameter research prototype on one low-end consumer
card, not a production training setup), but a 2.39× reduction in a linear-projection metric is a 2.39× reduction
regardless of the absolute scale it's measured at. The remaining gap to the 3.07-year backbone ceiling is the
size of the prize left if the output head's implementation (not just calibration) were also optimized.

## 10. What experiment should be run next?

In priority order:

1. **Validate the frequency-remap fix for quality, not just speed.** Train to at least a short-but-real
   convergence point with and without the remap, compare held-out BPB. This is the single highest-priority gate
   — everything else in this report assumes (correctly, by construction of an exact bijection, but unverified
   empirically) that it is quality-neutral. Wire up the inverse permutation for inference while at it.
2. **Re-run the seq_len scan (§5) and the Patch-1 vs Patch-2 comparison (Phase 10) with the fixed output
   head.** Both are currently bottlenecked by the same OOM ceiling / dominant-cost component; both answers
   should change once it's gone.
3. **Phase 6 — tokenizer comparison** (byte-baseline, a representative ~200K tokenizer if available, AGLM
   production vs. exact-minimum segmentation), on RAW UTF-8 bytes/second, not tokens/second. Not started this
   session.
4. **Only then** — Phase 8/9 hybrid Transformer+Mamba search. Per the brief's own instruction, this was
   deliberately not pursued yet: at seq_len=512 the backbone is 2.5% of step time, so no recurrent-memory
   architecture change would be visible above the noise floor of the current bottleneck. Revisit once §5's
   real crossover point is known.

---

## What was not done (explicit, per the brief's request for honesty about scope)

- **Phase 4** (fine-grained Transformer body audit — GQA/FlashAttention/SwiGLU/RMSNorm individually): superseded
  by the finding that the whole backbone is under 2.5% of step time; not worth the additional instrumentation
  effort until §10.4 is reached.
- **Phase 6** (tokenizer comparison, raw bytes/second across byte-baseline / ~200K tokenizer / AGLM production /
  AGLM exact-minimum segmentation): not started.
- **Phase 8/9** (T0–T7 Transformer+Mamba hybrid screening, Mamba kernel optimization): not started, deliberately,
  per Case A prioritization.
- **Phase 12** (full Pareto frontier across finalists): only two real candidates exist so far (baseline,
  frequency-remapped); see `AGLM_20TB_PARETO_REPORT.md` for what a 2-point preliminary comparison shows and what
  is missing to build the real frontier.
- **Phase 13** (quality safety battery: held-out BPB, exact copy, numbers, UUIDs, URLs, source code, rare
  tokens, multilingual/Romanized Indic, passkey retrieval, associative recall, induction): not run. Requires a
  trained-to-some-real-checkpoint model, not freshly-initialized weights; out of scope for a pure profiling
  session. This is item 1 in the priority list above for a reason — it gates whether the headline 2.39× result
  is usable at all.
- **Sparse input-embedding optimizer path** (flagged in `AGLM_GPU_COMPONENT_PROFILE.md`, ~72% of the 17.9%
  optimizer-step cost): identified and quantified via a synthetic isolation, not implemented or measured
  end-to-end.

## Raw artifacts index

- Phase 1: `benchmark_results/phase1/`
- Phase 2: `benchmark_results/phase2/`
- Phase 3: `benchmark_results/phase3/`
- Phase 5 (partial): `benchmark_results/phase5/`
- Phase 10: `benchmark_results/phase10/`
- Scripts (all instrumentation-only, frozen architecture, no checkpoints written): `profile_aglm_gpu_components.py`,
  `profile_aglm_ablations.py`, `profile_aglm_output_head_clusters.py`, `profile_aglm_freq_remap_full_model.py`,
  `profile_aglm_patch_components.py`, `profile_aglm_seqlen_backbone.py`
