# AGLM Output Head Audit — Phase 2 + Phase 3

Status: **COMPLETE** for the questions this session set out to answer; see "Not done" at the end.
Scripts: `profile_aglm_ablations.py`, `profile_aglm_output_head_clusters.py`, `profile_aglm_freq_remap_full_model.py`
All runs: batch=2, seq_len=512, bf16, 1,551,017-class output, cutoffs=(16384, 131072, 524288), div_value=4.0,
10 warmup + 100 stable steps unless noted otherwise (cluster-isolation microbenchmarks use 3+15, run one pattern
per subprocess for clean VRAM isolation — noted per table).

## Phase 2 — upper-bound ablations

Three controlled variants, same shape/precision/methodology, each its own fresh model instance:

| variant | tok/s | mean step ms | mean fwd ms | mean bwd ms | peak VRAM MiB | trainable params |
|---|---:|---:|---:|---:|---:|---:|
| **A — full model** (real training path) | 10,270.0 | 99.708 | 35.869 | 44.855 | 2,724.4 | 67,872,091 |
| **B — backbone only** (real embedding+backbone; output/loss replaced by cheapest valid dummy: MSE of final hidden state to zero; profiling only, never trained) | 42,000.5 | 24.381 | 3.544 | 6.836 | 866.8 | 51,409,888 |
| **C — output head only** (random hidden states, shape-matched to the real backbone output; **real** target IDs drawn from the production schedule; the real, unmodified `AdaptiveLogSoftmaxWithLoss` module) | 13,221.0 | 77.452 | 32.702 | 40.096 | 2,086.8 | 16,462,203 |

**Backbone-only runs at 4.09× the full model's tok/s.** That is the honest upper bound on how much faster this
architecture could train if the output/loss stage were free — not infinite, not 10×, but a real, measured 4.09×
ceiling.

**Output-head-only, in complete isolation from the backbone, costs 77.45 ms/step — 77.7% of the full model's
step time by itself.** This independently corroborates Phase 1's event-hook measurement of 70.0% via a
completely different method (no shared instrumentation code, no shared timing mechanism). Two independent
measurement techniques agreeing within ~8 percentage points on a noisy consumer GPU is strong convergent
evidence, not a coincidence of one flawed harness.

## Phase 3 — cluster-level audit: *why* is the output head this expensive?

`nn.AdaptiveLogSoftmaxWithLoss` with cutoffs `(16384, 131072, 524288)` on a 1,551,017-class vocabulary defines
four clusters:

| cluster | class range | width | tail projection shape |
|---|---|---:|---|
| head | [0, 16,384) | 16,384 | direct (+3 cluster-representative logits) |
| tail0 | [16,384, 131,072) | 114,688 | 192→48→114,688 |
| tail1 | [131,072, 524,288) | 393,216 | 192→12→393,216 |
| tail2 | [524,288, 1,551,017) | 1,026,729 | 192→3→1,026,729 |

Every row pays for the 16,387-wide head projection + log_softmax regardless of its target. **Additionally**,
whichever tail clusters have *at least one* row land in them pay for that cluster's own up-projection +
log_softmax over its full width, for the subset of rows that hit it. To find out whether this per-cluster cost
is driven by how many rows land in a cluster or by how wide the cluster is, six controlled target-distribution
patterns were fed through the real module (random hidden states, all other settings identical):

| pattern | rows in (head, tail0, tail1, tail2) | fwd ms | bwd ms | step ms |
|---|---|---:|---:|---:|
| `head_only` — all 1024 rows in the head cluster | (1024, 0, 0, 0) | 3.145 | 4.596 | **8.392** |
| `tail0_only` — all 1024 rows in tail0 (114,688-wide) | (0, 1024, 0, 0) | 22.967 | 28.806 | **53.531** |
| `tail1_only` — all 1024 rows in tail1 (393,216-wide) | (0, 0, 1024, 0) | — | — | **OOM** (tried to allocate 1.50 GiB with only ~1.4 GiB free) |
| `tail2_only` — all 1024 rows in tail2 (1,026,729-wide) | (0, 0, 0, 1024) | — | — | **OOM** (tried to allocate 3.92 GiB) |
| `tail2_single_row` — 1023 rows head, **1 single row** in tail2 | (1023, 0, 0, 1) | 4.024 | 6.355 | **11.615** |
| `natural` — real corpus target IDs (production distribution) | (324, 489, 146, 65) | 34.627 | 43.777 | **81.705** |

**The mechanism is column-width-bound, not row-count-bound.** Going from `head_only` to
`tail2_single_row` — literally one row out of 1,024 moved into the widest cluster, everything else held fixed —
adds **+3.2ms (+38%)** to step time. That single row forces a `(≥1, 3)→(≥1, 1,026,729)` up-projection and a
1,026,729-wide log_softmax reduction whose cost is dominated by the column dimension, not the row count. Scaling
that column-width cost up across *all* 1,024 rows (`tail1_only`, `tail2_only`) exceeds the 6 GiB budget outright.
And `natural` — the actual production target distribution — is the single *most* expensive row in this table,
worse than any pattern that forces every row into one cluster, because real corpus batches routinely touch
**all four clusters simultaneously**, paying each cluster's largely row-count-independent width tax every step.

### The root cause: cutoffs assume a frequency-sorted vocabulary. AGLM's IDs are not frequency-sorted.

Adaptive softmax's entire efficiency argument (Grave et al. 2017) depends on cutoffs aligning with a
frequency-ranked vocabulary, so the cheap head cluster captures the bulk of *occurrences* even though it is a
small slice of *unique IDs*. Checked directly against `aglm_tokenized_dataset/metadata/token_frequency.npy`
(1,479,404,911 total occurrences, the real training corpus):

| cluster (by current raw ID range) | occurrence mass | % of corpus |
|---|---:|---:|
| head [0, 16,384) | 457,700,931 | **30.94%** |
| tail0 [16,384, 131,072) | 794,268,511 | **53.69%** |
| tail1 [131,072, 524,288) | 149,788,452 | 10.12% |
| tail2 [524,288, V) | 77,647,017 | 5.25% |

Only 30.9% of real training tokens land in the cheap head cluster. Compare against what a *correctly
frequency-calibrated* head cluster of the same size (16,384 slots) would capture:

| true top-*k*-by-measured-frequency IDs | corpus coverage |
|---:|---:|
| 1,000 | 63.48% |
| 4,096 | 77.72% |
| 8,192 | 83.71% |
| **16,384** | **88.68%** |
| 32,768 | 92.75% |
| 114,688 (head+tail0-sized) | 98.02% |

Overlap between the true top-16,384-by-frequency token IDs and the current head cluster's ID range
`[0, 16384)`: **576 of 16,384 (3.5%)**. Correlation between raw token ID and measured corpus frequency:
**−0.0145** — statistically indistinguishable from zero. AGLM's 1.55M ID space is evidently allocated by
structural/lexical category (consistent with the tokenizer's own `allocation`/`morphology`/`pool` modules), not
by frequency. The cutoffs `(16384, 131072, 524288)` were chosen as round numbers against an ID space where
"round number of IDs" has no relationship to "round amount of occurrence mass." This is a **calibration bug**,
not an inherent property of needing 1.55M classes.

*(Correction note: the first pass at this frequency analysis used `np.argsort(-freq)` on the `uint64` frequency
array, which silently wraps around instead of erroring on an unsigned dtype and produced a nonsense ordering
— it initially reported 0% coverage at every *k*, which is impossible given the corpus's own most-common-token
counts. Caught before being reported by cross-checking against the dataset manifest's independently-computed
`top_tokens` list, fixed by sorting ascending and reversing instead of negating. Recorded here because it is
exactly the kind of silent-corruption bug worth distrusting on the first pass.)*

### The fix: an exact, lossless output-side frequency permutation

A bijection `remap: token_id → frequency_rank` was precomputed once from `token_frequency.npy` and applied
**only** at the loss boundary — immediately before `AdaptiveLogSoftmaxWithLoss(hidden, targets)` — via
`remapped_targets = remap[targets]`. Nothing else changes:

- The frozen tokenizer, the frozen input vocabulary, and the input embedding lookup are **completely
  untouched** — this operates purely on which internal cluster a class ends up in, at the loss, not on what
  token IDs mean.
- It is a bijection over the full 1,551,017-ID space: no class is merged, dropped, clamped, or approximated.
  Every class remains individually addressable and distinguishable; the permutation is exactly invertible for
  turning a predicted rank back into a real token ID at inference/generation time (not implemented this
  session — see "Not done").
- Same module, same cutoffs, same parameter count, same weights structure — only the *label space* the loss
  operates on changes.

Measured effect, output head alone (same `natural` real-target microbenchmark, targets remapped):

| | rows (head, tail0, tail1, tail2) | fwd ms | bwd ms | step ms |
|---|---|---:|---:|---:|
| `natural` (current, unremapped) | (324, 489, 146, 65) | 34.627 | 43.777 | 81.705 |
| `natural_freq_remapped` | (878, 95, 49, 2) | 10.528 | 28.100 | **41.907** |

**1.95× faster**, and cluster occupancy shifts to 85.7% head-cluster (vs. the 88.68% ceiling computed from the
whole corpus — close, sampling noise from a single 1,024-token batch).

Measured effect, **full end-to-end model** (real backbone + real remapped output head, same 100-step
methodology as Phase 1/2, fresh model each run):

| | tok/s | raw MiB/s | mean step ms | peak VRAM MiB |
|---|---:|---:|---:|---:|
| baseline (current, unremapped cutoffs) | 9,795.7 | 0.0459 | 104.535 | 2,724.4 |
| **frequency-remapped output classes** | **23,428.9** | **0.1098** | **43.707** | **1,485.9** |

**2.39× end-to-end training throughput, from an exact relabeling — no architecture change, no vocabulary
reduction, no approximation.** Peak VRAM also drops 45% (2,724 → 1,486 MiB), because the wide-tail-cluster
activations that dominated memory shrink along with how often those clusters get touched.

### Alternatives considered

Per Phase 3's request, other exact/well-justified output parameterizations were considered but not all were
implementable safely this session:

1. **Current hierarchical adaptive softmax, uncalibrated cutoffs** — measured (baseline above).
2. **Same hierarchical adaptive softmax, frequency-calibrated via output-side permutation** — measured above.
   **Recommended next candidate.**
3. **Fully dense softmax** (no clustering) — not attempted. Back-of-envelope: a `(1024, 1551017)` bf16 logits
   tensor alone is ≈3.2 GB; combined with a `(192, 1,551,017)` weight matrix's fp32 AdamW state (≈4.8 GB) and
   backbone activations, this does not fit in 6 GiB, and the cluster-audit table above already shows single
   wide-cluster tensors OOMing at far smaller effective widths. Would need a larger GPU to even attempt as a
   reference point; flagged as follow-up.
4. **Factorized/tied output, candidate-based (sampled) output** — not implemented this session. Worth
   revisiting only if the frequency-remap fix (item 2) turns out insufficient once validated end-to-end in a
   real training run.

## What this does *not* yet prove

- **Speed only.** The 2.39× full-model number comes from freshly-initialized weights over 100 steps, exactly
  like every other benchmark in this project — it is not a converged-quality comparison. The remap is
  mathematically an exact relabeling of the same 1,551,017-way categorical distribution, so the achievable loss
  *should* be identical; that expectation has **not** been empirically confirmed by an actual training run to
  convergence with a BPB comparison. This is squarely Phase 13's job and has not been run.
- **Inference-side plumbing not built.** Using this in production needs the inverse permutation applied when
  turning a predicted class back into a token ID (trivial — one more precomputed lookup table — but not wired
  up or tested this session).
- Items 3–4 above remain untested.

## Raw artifacts

- `benchmark_results/phase2/phase2_ablations.json`, `phase2_run.log`
- `benchmark_results/phase3/cluster_audit_*.json` (one file per pattern, isolated subprocess)
- `benchmark_results/phase3/freq_rank_permutation.npy` (the precomputed bijection, int64, shape (1,551,017,))
- `benchmark_results/phase3/freq_remap_full_model.json`
