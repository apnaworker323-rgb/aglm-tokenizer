# AGLM Patch-2 Bottleneck Report — Phase 10

Status: **COMPLETE** — root cause identified and quantified.
Script: `profile_aglm_patch_components.py`
Config: identical to `AGLM_LATENT_PATCHING_REPORT.md` — `HierarchicalLatentLM`, d_model=160, d_lexical=24,
layers=6, heads=5, seq_len=256, batch=2, bf16, cutoffs=(16384, 131072, 524288). 10 warmup + 100 stable steps.
Same class (`aglm_latent_patching.HierarchicalLatentLM`), unmodified; forward methods instrumented at call time
with CUDA-event checkpoints exactly like Phase 1 — zero math changes.

## Recap of the clue

`AGLM_LATENT_PATCHING_REPORT.md` found that `fixed-2` patching halves the number of *global* Transformer
positions (twice as many raw bytes per global position) while leaving sampled BPB essentially unchanged — but
measured **0.982× the GPU throughput of Patch-1**. Fewer positions did not translate into a speedup. This
session set out to find out exactly where the expected savings went.

## Component breakdown

| stage | Patch-1 (`fixed-1-control`) fwd ms | Patch-2 (`fixed-2`) fwd ms | Δ | note |
|---|---:|---:|---:|---|
| `patch_encode` (lexical lookup + projection + patch-summary pooling) | 0.603 | 0.599 | −0.004 | ~fixed cost, independent of patch size |
| `global_backbone` (6-layer causal Transformer over **patch** positions) | 4.450 | 4.297 | **−0.153** | patch count halved (512→256 across the batch); savings are small in absolute terms at this seq_len/depth |
| `local_codec` (`nn.GRU`, decodes individual tokens back out per patch) | 1.015 | 0.790 | **−0.225** | shrinks, doesn't grow — see note below |
| `output_head` (real 1.55M-class adaptive softmax) | 16.439 | 16.412 | −0.027 | **noise-level; unaffected by patch size** |
| **backward (total, one number)** | 33.318 | 33.161 | −0.157 | dominated by output-head backward in both configs |
| **mean step ms** | **70.321** | **69.647** | −0.674 | |
| mean global positions (summed over the 2-row batch) | 512.0 | 256.0 | −256 (exactly 2×, confirms the patch mechanism itself is working as designed) | |
| peak VRAM (MiB) | 1,745.2 | 1,722.0 | −23.2 | |

**Patch-2 / Patch-1 step-time ratio this session: 0.9904×** (i.e., ~1% faster on step time, ~1% higher
throughput) — in the same near-parity ballpark as the original report's 0.982×, on the opposite side of 1.0 by
a margin well within run-to-run noise for a Python-loop-driven benchmark on a small consumer GPU with thermal
and background-load variance. The two measurements agree on the finding that matters: **the effect, in either
direction, is within a couple of percent — nowhere near what "2× fewer global positions" would suggest if the
backbone were the bottleneck.**

## Why: the output head evaluates every original token, patched or not

`HierarchicalLatentLM.token_hidden()` returns `decoded[mask], ids[mask]` — one row **per original token**,
identical in count for Patch-1 and Patch-2 (only the *intermediate* global-position count differs). `loss()`
then calls the real `AdaptiveLogSoftmaxWithLoss` on that full per-token set. Patching changes how many positions
the *global* Transformer sees; it does not and cannot change how many predictions the output head has to score,
because the model still needs to predict every individual token, not every patch. The measured numbers confirm
this exactly: `output_head` forward time is 16.44ms vs. 16.41ms — statistically identical — while
`global_backbone` and `local_codec`, the only two stages patching actually touches, together account for **just
5.5ms of the ~70ms step (≈7.8%)**, of which patching saved a combined 0.38ms.

This is a hard ceiling, not a tuning problem: at this model size and seq_len, **even eliminating
`global_backbone` and `local_codec` entirely would only remove ~8% of step time**, because the output head — at
70%+ of the step, exactly as found in `AGLM_OUTPUT_HEAD_AUDIT.md` — is completely insensitive to how the
Transformer body's sequence length is organized.

## A note on `local_codec`

The per-patch `nn.GRU` decoder got *cheaper* under Patch-2 (1.015ms → 0.790ms), not more expensive as one might
guess from "now it unrolls 2 steps per patch instead of 1." Total token-decode work is identical either way
(patches × width is constant = total tokens), but Patch-1's width=1 config runs the GRU for exactly one
timestep across 512 independent sequences — a degenerate case that pays full recurrent-kernel launch overhead
per sequence for zero actual recurrence benefit. Patch-2 runs half as many (256) sequences of width 2, netting
slightly less overhead. This is a minor effect at this scale (0.2ms) but worth remembering if patch width grows
in a future test: the GRU codec's overhead scales more with *sequence count* than with *total elements* at these
small widths.

## Answer to the primary question

**Why does 2× global-position compression not currently increase GPU throughput?** Because global-position
compression only has purchase over the ~5–8% of step time that lives in the Transformer body and local codec.
The other ~92%, dominated by the output head (~70%+, completely position-count-invariant) and the optimizer
step (~18%, dominated by the two vocabulary-sized parameter tables, also unaffected by patching), cannot be
touched by any amount of sequence/patch compression. Patch-2 is not "useless" as an idea — the mechanism does
exactly what it claims (fewer global positions, preserved BPB, exact lossless codec) — it is simply solving a
problem that is not currently the bottleneck.

## What this implies for sequencing (see also `AGLM_ARCHITECTURE_OPTIMIZATION_REPORT.md`)

Fix the output head first (measured 2.39× available via the exact frequency-remap in
`AGLM_OUTPUT_HEAD_AUDIT.md`) and re-run this same Patch-1 vs. Patch-2 comparison afterward. Once the output head
shrinks from ~70% to something smaller, the backbone+codec's *relative* share of the now-smaller step roughly
doubles — patch compression only becomes worth developing further once it is no longer trying to shave time off
a component that is already a rounding error next to the output head. Recommend **holding Patch-2 in its
current "research candidate, not promoted to production" status** (unchanged from `AGLM_LATENT_PATCHING_REPORT.md`'s
own conclusion) and revisiting it after the output-head fix, not before.

## Raw artifacts

- `benchmark_results/phase10/patch_component_profile.json`
- `benchmark_results/phase10/phase10_run.log`
