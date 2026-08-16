# AGLM 20TB Pareto Report

Status: **PRELIMINARY — 2 candidates, not the full Phase 12 frontier.** A real Pareto frontier needs the
tokenizer comparison (Phase 6) and hybrid/Mamba screening (Phase 8/9) as additional candidates, neither of which
was run this session (see `AGLM_ARCHITECTURE_OPTIMIZATION_REPORT.md`). This document reports what the two real,
measured candidates from this session look like on the requested axes, and is explicit about what is still
missing rather than inventing a combined score or a fabricated BPB column.

## Candidates

| candidate | architecture change | vocabulary change | tokenizer change |
|---|---|---|---|
| **baseline** | none (production Patch-1) | none | none |
| **frequency-remapped output head** | none — same modules, same parameter count, same cutoffs | none — exact bijection over the same 1,551,017 IDs | none |

The two candidates are identical in every respect except which internal adaptive-softmax cluster a given target
class is routed to at the loss. This is deliberate: it isolates the calibration fix from every other variable.

## Measured axes

| metric | baseline | freq-remapped | Δ |
|---|---:|---:|---:|
| **raw MiB/s** (X axis) | 0.0459 | **0.1098** | **2.39×** |
| tok/s | 9,795.7 | 23,428.9 | 2.39× |
| peak VRAM (MiB) | 2,724.4 | 1,485.9 | −45.5% |
| parameters | 67,872,091 | 67,872,091 | unchanged |
| projected 20TB days (linear projection, see caveat below) | 4,808.8 | 2,010.6 | 2.39× |
| **validation BPB** (Y axis) | **not measured** | **not measured** | — |
| retrieval / quality battery | not measured | not measured | — |

## Why the Y axis (BPB) is blank, not estimated

Every run behind these numbers uses a freshly-initialized model over ~100 steps — that is sufficient for timing
but says nothing valid about converged quality, and the loss values recorded in the raw JSON artifacts should
not be read as BPB. Plotting a real X-vs-Y Pareto scatter requires the Phase 13 quality battery run against
each candidate trained to at least a short real convergence point. That has not happened yet. Reporting a
fabricated or extrapolated BPB number here would violate the brief's explicit instruction not to make quality
claims from synthetic benchmarks — so this report stops at the one axis that is actually measured.

## What can be said with the data in hand

The frequency-remapped candidate is not merely "on the frontier" relative to baseline — on every axis measured
so far it **strictly dominates**: faster (2.39×), lower peak memory (−45%), identical parameter count, identical
architecture, identical tokenizer, identical input vocabulary. The only way baseline would remain preferable is
if the remap turns out to regress quality (BPB, retrieval, or the exact-copy/rare-token/multilingual battery) —
which is mathematically not expected (it is an exact relabeling of the same categorical distribution) but is,
per the brief's own standing rule, not yet *measured*, so it is not yet claimed.

## Rejected / not-yet-candidates

- **Patch-2**: not rejected outright, but currently dominated by baseline on speed at equal architecture
  complexity (0.99–1.02× baseline throughput depending on run, see `AGLM_PATCH2_BOTTLENECK_REPORT.md`) while
  adding a codec + GRU decoder. Revisit after the output-head fix changes the backbone's relative weight.
- **Dense (non-adaptive) softmax**: not evaluated — reasoned infeasible on this GPU from the cluster-audit OOM
  evidence in `AGLM_OUTPUT_HEAD_AUDIT.md` (a single 393K-wide or 1.03M-wide cluster already exceeds 6 GiB when
  all rows land in it; a fully dense 1.55M-wide logits tensor would be worse). Not strictly rejected — just
  untested on hardware capable of attempting it.
- **Mamba/hybrid variants (T0–T7)**: not screened. Per Case A prioritization, no backbone-level change is
  expected to be visible above the current bottleneck; screening them now would not produce a meaningful
  frontier position.

## Required before this becomes a real Pareto frontier

1. Phase 13 quality battery (held-out BPB at minimum) for both candidates here, trained to a real, comparable
   checkpoint — not the 100-step timing runs this report is built from.
2. Phase 6 tokenizer comparison, to know whether the frozen 1.55M vocabulary itself is the right X-axis anchor
   or whether a different tokenizer changes bytes/token enough to matter.
3. Phase 8/9 hybrid screening, once §5 of the optimization report (the real attention-vs-output-head crossover
   point, currently unmeasurable due to the unfixed output head's OOM ceiling) is resolved.

Until those land, treat this report as: *one exact, low-risk, high-confidence speed win, awaiting a quality
gate before being called a finalist* — not a frontier.
