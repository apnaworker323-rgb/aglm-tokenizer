# AGLM Lossless Hierarchical Latent-Patching Report

Status: **PASSED**  
Frozen vocabulary: **1,551,017** IDs, SHA256 `1f865241d0f3ebcc41bc2e75de8eb6ef190dd23e2fa5a444f69d40ad250c74eb`  
Identical train schedule SHA256: `fa0973bc40a283bce0da545b10b99db6737c13ee167d082ff1528e4e54a9699b`

| Policy | tokens/global pos | raw B/global pos | val BPB | codec retrieval | learned repeat retrieval | learned token accuracy | end-to-end raw MiB/s | GPU-step raw MiB/s | peak MiB | GPU ratio | projected 20TB days | quality preserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| fixed-1-control | 1.000 | 4.852 | 2.96148 | 1.000 | 0.09340659340659341 | 0.04492 | 0.0274 | 0.0322 | 1826.0 | 1.000x | 8070.2 | YES |
| fixed-2 | 2.000 | 9.704 | 2.96102 | 1.000 | 0.09340659340659341 | 0.04492 | 0.0289 | 0.0316 | 1803.2 | 0.982x | 7627.3 | YES |
| fixed-4 | 4.000 | 19.407 | 2.96376 | 1.000 | 0.08691308691308691 | 0.04238 | 0.0305 | 0.0320 | 1793.4 | 0.995x | 7239.1 | NO |
| fixed-8 | 8.000 | 38.814 | 2.95813 | 1.000 | 0.09040959040959042 | 0.04336 | 0.0310 | 0.0318 | 1787.4 | 0.990x | 7115.1 | NO |
| dynamic-entropy | 4.653 | 22.578 | 2.96392 | 1.000 | 0.0899100899100899 | 0.04453 | 0.0304 | 0.0318 | 1792.8 | 0.988x | 7261.1 | NO |

## Hard gates

- Vocabulary artifacts are hash-identical before and after; no token was added, removed, or renumbered.
- Every patch policy reconstructs all 61,440 audited uint32 tokens bit-for-bit; original and decoded SHA256 values match.
- Exact random-access local-code retrieval is 100% across 240 trials per policy.
- The deterministic exact decoder and the learned causal LM predictor are separate measurements. The learned predictor is not claimed to reconstruct every token.
- Future-token perturbation produced exactly zero change in all checked earlier prediction states for every policy.
- `quality preserved` is strict: validation BPB may not exceed patch-1 and learned long-range repeat retrieval may not fall below patch-1.
- Position reduction is never labeled a speedup. `GPU wall-clock ratio` uses synchronized forward + backward + clipping + optimizer time and excludes CPU patch construction.

## Decision

- Non-control policies passing the sampled BPB + learned-retrieval quality gate: **fixed-2**.
- Policies with quality-preserving GPU wall-clock speedup: **none**.
- The best quality-preserving candidate, fixed-2, ran at 0.982x patch-1 GPU throughput. It is therefore not a confirmed GPU speedup.
- Production remains on patch-1. Fixed-2 is a research candidate only; fixed-4, fixed-8, and dynamic entropy patching fail the current strict quality gate.
- The 20TB projections are configuration-specific linear extrapolations from this small GPU prototype, not production forecasts.

## Scope caveat

These are 100-step controlled prototype results on a compact full-ID model, not converged final-model quality. The exact FP32 local code is structurally reversible; that does not make a learned generative prediction exact. A policy failing the strict quality gate remains experimental even when its codec is mathematically lossless.
