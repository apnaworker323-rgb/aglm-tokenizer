# AGLM GPU Component Profile — Phase 1

Status: **COMPLETE**
Scope: Patch-1 production baseline only, architecture frozen and unmodified.
Manifest: `aglm_tokenized_dataset/metadata/dataset_manifest.json` (1,479,404,911 tokens, 4.914080 raw bytes/token)
GPU: NVIDIA GeForce RTX 3050 (5804 MiB), driver 595.84, CUDA 13.2
PyTorch/CUDA: 2.12.0+cu130 (system interpreter, same environment that produced `AGLM_GPU_TRAINING_BENCHMARK_REPORT.md`)
Shape: batch=2, seq_len=512, d_model=192, d_lexical=32, layers=4, heads=6, bf16, cutoffs=(16384, 131072, 524288)
Steps measured: 10 warmup + **100 stable steps**
Scripts: `profile_aglm_gpu_components.py` (instrumentation only — see Methodology)

## Methodology

The real `AGLMBenchmarkLM` / `CausalTransformerBlock` classes from `benchmark_aglm_gpu_training.py` are imported
unmodified. Their `forward` methods are monkey-patched **at call time** to insert `torch.cuda.Event` markers
between existing statements — the arithmetic executed is byte-for-byte identical to the production benchmark;
nothing about weights, shapes, or control flow changes. `register_full_backward_hook` on the same module
boundaries recovers an equivalent breakdown for the backward pass, read off in true reverse-execution firing
order. A `torch.cuda.synchronize()` happens once per step; all deltas are computed from real GPU timestamps
after that, not CPU wall-clock sampling.

A second, fully independent pass uses `torch.profiler` (CPU+CUDA activities, memory profiling on) over 10 more
steps, purely as a cross-check against the event-based numbers and to get an operator-level table, VRAM, and a
kernel count from the exported Chrome trace. No checkpoint is written by either pass.

**Known instrumentation caveat:** PyTorch's full-backward-hook fires *early* (at grad-output availability, not
after its own compute) for the one module whose forward input never requires grad — `nn.Embedding`'s index
tensor. This makes the "input_embedding" backward bucket read near-zero; the real ~1.3ms of embedding-backward
compute (confirmed via the profiler pass's `aten::embedding_dense_backward` row) is not lost, it is folded into
`misc/overhead` instead. This affects only that one bucket and is small (~1.3% of step time); no other bucket is
affected since every other module's forward input is a real tensor.

## Result — component breakdown (mean ms/step, N=100)

| component | forward ms | backward ms | total ms | % of step |
|---|---:|---:|---:|---:|
| **output_head (hierarchical adaptive softmax)** | 32.711 | 40.497 | **73.208** | **70.0%** |
| optimizer_step (fused AdamW) | — | — | 18.709 | 17.9% |
| misc / overhead (see below) | — | — | 9.878 | 9.4% |
| attention (qkv+SDPA+attn_out, all 4 blocks) | 0.491 | 0.205 | 0.696 | 0.67% |
| ffn (gate_up+SiLU+down, all 4 blocks) | 0.449 | 0.265 | 0.715 | 0.68% |
| normalization (attn_norm+ffn_norm×4 + final_norm) | 0.410 | 0.139 | 0.549 | 0.52% |
| embedding_projection (Linear 32→192) | 0.400 | 0.030 | 0.430 | 0.41% |
| input_embedding (lookup, 1,551,017×32) | 0.241 | 0.001† | 0.242 | 0.23% |
| h2d_data_copy | — | — | 0.109 | 0.10% |

† undercounted by the instrumentation caveat above; true cost ≈1.3ms sits inside misc/overhead.

**Cross-check:** measured mean step = 104.535 ms. Forward wall (event pair around the whole forward call) =
40.800 ms; backward wall = 44.995 ms; optimizer + data ≈ 18.8 ms. Sum ≈ 104.6 ms, matching the wall-clock step
time to within noise. Summed component means (94.657 ms) + misc/overhead (9.878 ms, 9.4%) = 104.535 ms exactly
by construction. The 9.4% overhead is consistent with ~40 CUDA-event insertions, ~20 Python-level backward
hooks, and the GradScaler no-op checks per step on a small GPU with real kernel-launch latency — not hidden
compute (the profiler pass's independent operator table accounts for the same top consumers with no unexplained
gap).

## GPU utilization — read this carefully

`nvidia-smi`-reported utilization during the measured window: **mean 92.3%, max 100%**. This number is commonly
misread as "the GPU is being used efficiently." It is not that. `nvidia-smi` utilization is a coarse
busy/idle sample (≥1 kernel running in the sampling window) — it says nothing about how much of the SM's
throughput a kernel actually uses. The profiler pass shows **≈1,260 CUDA kernel launches per step**, many of
them the tiny gather/scatter/copy kernels inside the adaptive-softmax cluster routing (see
`AGLM_OUTPUT_HEAD_AUDIT.md`). A step that is "97% busy" issuing hundreds of small memory-bound kernels back to
back looks identical in `nvidia-smi` to a step doing dense, efficient tensor-core GEMMs. High utilization here is
evidence of *no idle gaps*, not evidence of efficient FLOP usage.

Tensor Core / memory-bandwidth utilization are **not precisely measurable on this machine**: no `ncu`/`nsys`
binary is installed, and `/proc/sys/kernel/perf_event_paranoid=3` blocks the low-level counters Nsight Compute
needs on a consumer GPU without root. Qualitative evidence from the profiler trace instead:
- The backbone's `Linear` layers (bf16, Ampere) do dispatch to a `cutlass::Kernel2<...wmma_tensorop_bf16...>`
  tensor-core GEMM kernel — confirmed by name in the trace.
- `aten::log_softmax` / `aten::_log_softmax_backward_data` (inside the output head) are **not** tensor-core
  kernels; they are memory-bound reduction kernels. Their CUDA-memory churn in the profiler trace peaks at
  **6.93 GB allocated/freed** for a single log_softmax call at this batch×seq shape (dominated by whichever tail
  cluster is widest) — this is why the output head is expensive: it is bandwidth-bound on very wide, sparsely
  populated tensors, not compute-bound on dense GEMMs. Full mechanism in `AGLM_OUTPUT_HEAD_AUDIT.md`.

## Parameter count by module — a second vocabulary-shaped cost

| module | parameters | % of total |
|---|---:|---:|
| **input embedding** (`nn.Embedding(1,551,017, 32)`, **no** `sparse=True`) | 49,632,544 | **73.13%** |
| output head (hierarchical adaptive softmax) | 16,462,203 | 24.25% |
| transformer blocks ×4 | 1,771,008 | 2.61% |
| embedding projection + final norm | 6,336 | 0.01% |
| **total** | 67,872,091 | 100% |

The input embedding table — not the output head — is the single largest parameter block. It has no
`sparse=True` flag, so every AdamW step performs a **dense** update over the full 1,551,017-row table even
though a batch of 1,024 tokens touches at most 1,024 distinct rows (≤0.07% of the table). A synthetic isolation
(same shapes, fused AdamW, `optimizer.step()` timed alone, 50 iterations) attributes the 18.7ms/step optimizer
cost roughly as:

| parameter group | synthetic params | `optimizer.step()` ms (isolated) | share |
|---|---:|---:|---:|
| input embedding | 49,632,544 | 9.603 | ~72% |
| output head | 16,462,203 | 3.298 | ~25% |
| backbone (approx. shapes) | 2,311,872 | 0.450 | ~3% |

(Isolated sum is 13.35ms vs. the measured combined 18.7ms because the production run fuses all parameter groups
into one multi-tensor AdamW call while this isolation used three separate optimizer instances — the
per-call kernel-launch overhead triples. Treat the split as order-of-magnitude, not exact.) The practical
takeaway: **97.4% of the parameter count, and the dominant share of optimizer cost, lives in the two
vocabulary-indexed tables (input embedding + output head), not in the Transformer body.** A `sparse=True`
input embedding + `SparseAdam` (or an equivalent masked/selective update) is a second, independent, and
comparatively simple lever beyond the output-head fix in `AGLM_OUTPUT_HEAD_AUDIT.md` — not measured
end-to-end this session; flagged as follow-up work.

## Answer to Phase 1's implicit question

Attention, FFN, normalization, and the embedding lookup — i.e., "the Transformer" in the everyday sense —
account for **under 2.5% of wall-clock step time combined**, at the current production shape (seq_len=512,
d_model=192, 4 layers). The output head alone is **70.0%**. The optimizer step, itself mostly a tax on the two
vocabulary-sized parameter tables, is another 17.9%. **Before touching attention, FFN, or considering
Mamba/recurrent alternatives, the two components worth optimizing are the output head and the input embedding's
optimizer path.** See `AGLM_OUTPUT_HEAD_AUDIT.md` for the output-head deep dive and a measured fix.

## Raw artifacts

- `benchmark_results/phase1/phase1_component_profile.json` — full event-pass + profiler-pass payload
- `benchmark_results/phase1/phase1_trace.json` — Chrome trace from the profiler pass (24.9 MB)
- `benchmark_results/phase1/phase1_run.log` — full stdout including the top-40 operator table
