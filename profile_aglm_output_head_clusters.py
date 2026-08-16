#!/usr/bin/env python3
"""Phase 3 cluster-level audit of the real AdaptiveLogSoftmaxWithLoss output head.

Isolates the head cluster and each tail cluster by forcing the target
distribution of a fixed-shape synthetic batch (random hidden states, matching
production batch*seq=1024 rows, d_model=192) into controlled patterns:

  head_only        every row's target is a head-cluster (frequent) token id
  tail0_only        every row's target lands in tail cluster 0 (cutoffs[0..1))
  tail1_only        every row's target lands in tail cluster 1 (cutoffs[1..2))
  tail2_only        every row's target lands in tail cluster 2 (cutoffs[2..V))
  natural           real target ids drawn from the production dataset schedule
  tail2_single_row  only ONE row's target is in tail cluster 2, the rest head

This is a pure output-head microbenchmark: the real nn.AdaptiveLogSoftmaxWithLoss
module, unmodified, with its real weights (random init). No backbone, no
checkpoint, not a trainer.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

import benchmark_aglm_gpu_training as bench

EXPECTED_VOCAB_SIZE = bench.EXPECTED_VOCAB_SIZE
FREQ_PERMUTATION_PATH = Path(__file__).parent / "benchmark_results" / "phase3" / "freq_rank_permutation.npy"


def make_targets(pattern: str, n_rows: int, cutoffs, device, real_targets: torch.Tensor) -> torch.Tensor:
    low = [0] + list(cutoffs)
    high = list(cutoffs) + [EXPECTED_VOCAB_SIZE]
    generator = torch.Generator(device="cpu").manual_seed(0)

    def uniform_in(lo: int, hi: int, count: int) -> torch.Tensor:
        return (torch.rand(count, generator=generator) * (hi - lo) + lo).long()

    if pattern == "head_only":
        return uniform_in(low[0], high[0], n_rows).to(device)
    if pattern == "tail0_only":
        return uniform_in(low[1], high[1], n_rows).to(device)
    if pattern == "tail1_only":
        return uniform_in(low[2], high[2], n_rows).to(device)
    if pattern == "tail2_only":
        return uniform_in(low[3], high[3], n_rows).to(device)
    if pattern == "natural":
        return real_targets[:n_rows].to(device)
    if pattern == "natural_freq_remapped":
        permutation = np.load(FREQ_PERMUTATION_PATH)
        table = torch.from_numpy(permutation)
        remapped = table[real_targets[:n_rows].cpu()]
        return remapped.to(device)
    if pattern == "tail2_single_row":
        base = uniform_in(low[0], high[0], n_rows)
        base[0] = torch.randint(low[3], high[3], (1,), generator=generator)[0]
        return base.to(device)
    raise ValueError(pattern)


def run_pattern(pattern: str, config, real_targets, device, warmup, steps) -> Dict[str, Any]:
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    output_head = nn.AdaptiveLogSoftmaxWithLoss(
        config.d_model, EXPECTED_VOCAB_SIZE, list(config.cutoffs), div_value=4.0, head_bias=False,
    ).to(device)
    output_head.train()
    n_rows = config.batch_size * config.seq_len
    optimizer = torch.optim.AdamW(output_head.parameters(), lr=3e-4, weight_decay=0.1,
                                   betas=(0.9, 0.95), fused=True)
    dtype = bench._autocast_dtype(config.precision)

    fwd_ms: List[float] = []
    bwd_ms: List[float] = []
    step_ms: List[float] = []
    cluster_hits = None

    for i in range(warmup + steps):
        targets = make_targets(pattern, n_rows, config.cutoffs, device, real_targets)
        if i == 0:
            low = [0] + list(config.cutoffs)
            high = list(config.cutoffs) + [EXPECTED_VOCAB_SIZE]
            cluster_hits = [int(((targets >= lo) & (targets < hi)).sum()) for lo, hi in zip(low, high)]
        hidden = torch.randn(n_rows, config.d_model, device=device, dtype=torch.float32, requires_grad=True)
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize(device)
        wall_started = time.perf_counter()
        fwd_start = torch.cuda.Event(enable_timing=True); fwd_start.record()
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=config.precision != "fp32"):
            loss = output_head(hidden, targets).loss
        fwd_end = torch.cuda.Event(enable_timing=True); fwd_end.record()

        bwd_start = torch.cuda.Event(enable_timing=True); bwd_start.record()
        loss.backward()
        bwd_end = torch.cuda.Event(enable_timing=True); bwd_end.record()
        optimizer.step()
        torch.cuda.synchronize(device)
        wall_elapsed = (time.perf_counter() - wall_started) * 1000.0

        if i >= warmup:
            fwd_ms.append(fwd_start.elapsed_time(fwd_end))
            bwd_ms.append(bwd_start.elapsed_time(bwd_end))
            step_ms.append(wall_elapsed)

    del output_head, optimizer
    gc.collect(); torch.cuda.empty_cache()

    return {
        "pattern": pattern,
        "cluster_row_counts": cluster_hits,
        "mean_forward_ms": statistics.fmean(fwd_ms),
        "mean_backward_ms": statistics.fmean(bwd_ms),
        "mean_step_ms": statistics.fmean(step_ms),
        "p50_step_ms": bench._percentile(step_ms, 0.50),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-json", default="benchmark_results/phase3/cluster_audit.json")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--cutoffs", default="16384,131072,524288")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--pattern", default=None, help="run only this one pattern (for subprocess isolation)")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    manifest_path = Path(args.manifest).expanduser().resolve()
    cutoffs = tuple(int(v) for v in args.cutoffs.split(","))
    config = bench.BenchmarkConfig(
        manifest=str(manifest_path), split="train", seq_len=args.seq_len, batch_size=args.batch_size,
        warmup_steps=args.warmup, steps=args.steps, seed=20260815, d_model=args.d_model, d_lexical=32,
        n_layers=4, n_heads=6, ffn_multiple=8 / 3, precision=args.precision, learning_rate=3e-4,
        weight_decay=0.1, clip_grad=1.0, cutoffs=cutoffs,
    )

    dataset = bench.AGLMShardedDataset(manifest=str(manifest_path), split="train",
                                        seq_len=args.seq_len, seed=20260815)
    try:
        inputs, targets, _ = dataset.get_batch(batch_size=64, start_index=0)
    finally:
        dataset.close()
    real_targets_flat = targets.reshape(-1)

    all_patterns = ["head_only", "tail0_only", "tail1_only", "tail2_only", "natural", "tail2_single_row",
                     "natural_freq_remapped"]
    patterns = [args.pattern] if args.pattern else all_patterns
    results = {}
    for pattern in patterns:
        print(f"\n=== pattern: {pattern} ===", flush=True)
        try:
            results[pattern] = run_pattern(pattern, config, real_targets_flat, device, args.warmup, args.steps)
            r = results[pattern]
            print(f"cluster_row_counts(head,tail0,tail1,tail2)={r['cluster_row_counts']}  "
                  f"fwd={r['mean_forward_ms']:.3f}ms bwd={r['mean_backward_ms']:.3f}ms step={r['mean_step_ms']:.3f}ms")
        except torch.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            results[pattern] = {"pattern": pattern, "oom": True, "error": str(exc)}
            print(f"OOM on pattern {pattern}: {exc}")

    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    if args.pattern:
        output_json = output_json.with_name(f"{output_json.stem}_{args.pattern}{output_json.suffix}")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"\nWrote {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
