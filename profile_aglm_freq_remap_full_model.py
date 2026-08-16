#!/usr/bin/env python3
"""End-to-end measurement: full production model, output-head classes remapped
to frequency rank via a precomputed exact permutation, applied only at the
loss boundary. Input token IDs (embedding lookups) are completely untouched —
this changes nothing about the frozen tokenizer or input vocabulary. It only
changes which internal adaptive-softmax cluster a given target lands in, by
relabeling classes so cluster boundaries align with measured corpus frequency
instead of raw (unordered) token ID.

This is exact and lossless: the permutation is a bijection over the frozen
1,551,017-ID space, computed once from aglm_tokenized_dataset's own
token_frequency.npy. It is invertible, so predicted classes can be mapped
back to real token IDs. No vocabulary is reduced, clamped, or approximated.

Profiling only. Writes no checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

import benchmark_aglm_gpu_training as bench

EXPECTED_VOCAB_SIZE = bench.EXPECTED_VOCAB_SIZE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--permutation", required=True)
    parser.add_argument("--output-json", default="benchmark_results/phase3/freq_remap_full_model.json")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--cutoffs", default="16384,131072,524288")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest, raw_bytes_per_token = bench._load_and_validate_manifest(manifest_path, "train")
    cutoffs = tuple(int(v) for v in args.cutoffs.split(","))
    config = bench.BenchmarkConfig(
        manifest=str(manifest_path), split="train", seq_len=args.seq_len, batch_size=args.batch_size,
        warmup_steps=args.warmup_steps, steps=args.steps, seed=20260815, d_model=192, d_lexical=32,
        n_layers=4, n_heads=6, ffn_multiple=8 / 3, precision=args.precision, learning_rate=3e-4,
        weight_decay=0.1, clip_grad=1.0, cutoffs=cutoffs,
    )

    permutation_np = np.load(args.permutation)
    assert permutation_np.shape[0] == EXPECTED_VOCAB_SIZE, "permutation must cover the full frozen vocabulary"
    permutation = torch.from_numpy(permutation_np).to(device)

    dataset = bench.AGLMShardedDataset(manifest=str(manifest_path), split="train", seq_len=args.seq_len, seed=20260815)
    try:
        schedule, _ = bench._build_schedule(dataset, config.warmup_steps + config.steps, config.batch_size)
    finally:
        dataset.close()

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    model = bench.AGLMBenchmarkLM(config).to(device)
    model.train()
    optimizer, fused = bench._make_optimizer(model, config)
    dtype = bench._autocast_dtype(config.precision)

    def forward_fn(inputs, targets):
        hidden = model.hidden(inputs).reshape(-1, config.d_model)
        remapped_targets = permutation[targets.reshape(-1)]
        return model.output(hidden, remapped_targets).loss

    step_ms, fwd_ms, bwd_ms, losses = [], [], [], []
    torch.cuda.reset_peak_memory_stats(device)
    monitor = bench.NvidiaSmiMonitor()
    total_batches = config.warmup_steps + config.steps

    for batch_index in range(total_batches):
        cpu_inputs, cpu_targets = schedule[batch_index]
        inputs = cpu_inputs.to(device)
        targets = cpu_targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize(device)
        wall_started = time.perf_counter()
        fwd_start = torch.cuda.Event(enable_timing=True); fwd_start.record()
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=args.precision != "fp32"):
            loss = forward_fn(inputs, targets)
        fwd_end = torch.cuda.Event(enable_timing=True); fwd_end.record()

        bwd_start = torch.cuda.Event(enable_timing=True); bwd_start.record()
        loss.backward()
        bwd_end = torch.cuda.Event(enable_timing=True); bwd_end.record()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        optimizer.step()
        torch.cuda.synchronize(device)
        wall_elapsed = (time.perf_counter() - wall_started) * 1000.0

        if batch_index == config.warmup_steps - 1:
            monitor.start()
        if batch_index >= config.warmup_steps:
            step_ms.append(wall_elapsed)
            fwd_ms.append(fwd_start.elapsed_time(fwd_end))
            bwd_ms.append(bwd_start.elapsed_time(bwd_end))
            losses.append(float(loss.detach().cpu()))
        print(f"[freq-remap-full] {batch_index + 1:03d}/{total_batches:03d} wall={wall_elapsed:.2f}ms "
              f"loss={float(loss.detach().cpu()):.5f}", flush=True)

    telemetry = monitor.stop()
    tokens = config.steps * config.batch_size * config.seq_len
    total_s = sum(step_ms) / 1000.0
    result = {
        "tokens_per_second": tokens / max(total_s, 1e-9),
        "effective_raw_mib_per_second": tokens * raw_bytes_per_token / (1 << 20) / max(total_s, 1e-9),
        "mean_step_ms": statistics.fmean(step_ms),
        "p50_step_ms": bench._percentile(step_ms, 0.50),
        "p95_step_ms": bench._percentile(step_ms, 0.95),
        "mean_forward_ms": statistics.fmean(fwd_ms),
        "mean_backward_ms": statistics.fmean(bwd_ms),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1 << 20),
        "nvidia_smi": telemetry,
        "loss_first": losses[0], "loss_last": losses[-1],
        "note": "loss values are in permuted class-id space; not comparable to un-remapped loss magnitudes token-for-token, "
                "though both are exact NLL over the same 1,551,017-way distribution under a relabeling bijection.",
    }

    del optimizer, model
    gc.collect(); torch.cuda.empty_cache()

    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    print(f"\nWrote {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
