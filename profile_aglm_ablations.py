#!/usr/bin/env python3
"""Phase 2 upper-bound ablations for the frozen Patch-1 AGLM production baseline.

Three controlled variants, same batch=2/seq_len=512/bf16 shape and same
warmup/measurement methodology as the production benchmark and the Phase 1
component profiler, so results are directly comparable:

  A  FULL MODEL       - the real training path (embedding -> backbone -> real
                         1.55M-class hierarchical adaptive softmax loss).
  B  BACKBONE ONLY     - identical embedding+backbone forward/backward, but the
                         real output/loss is replaced by the cheapest
                         mathematically valid dummy objective (mean-square of
                         the final hidden state). Profiling only; not a model
                         anyone would train.
  C  OUTPUT HEAD ONLY  - random hidden states (matching shape/dtype/device of
                         real backbone output) with REAL target IDs drawn from
                         the production schedule, fed through the real
                         AdaptiveLogSoftmaxWithLoss module. Isolates the cost
                         of the output head independent of the backbone.

No checkpoints are written; this does not train a usable model.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

import benchmark_aglm_gpu_training as bench

EXPECTED_VOCAB_SIZE = bench.EXPECTED_VOCAB_SIZE


def _run_generic(
    label: str,
    config: bench.BenchmarkConfig,
    schedule,
    device: torch.device,
    forward_fn,
    trainable_params,
) -> Dict[str, Any]:
    """forward_fn(inputs, targets) -> scalar loss tensor, under autocast."""
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate,
                                   weight_decay=config.weight_decay, betas=(0.9, 0.95), fused=True)
    dtype = bench._autocast_dtype(config.precision)
    autocast_enabled = config.precision != "fp32"
    scaler = torch.amp.GradScaler("cuda", enabled=config.precision == "fp16")

    step_ms: List[float] = []
    fwd_ms: List[float] = []
    bwd_ms: List[float] = []
    losses: List[float] = []
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
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=autocast_enabled):
            loss = forward_fn(inputs, targets)
        fwd_end = torch.cuda.Event(enable_timing=True); fwd_end.record()

        bwd_start = torch.cuda.Event(enable_timing=True); bwd_start.record()
        scaler.scale(loss).backward()
        bwd_end = torch.cuda.Event(enable_timing=True); bwd_end.record()

        if config.clip_grad > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, config.clip_grad)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize(device)
        wall_elapsed = (time.perf_counter() - wall_started) * 1000.0

        if batch_index == config.warmup_steps - 1:
            monitor.start()
        if batch_index >= config.warmup_steps:
            step_ms.append(wall_elapsed)
            fwd_ms.append(fwd_start.elapsed_time(fwd_end))
            bwd_ms.append(bwd_start.elapsed_time(bwd_end))
            losses.append(float(loss.detach().cpu()))
        print(f"[{label}] {batch_index + 1:03d}/{total_batches:03d} wall={wall_elapsed:.2f}ms "
              f"loss={float(loss.detach().cpu()):.5f}", flush=True)

    telemetry = monitor.stop()
    tokens = config.steps * config.batch_size * config.seq_len
    total_s = sum(step_ms) / 1000.0
    result = {
        "label": label,
        "tokens_per_second": tokens / max(total_s, 1e-9),
        "mean_step_ms": statistics.fmean(step_ms),
        "p50_step_ms": bench._percentile(step_ms, 0.50),
        "p95_step_ms": bench._percentile(step_ms, 0.95),
        "mean_forward_ms": statistics.fmean(fwd_ms),
        "mean_backward_ms": statistics.fmean(bwd_ms),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1 << 20),
        "trainable_param_count": sum(p.numel() for p in trainable_params),
        "nvidia_smi": telemetry,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
    }
    del optimizer, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_full_model(config, schedule, device) -> Dict[str, Any]:
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    model = bench.AGLMBenchmarkLM(config).to(device)
    model.train()

    def forward_fn(inputs, targets):
        return model.loss(inputs, targets)

    result = _run_generic("A_full_model", config, schedule, device, forward_fn, list(model.parameters()))
    del model
    gc.collect(); torch.cuda.empty_cache()
    return result


def run_backbone_only(config, schedule, device) -> Dict[str, Any]:
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    model = bench.AGLMBenchmarkLM(config).to(device)
    model.train()
    # Drop the real output head's parameters from the trainable set entirely;
    # it is never called in this mode.
    backbone_params = (
        list(model.lexical.parameters())
        + list(model.lexical_projection.parameters())
        + list(model.blocks.parameters())
        + list(model.final_norm.parameters())
    )

    def forward_fn(inputs, targets):
        del targets
        hidden = model.hidden(inputs)
        # Cheapest mathematically valid dummy objective: MSE of the hidden
        # state against zero. Provides a real scalar with a real gradient
        # through the full backbone; carries no linguistic meaning.
        return hidden.float().pow(2).mean()

    result = _run_generic("B_backbone_only", config, schedule, device, forward_fn, backbone_params)
    del model
    gc.collect(); torch.cuda.empty_cache()
    return result


def run_output_head_only(config, schedule, device) -> Dict[str, Any]:
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    output_head = nn.AdaptiveLogSoftmaxWithLoss(
        config.d_model, EXPECTED_VOCAB_SIZE, list(config.cutoffs), div_value=4.0, head_bias=False,
    ).to(device)
    output_head.train()
    n_rows = config.batch_size * config.seq_len

    def forward_fn(inputs, targets):
        del inputs
        # Random hidden states, same shape/dtype/device as the real backbone's
        # final_norm output would be; REAL target IDs from the production
        # schedule so the class distribution seen by the head is realistic.
        hidden = torch.randn(n_rows, config.d_model, device=device, dtype=torch.float32, requires_grad=True)
        return output_head(hidden, targets.reshape(-1)).loss

    result = _run_generic("C_output_head_only", config, schedule, device, forward_fn, list(output_head.parameters()))
    del output_head
    gc.collect(); torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-json", default="benchmark_results/phase2/phase2_ablations.json")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--d-lexical", type=int, default=32)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--ffn-multiple", type=float, default=8 / 3)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--cutoffs", default="16384,131072,524288")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    device = torch.device("cuda:0")
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest, raw_bytes_per_token = bench._load_and_validate_manifest(manifest_path, args.split)
    cutoffs = tuple(int(v) for v in args.cutoffs.split(","))
    config = bench.BenchmarkConfig(
        manifest=str(manifest_path), split=args.split, seq_len=args.seq_len,
        batch_size=args.batch_size, warmup_steps=args.warmup_steps, steps=args.steps,
        seed=args.seed, d_model=args.d_model, d_lexical=args.d_lexical,
        n_layers=args.layers, n_heads=args.heads, ffn_multiple=args.ffn_multiple,
        precision=args.precision, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, clip_grad=args.clip_grad, cutoffs=cutoffs,
    )

    dataset = bench.AGLMShardedDataset(manifest=str(manifest_path), split=args.split,
                                        seq_len=args.seq_len, seed=args.seed)
    try:
        schedule, schedule_info = bench._build_schedule(dataset, config.warmup_steps + config.steps, config.batch_size)
    finally:
        dataset.close()

    results = {}
    for name, fn in (("A_full_model", run_full_model),
                      ("B_backbone_only", run_backbone_only),
                      ("C_output_head_only", run_output_head_only)):
        print(f"\n=== {name} ===", flush=True)
        gc.collect(); torch.cuda.empty_cache()
        results[name] = fn(config, schedule, device)

    a_tps = results["A_full_model"]["tokens_per_second"]
    b_tps = results["B_backbone_only"]["tokens_per_second"]
    c_step_ms = results["C_output_head_only"]["mean_step_ms"]

    summary = {
        "config": {
            "seq_len": args.seq_len, "batch_size": args.batch_size, "precision": args.precision,
            "d_model": args.d_model, "layers": args.layers, "heads": args.heads, "cutoffs": cutoffs,
        },
        "raw_bytes_per_token": raw_bytes_per_token,
        "results": results,
        "backbone_speedup_over_full_model": b_tps / a_tps,
        "output_head_share_of_full_step_pct": c_step_ms / results["A_full_model"]["mean_step_ms"] * 100.0,
    }

    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")

    print("\n=== PHASE 2 SUMMARY ===")
    for name, r in results.items():
        print(f"{name:22s} tok/s={r['tokens_per_second']:9.1f}  mean_step={r['mean_step_ms']:8.3f}ms  "
              f"fwd={r['mean_forward_ms']:7.3f}ms bwd={r['mean_backward_ms']:7.3f}ms "
              f"peak_vram={r['peak_allocated_mib']:8.1f}MiB  params={r['trainable_param_count']:,}")
    print(f"\nBackbone-only is {summary['backbone_speedup_over_full_model']:.2f}x the full-model tok/s.")
    print(f"Output-head-only step time is {summary['output_head_share_of_full_step_pct']:.1f}% of the full-model step time.")
    print(f"\nWrote {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
