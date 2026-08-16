#!/usr/bin/env python3
"""Phase 5 (partial): backbone-only forward/backward ms across sequence lengths.

Reuses the Phase 2 backbone-only harness (dummy MSE objective on the final
hidden state; real output head never called) to isolate how attention/FFN
cost inside the backbone scales with seq_len, without the real output head's
memory footprint confounding the picture (Phase 1 showed the real output head
OOMs by seq_len=2048 on this 6GB GPU well before attention would matter).
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

import torch

import benchmark_aglm_gpu_training as bench
import profile_aglm_ablations as ablations


def run_one(seq_len: int, batch_size: int, manifest_path: Path, device, warmup: int, steps: int) -> Dict[str, Any]:
    config = bench.BenchmarkConfig(
        manifest=str(manifest_path), split="train", seq_len=seq_len, batch_size=batch_size,
        warmup_steps=warmup, steps=steps, seed=20260815, d_model=192, d_lexical=32,
        n_layers=4, n_heads=6, ffn_multiple=8 / 3, precision="bf16", learning_rate=3e-4,
        weight_decay=0.1, clip_grad=1.0, cutoffs=(16384, 131072, 524288),
    )
    dataset = bench.AGLMShardedDataset(manifest=str(manifest_path), split="train", seq_len=seq_len, seed=20260815)
    try:
        schedule, _ = bench._build_schedule(dataset, warmup + steps, batch_size)
    finally:
        dataset.close()
    try:
        result = ablations.run_backbone_only(config, schedule, device)
        result["seq_len"] = seq_len
        result["batch_size"] = batch_size
        result["oom"] = False
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        result = {"seq_len": seq_len, "batch_size": batch_size, "oom": True, "error": str(exc)[:200]}
    gc.collect(); torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-json", default="benchmark_results/phase5/backbone_seqlen_scan.json")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    manifest_path = Path(args.manifest).expanduser().resolve()

    # (seq_len, batch_size) chosen to keep batch*seq_len roughly bounded while
    # pushing seq_len itself up, since attention cost depends on seq_len not
    # batch*seq_len.
    configs = [(512, 2), (1024, 2), (2048, 2), (4096, 1), (8192, 1)]
    results = []
    for seq_len, batch_size in configs:
        print(f"\n=== seq_len={seq_len} batch={batch_size} ===", flush=True)
        r = run_one(seq_len, batch_size, manifest_path, device, args.warmup, args.steps)
        results.append(r)
        if r.get("oom"):
            print(f"OOM: {r['error']}")
        else:
            print(f"tok/s={r['tokens_per_second']:.1f}  fwd={r['mean_forward_ms']:.3f}ms  "
                  f"bwd={r['mean_backward_ms']:.3f}ms  peak_vram={r['peak_allocated_mib']:.1f}MiB")

    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"\nWrote {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
