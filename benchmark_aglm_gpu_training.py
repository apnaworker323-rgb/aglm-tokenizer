#!/usr/bin/env python3
"""Bounded, deterministic GPU training benchmark for verified AGLM uint32 shards.

This is deliberately a benchmark, not a long-running trainer.  It consumes exact
1.55M-vocabulary token IDs without truncation or modulo mapping and compares two
input paths over the same batch schedule:

* ``resident``: the bounded benchmark schedule is materialized in CPU RAM first.
* ``mmap``: every batch is fetched from the canonical uint32 shard loader.

The model uses a compact causal Transformer and PyTorch's exact hierarchical
``AdaptiveLogSoftmaxWithLoss``.  The adaptive output avoids materializing a
``batch * sequence * 1,551,017`` dense-logit tensor, which is not viable on a
6 GiB GPU.  It changes the output parameterization, but not token IDs or targets.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from aglm_dataset_loader import AGLMShardedDataset


EXPECTED_VOCAB_SIZE = 1_551_017
EXPECTED_MAX_ID = 1_551_016
EXPECTED_TOKENIZER_SHA256 = "1f865241d0f3ebcc41bc2e75de8eb6ef190dd23e2fa5a444f69d40ad250c74eb"


@dataclass(frozen=True)
class BenchmarkConfig:
    manifest: str
    split: str
    seq_len: int
    batch_size: int
    warmup_steps: int
    steps: int
    seed: int
    d_model: int
    d_lexical: int
    n_layers: int
    n_heads: int
    ffn_multiple: float
    precision: str
    learning_rate: float
    weight_decay: float
    clip_grad: float
    cutoffs: Tuple[int, ...]


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


class CausalTransformerBlock(nn.Module):
    def __init__(self, width: int, heads: int, ffn_multiple: float) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.attn_norm = RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.attn_out = nn.Linear(width, width, bias=False)
        hidden = int(math.ceil(width * ffn_multiple / 64.0) * 64)
        self.ffn_norm = RMSNorm(width)
        self.ffn_gate_up = nn.Linear(width, 2 * hidden, bias=False)
        self.ffn_down = nn.Linear(hidden, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        residual = x
        qkv = self.qkv(self.attn_norm(x))
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = attended.transpose(1, 2).contiguous().view(batch, length, self.width)
        x = residual + self.attn_out(attended)
        gate, up = self.ffn_gate_up(self.ffn_norm(x)).chunk(2, dim=-1)
        return x + self.ffn_down(F.silu(gate) * up)


class AGLMBenchmarkLM(nn.Module):
    """Full-ID input model with a memory-safe, normalized hierarchical LM loss."""

    def __init__(self, config: BenchmarkConfig) -> None:
        super().__init__()
        self.lexical = nn.Embedding(EXPECTED_VOCAB_SIZE, config.d_lexical)
        self.lexical_projection = nn.Linear(config.d_lexical, config.d_model, bias=False)
        self.blocks = nn.ModuleList(
            CausalTransformerBlock(config.d_model, config.n_heads, config.ffn_multiple)
            for _ in range(config.n_layers)
        )
        self.final_norm = RMSNorm(config.d_model)
        self.output = nn.AdaptiveLogSoftmaxWithLoss(
            config.d_model,
            EXPECTED_VOCAB_SIZE,
            list(config.cutoffs),
            div_value=4.0,
            head_bias=False,
        )
        nn.init.normal_(self.lexical.weight, mean=0.0, std=0.02)

    def hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.lexical_projection(self.lexical(input_ids))
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)

    def loss(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.hidden(input_ids).reshape(-1, self.final_norm.weight.numel())
        targets = target_ids.reshape(-1)
        return self.output(hidden, targets).loss


class NvidiaSmiMonitor:
    """Best-effort background telemetry; failures never invalidate timing."""

    def __init__(self, interval: float = 0.2) -> None:
        self.interval = interval
        self.samples: List[Tuple[float, float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                line = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=2).splitlines()[0]
                util, memory, power = (float(value.strip()) for value in line.split(",")[:3])
                self.samples.append((util, memory, power))
            except (FileNotFoundError, IndexError, ValueError, subprocess.SubprocessError):
                pass
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="nvidia-smi-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, float | int | None]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        if not self.samples:
            return {"samples": 0, "gpu_util_mean_pct": None, "gpu_util_max_pct": None,
                    "memory_used_max_mib": None, "power_mean_w": None}
        utils, memory, power = zip(*self.samples)
        return {
            "samples": len(self.samples),
            "gpu_util_mean_pct": statistics.fmean(utils),
            "gpu_util_max_pct": max(utils),
            "memory_used_max_mib": max(memory),
            "power_mean_w": statistics.fmean(power),
        }


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _load_and_validate_manifest(path: Path, split: str) -> Tuple[Dict[str, Any], float]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    tokenizer = manifest.get("tokenizer", {})
    failures = []
    if manifest.get("complete") is not True:
        failures.append("manifest.complete is not true")
    if manifest.get("sample_run") is not False:
        failures.append("manifest is not a completed full-corpus conversion")
    if manifest.get("numpy_dtype") != "<u4" or manifest.get("endian") != "little":
        failures.append("manifest is not canonical little-endian uint32")
    if int(tokenizer.get("vocab_size", -1)) != EXPECTED_VOCAB_SIZE:
        failures.append("tokenizer vocab size mismatch")
    if int(tokenizer.get("max_token_id", -1)) != EXPECTED_MAX_ID:
        failures.append("tokenizer maximum ID mismatch")
    if tokenizer.get("sha256") != EXPECTED_TOKENIZER_SHA256:
        failures.append("tokenizer SHA256 mismatch")
    if not manifest.get("shards", {}).get(split):
        failures.append(f"manifest has no {split} shards")
    if failures:
        raise RuntimeError("Refusing GPU benchmark: " + "; ".join(failures))
    stats = manifest.get("statistics", {})
    raw_bytes_per_token = float(stats.get("raw_bytes_per_token", 0.0))
    if not math.isfinite(raw_bytes_per_token) or raw_bytes_per_token <= 0:
        raise RuntimeError("manifest has no valid measured raw_bytes_per_token")
    return manifest, raw_bytes_per_token


def _build_schedule(
    dataset: AGLMShardedDataset,
    batches: int,
    batch_size: int,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], Dict[str, Any]]:
    schedule: List[Tuple[torch.Tensor, torch.Tensor]] = []
    digest = hashlib.sha256()
    minimum = EXPECTED_MAX_ID
    maximum = 0
    for batch_index in range(batches):
        inputs, targets, _ = dataset.get_batch(
            batch_size=batch_size,
            start_index=batch_index * batch_size,
        )
        batch_min = min(int(inputs.min()), int(targets.min()))
        batch_max = max(int(inputs.max()), int(targets.max()))
        if batch_min < 0 or batch_max >= EXPECTED_VOCAB_SIZE:
            raise RuntimeError(f"out-of-range schedule IDs: min={batch_min}, max={batch_max}")
        minimum = min(minimum, batch_min)
        maximum = max(maximum, batch_max)
        digest.update(inputs.numpy().astype("<i8", copy=False).tobytes())
        digest.update(targets.numpy().astype("<i8", copy=False).tobytes())
        schedule.append((inputs, targets))
    return schedule, {
        "sha256": digest.hexdigest(),
        "batches": batches,
        "samples": batches * batch_size,
        "min_token_id": minimum,
        "max_token_id": maximum,
    }


def _make_optimizer(model: nn.Module, config: BenchmarkConfig) -> Tuple[torch.optim.Optimizer, bool]:
    kwargs: Dict[str, Any] = {
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
        "betas": (0.9, 0.95),
    }
    try:
        return torch.optim.AdamW(model.parameters(), fused=True, **kwargs), True
    except (RuntimeError, TypeError):
        return torch.optim.AdamW(model.parameters(), **kwargs), False


def _autocast_dtype(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def _run_mode(
    mode: str,
    config: BenchmarkConfig,
    dataset: AGLMShardedDataset,
    schedule: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    raw_bytes_per_token: float,
    device: torch.device,
) -> Dict[str, Any]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    model = AGLMBenchmarkLM(config).to(device)
    optimizer, fused_optimizer = _make_optimizer(model, config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    dtype = _autocast_dtype(config.precision)
    autocast_enabled = config.precision != "fp32"
    scaler = torch.amp.GradScaler("cuda", enabled=config.precision == "fp16")

    def cpu_batch(batch_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if mode == "resident":
            return schedule[batch_index]
        inputs, targets, _ = dataset.get_batch(
            batch_size=config.batch_size,
            start_index=batch_index * config.batch_size,
        )
        # Prove that the separately fetched mmap batch is the scheduled batch.
        expected_inputs, expected_targets = schedule[batch_index]
        if not torch.equal(inputs, expected_inputs) or not torch.equal(targets, expected_targets):
            raise RuntimeError(f"mmap schedule mismatch at batch {batch_index}")
        return inputs, targets

    model.train()
    step_seconds: List[float] = []
    data_seconds: List[float] = []
    losses: List[float] = []
    total_batches = config.warmup_steps + config.steps
    torch.cuda.reset_peak_memory_stats(device)
    monitor = NvidiaSmiMonitor()
    timed_started = 0.0
    try:
        for batch_index in range(total_batches):
            data_started = time.perf_counter()
            cpu_inputs, cpu_targets = cpu_batch(batch_index)
            inputs = cpu_inputs.to(device, non_blocking=False)
            targets = cpu_targets.to(device, non_blocking=False)
            torch.cuda.synchronize(device)
            data_elapsed = time.perf_counter() - data_started

            optimizer.zero_grad(set_to_none=True)
            step_started = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=autocast_enabled):
                loss = model.loss(inputs, targets)
            scaler.scale(loss).backward()
            if config.clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize(device)
            step_elapsed = time.perf_counter() - step_started

            if batch_index == config.warmup_steps - 1:
                monitor.start()
                timed_started = time.perf_counter()
            if batch_index >= config.warmup_steps:
                step_seconds.append(step_elapsed)
                data_seconds.append(data_elapsed)
                losses.append(float(loss.detach().cpu()))
                completed = batch_index - config.warmup_steps + 1
                print(
                    f"[{mode}] {completed:03d}/{config.steps:03d} "
                    f"loss={losses[-1]:.5f} step={step_elapsed * 1000:.1f}ms "
                    f"data={data_elapsed * 1000:.1f}ms",
                    flush=True,
                )
        torch.cuda.synchronize(device)
        timed_elapsed = time.perf_counter() - timed_started
    finally:
        telemetry = monitor.stop()

    tokens = config.steps * config.batch_size * config.seq_len
    total_with_data = timed_elapsed
    compute_seconds = sum(step_seconds)
    data_path_seconds = sum(data_seconds)
    result: Dict[str, Any] = {
        "mode": mode,
        "input_schedule_sha256": None,
        "tokens": tokens,
        "timed_steps": config.steps,
        "elapsed_seconds_including_data": total_with_data,
        "summed_gpu_step_seconds": compute_seconds,
        "summed_data_path_seconds": data_path_seconds,
        "tokens_per_second": tokens / max(total_with_data, 1e-9),
        "effective_raw_mib_per_second": tokens * raw_bytes_per_token / (1 << 20) / max(total_with_data, 1e-9),
        "steps_per_second": config.steps / max(total_with_data, 1e-9),
        "step_ms_p50": _percentile(step_seconds, 0.50) * 1000,
        "step_ms_p95": _percentile(step_seconds, 0.95) * 1000,
        "data_ms_p50": _percentile(data_seconds, 0.50) * 1000,
        "data_ms_p95": _percentile(data_seconds, 0.95) * 1000,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "fused_adamw": fused_optimizer,
        "torch_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1 << 20),
        "torch_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1 << 20),
        "nvidia_smi": telemetry,
    }
    del optimizer, model, scaler, inputs, targets
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _gpu_identity() -> Dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }


def _write_reports(output_json: Path, payload: Dict[str, Any]) -> Path:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_json.with_suffix(output_json.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, output_json)

    report_path = output_json.with_suffix(".md")
    modes = {row["mode"]: row for row in payload["results"]}
    resident = modes["resident"]
    mmap = modes["mmap"]
    overhead = (resident["tokens_per_second"] - mmap["tokens_per_second"]) / resident["tokens_per_second"] * 100
    conversion_headroom = payload["dataset"]["conversion_raw_mib_per_second"] / mmap["effective_raw_mib_per_second"]
    benchmark_epoch_hours = payload["dataset"]["total_tokens"] / mmap["tokens_per_second"] / 3600
    markdown = f"""# AGLM Identical-Data GPU Training Benchmark

Status: **PASSED**  
Dataset manifest: `{payload['config']['manifest']}`  
Input schedule SHA256 (identical in both runs): `{payload['schedule']['sha256']}`

## Result

| Input path | tokens/s | effective raw MiB/s | p50 step ms | p95 step ms | peak allocated MiB | mean GPU util |
|---|---:|---:|---:|---:|---:|---:|
| Resident bounded schedule | {resident['tokens_per_second']:.1f} | {resident['effective_raw_mib_per_second']:.3f} | {resident['step_ms_p50']:.2f} | {resident['step_ms_p95']:.2f} | {resident['torch_peak_allocated_mib']:.1f} | {resident['nvidia_smi']['gpu_util_mean_pct']} |
| uint32 mmap shards | {mmap['tokens_per_second']:.1f} | {mmap['effective_raw_mib_per_second']:.3f} | {mmap['step_ms_p50']:.2f} | {mmap['step_ms_p95']:.2f} | {mmap['torch_peak_allocated_mib']:.1f} | {mmap['nvidia_smi']['gpu_util_mean_pct']} |

Mmap end-to-end throughput difference from resident compute ceiling: **{overhead:.2f}%**.

## Interpretation

- The canonical mmap path is within **{overhead:.2f}%** of the bounded resident-data compute ceiling, so shard I/O is not the limiting component in this run.
- Dataset conversion ran **{conversion_headroom:.1f}x faster** in raw-byte terms than this GPU training benchmark. Pre-tokenization can therefore stay comfortably ahead of this exact benchmark workload.
- At the measured mmap rate, one pass over all {payload['dataset']['total_tokens']:,} token positions would take about **{benchmark_epoch_hours:.2f} hours**. This is a projection for this compact benchmark model and configuration, not a final-model training estimate.

## Experimental contract

- Both runs start from the same deterministic seed and consume the exact same input/target batch schedule.
- No token is truncated, cast to uint16, clamped, or reduced modulo another vocabulary.
- The input embedding directly addresses all 1,551,017 AGLM IDs.
- The loss is normalized hierarchical adaptive softmax over all 1,551,017 targets. It is memory-safe on this GPU, but is not a flat dense-softmax parameterization.
- This is a bounded forward + backward + gradient clipping + AdamW benchmark. It is not a full training run and writes no checkpoint.
- Effective raw MiB/s uses the completed corpus manifest's measured `{payload['raw_bytes_per_token']:.6f}` raw bytes/token.

## Hardware and model

- GPU: {payload['gpu']['name']} ({payload['gpu']['total_memory_bytes'] / (1 << 20):.0f} MiB)
- PyTorch/CUDA: {payload['gpu']['torch_version']} / {payload['gpu']['torch_cuda_version']}
- Precision: {payload['config']['precision']}
- Shape: batch={payload['config']['batch_size']}, seq_len={payload['config']['seq_len']}, d_model={payload['config']['d_model']}, layers={payload['config']['n_layers']}, heads={payload['config']['n_heads']}
- Trainable parameters: {resident['trainable_parameter_count']:,}
- Warmup/timed steps per path: {payload['config']['warmup_steps']} / {payload['config']['steps']}
"""
    report_tmp = report_path.with_suffix(report_path.suffix + ".tmp")
    report_tmp.write_text(markdown, encoding="utf-8")
    with report_tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(report_tmp, report_path)
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-json", default="benchmark_results/aglm_gpu_training_benchmark.json")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if min(args.seq_len, args.batch_size, args.warmup_steps, args.steps, args.d_model,
           args.d_lexical, args.layers, args.heads) <= 0:
        raise ValueError("all size and step arguments must be positive")
    cutoffs = tuple(int(value) for value in args.cutoffs.split(",") if value.strip())
    if not cutoffs or tuple(sorted(set(cutoffs))) != cutoffs or cutoffs[-1] >= EXPECTED_VOCAB_SIZE:
        raise ValueError("--cutoffs must be increasing, unique, and below the vocabulary size")
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest, raw_bytes_per_token = _load_and_validate_manifest(manifest_path, args.split)
    config = BenchmarkConfig(
        manifest=str(manifest_path), split=args.split, seq_len=args.seq_len,
        batch_size=args.batch_size, warmup_steps=args.warmup_steps, steps=args.steps,
        seed=args.seed, d_model=args.d_model, d_lexical=args.d_lexical,
        n_layers=args.layers, n_heads=args.heads, ffn_multiple=args.ffn_multiple,
        precision=args.precision, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, clip_grad=args.clip_grad, cutoffs=cutoffs,
    )
    device = torch.device("cuda:0")
    dataset = AGLMShardedDataset(
        manifest=str(manifest_path), split=args.split, seq_len=args.seq_len, seed=args.seed,
    )
    try:
        schedule, schedule_info = _build_schedule(
            dataset, args.warmup_steps + args.steps, args.batch_size,
        )
        print(json.dumps({"gpu": _gpu_identity(), "schedule": schedule_info}, indent=2), flush=True)
        results = []
        for mode in ("resident", "mmap"):
            print(f"\nStarting identical-data mode: {mode}", flush=True)
            row = _run_mode(mode, config, dataset, schedule, raw_bytes_per_token, device)
            row["input_schedule_sha256"] = schedule_info["sha256"]
            results.append(row)
    finally:
        dataset.close()

    payload = {
        "status": "PASSED",
        "created_at_unix": time.time(),
        "config": asdict(config),
        "gpu": _gpu_identity(),
        "raw_bytes_per_token": raw_bytes_per_token,
        "dataset": {
            "raw_utf8_bytes": int(manifest["statistics"]["raw_utf8_bytes"]),
            "total_tokens": int(manifest["statistics"]["total_tokens"]),
            "train_tokens": int(manifest["statistics"]["train_tokens"]),
            "val_tokens": int(manifest["statistics"]["val_tokens"]),
            "conversion_raw_mib_per_second": float(manifest["statistics"]["raw_tokenized_mb_s"]),
        },
        "schedule": schedule_info,
        "results": results,
    }
    output_json = Path(args.output_json).expanduser().resolve()
    report_path = _write_reports(output_json, payload)
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Reports: {output_json} and {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
