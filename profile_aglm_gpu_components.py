#!/usr/bin/env python3
"""Phase 1 component-level GPU profiler for the frozen Patch-1 AGLM production baseline.

This is instrumentation ONLY. It imports the exact model classes from
``benchmark_aglm_gpu_training.py`` unmodified and monkey-patches their ``forward``
methods at call time to insert ``torch.cuda.Event`` markers around logically
distinct sub-computations (embedding, per-block attention, per-block FFN,
per-block norms, final norm, output head). The arithmetic performed is
byte-for-byte identical to the production benchmark; nothing about the
architecture is changed. Full-backward hooks on the same module boundaries
recover an equivalent breakdown for the backward pass, in true firing order.

A second, independent pass uses ``torch.profiler`` (CPU+CUDA activities) to
cross-check with an operator-level table, kernel counts, and peak/allocated
VRAM. This is not a trainer: no checkpoint is written and no optimizer state
is persisted after the run.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function

import benchmark_aglm_gpu_training as bench

EXPECTED_VOCAB_SIZE = bench.EXPECTED_VOCAB_SIZE

# Captured before any monkey-patching so the profiler cross-check pass (Pass 2)
# can run the untouched, un-instrumented forward methods.
_ORIGINAL_BLOCK_FORWARD = bench.CausalTransformerBlock.forward
_ORIGINAL_HIDDEN = bench.AGLMBenchmarkLM.hidden
_ORIGINAL_LOSS = bench.AGLMBenchmarkLM.loss


# --------------------------------------------------------------------------
# Ordered event log
# --------------------------------------------------------------------------

class StepTimer:
    """Collects (label, cuda.Event) pairs in true execution order for one step."""

    def __init__(self) -> None:
        self.forward: List[Tuple[str, torch.cuda.Event, int]] = []
        self.backward: List[Tuple[str, torch.cuda.Event, int]] = []
        self.mem_allocated: Dict[str, int] = {}

    def mark_forward(self, label: str, device: torch.device) -> None:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.forward.append((label, event, torch.cuda.memory_allocated(device)))

    def mark_backward(self, label: str, device: torch.device) -> None:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.backward.append((label, event, torch.cuda.memory_allocated(device)))


ACTIVE: List[StepTimer] = []  # single-slot stack; forward is synchronous/single-threaded


def _timer() -> StepTimer:
    return ACTIVE[-1]


# --------------------------------------------------------------------------
# Instrumented forward methods (identical math; extra event markers only)
# --------------------------------------------------------------------------

def instrumented_block_forward(self: bench.CausalTransformerBlock, x: torch.Tensor) -> torch.Tensor:
    device = x.device
    batch, length, _ = x.shape
    residual = x
    normed = self.attn_norm(x)
    _timer().mark_forward(f"{self._profile_name}.attn_norm", device)
    qkv = self.qkv(normed)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
    k = k.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
    v = v.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
    attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    attended = attended.transpose(1, 2).contiguous().view(batch, length, self.width)
    x = residual + self.attn_out(attended)
    _timer().mark_forward(f"{self._profile_name}.attention", device)
    normed2 = self.ffn_norm(x)
    _timer().mark_forward(f"{self._profile_name}.ffn_norm", device)
    gate, up = self.ffn_gate_up(normed2).chunk(2, dim=-1)
    out = x + self.ffn_down(F.silu(gate) * up)
    _timer().mark_forward(f"{self._profile_name}.ffn", device)
    return out


def instrumented_hidden(self: bench.AGLMBenchmarkLM, input_ids: torch.Tensor) -> torch.Tensor:
    device = input_ids.device
    x = self.lexical(input_ids)
    _timer().mark_forward("embed.lookup", device)
    x = self.lexical_projection(x)
    _timer().mark_forward("embed.projection", device)
    for block in self.blocks:
        x = block(x)
    x = self.final_norm(x)
    _timer().mark_forward("final_norm", device)
    return x


def instrumented_loss(self: bench.AGLMBenchmarkLM, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    hidden = self.hidden(input_ids).reshape(-1, self.final_norm.weight.numel())
    targets = target_ids.reshape(-1)
    out = self.output(hidden, targets).loss
    _timer().mark_forward("output_head", hidden.device)
    return out


# --------------------------------------------------------------------------
# Backward hooks (module-boundary, true firing order == reverse exec order)
# --------------------------------------------------------------------------

def _install_backward_hooks(model: bench.AGLMBenchmarkLM, device: torch.device) -> List[Any]:
    handles: List[Any] = []

    def make_hook(label: str):
        def hook(module, grad_input, grad_output):
            _timer().mark_backward(label, device)
        return hook

    handles.append(model.output.register_full_backward_hook(make_hook("output_head")))
    handles.append(model.final_norm.register_full_backward_hook(make_hook("final_norm")))
    for block in reversed(list(model.blocks)):
        name = block._profile_name
        handles.append(block.ffn_gate_up.register_full_backward_hook(make_hook(f"{name}.ffn")))
        handles.append(block.ffn_norm.register_full_backward_hook(make_hook(f"{name}.ffn_norm")))
        handles.append(block.qkv.register_full_backward_hook(make_hook(f"{name}.attention")))
        handles.append(block.attn_norm.register_full_backward_hook(make_hook(f"{name}.attn_norm")))
    handles.append(model.lexical_projection.register_full_backward_hook(make_hook("embed.projection")))
    handles.append(model.lexical.register_full_backward_hook(make_hook("embed.lookup")))
    return handles


# --------------------------------------------------------------------------
# Component bucketing for the report
# --------------------------------------------------------------------------

def _bucket(label: str) -> str:
    if label.startswith("embed.lookup"):
        return "input_embedding"
    if label.startswith("embed.projection"):
        return "embedding_projection"
    if label.endswith(".attention"):
        return "attention"
    if label.endswith(".ffn"):
        return "ffn"
    if label.endswith(".attn_norm") or label.endswith(".ffn_norm") or label == "final_norm":
        return "normalization"
    if label == "output_head":
        return "output_head_adaptive_softmax"
    return "other"


@dataclass
class ComponentAccumulator:
    forward_ms: Dict[str, List[float]] = field(default_factory=lambda: {})
    backward_ms: Dict[str, List[float]] = field(default_factory=lambda: {})

    def add_forward(self, bucket: str, ms: float) -> None:
        self.forward_ms.setdefault(bucket, []).append(ms)

    def add_backward(self, bucket: str, ms: float) -> None:
        self.backward_ms.setdefault(bucket, []).append(ms)


def _consume_step_timer(timer: StepTimer, acc: ComponentAccumulator, step_start_event: torch.cuda.Event,
                         backward_start_event: torch.cuda.Event) -> Dict[str, float]:
    """Convert one step's ordered event log into per-bucket forward/backward ms."""
    prev_event = step_start_event
    fwd_total = 0.0
    for label, event, _mem in timer.forward:
        delta = prev_event.elapsed_time(event)
        acc.add_forward(_bucket(label), delta)
        fwd_total += delta
        prev_event = event

    prev_event = backward_start_event
    bwd_total = 0.0
    for label, event, _mem in timer.backward:
        delta = prev_event.elapsed_time(event)
        acc.add_backward(_bucket(label), delta)
        bwd_total += delta
        prev_event = event

    return {"forward_sum_ms": fwd_total, "backward_sum_ms": bwd_total}


# --------------------------------------------------------------------------
# Main measurement loop (manual CUDA-event pass)
# --------------------------------------------------------------------------

def run_event_pass(args: argparse.Namespace) -> Dict[str, Any]:
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

    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    model = bench.AGLMBenchmarkLM(config).to(device)
    for index, block in enumerate(model.blocks):
        block._profile_name = f"block{index}"

    # Monkey-patch (instrumentation only; identical math) on the CLASS so bound
    # calls inside the model use the timed versions without editing the frozen file.
    bench.CausalTransformerBlock.forward = instrumented_block_forward
    bench.AGLMBenchmarkLM.hidden = instrumented_hidden
    bench.AGLMBenchmarkLM.loss = instrumented_loss

    hook_handles = _install_backward_hooks(model, device)

    optimizer, fused_optimizer = bench._make_optimizer(model, config)
    parameter_count = sum(p.numel() for p in model.parameters())
    dtype = bench._autocast_dtype(config.precision)
    autocast_enabled = config.precision != "fp32"
    scaler = torch.amp.GradScaler("cuda", enabled=config.precision == "fp16")

    acc = ComponentAccumulator()
    optimizer_ms: List[float] = []
    data_ms: List[float] = []
    wall_step_ms: List[float] = []
    forward_wall_ms: List[float] = []
    backward_wall_ms: List[float] = []
    losses: List[float] = []
    peak_mem_by_step: List[float] = []

    monitor = bench.NvidiaSmiMonitor()
    total_batches = config.warmup_steps + config.steps
    torch.cuda.reset_peak_memory_stats(device)

    model.train()
    for batch_index in range(total_batches):
        timer = StepTimer()
        ACTIVE.append(timer)

        data_started = time.perf_counter()
        cpu_inputs, cpu_targets = schedule[batch_index]
        inputs = cpu_inputs.to(device, non_blocking=False)
        targets = cpu_targets.to(device, non_blocking=False)
        torch.cuda.synchronize(device)
        data_elapsed_ms = (time.perf_counter() - data_started) * 1000.0

        optimizer.zero_grad(set_to_none=True)

        step_start_event = torch.cuda.Event(enable_timing=True)
        step_start_event.record()
        wall_step_started = time.perf_counter()

        with torch.autocast(device_type="cuda", dtype=dtype, enabled=autocast_enabled):
            loss = model.loss(inputs, targets)

        forward_done_event = torch.cuda.Event(enable_timing=True)
        forward_done_event.record()

        backward_start_event = torch.cuda.Event(enable_timing=True)
        backward_start_event.record()
        scaler.scale(loss).backward()
        backward_done_event = torch.cuda.Event(enable_timing=True)
        backward_done_event.record()

        optimizer_start_event = torch.cuda.Event(enable_timing=True)
        optimizer_start_event.record()
        if config.clip_grad > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        scaler.step(optimizer)
        scaler.update()
        optimizer_done_event = torch.cuda.Event(enable_timing=True)
        optimizer_done_event.record()

        torch.cuda.synchronize(device)
        wall_step_ms_value = (time.perf_counter() - wall_step_started) * 1000.0

        ACTIVE.pop()

        if batch_index >= config.warmup_steps:
            sums = _consume_step_timer(timer, acc, step_start_event, backward_start_event)
            optimizer_ms.append(optimizer_start_event.elapsed_time(optimizer_done_event))
            data_ms.append(data_elapsed_ms)
            wall_step_ms.append(wall_step_ms_value)
            forward_wall_ms.append(step_start_event.elapsed_time(forward_done_event))
            backward_wall_ms.append(backward_start_event.elapsed_time(backward_done_event))
            losses.append(float(loss.detach().cpu()))
            peak_mem_by_step.append(torch.cuda.max_memory_allocated(device) / (1 << 20))

            if len(timer.forward) == 0 or len(timer.backward) == 0:
                raise RuntimeError("instrumentation captured zero events; hooks did not fire")

            expected_bwd_events = 4 * config.n_layers + 4  # per block: ffn,ffn_norm,attn,attn_norm + output,final_norm,proj,lookup
            if len(timer.backward) != expected_bwd_events:
                raise RuntimeError(
                    f"backward hook count mismatch: got {len(timer.backward)}, expected {expected_bwd_events}. "
                    f"labels={[l for l, _, _ in timer.backward]}"
                )

        if batch_index == config.warmup_steps - 1:
            monitor.start()
        print(f"[event-pass] {batch_index + 1:03d}/{total_batches:03d} loss={float(loss.detach().cpu()):.5f} "
              f"wall={wall_step_ms_value:.1f}ms", flush=True)

    telemetry = monitor.stop()
    for handle in hook_handles:
        handle.remove()

    tokens = config.steps * config.batch_size * config.seq_len
    total_wall_s = sum(wall_step_ms) / 1000.0

    def summarize(values: List[float]) -> Dict[str, float]:
        return {
            "mean_ms": statistics.fmean(values) if values else 0.0,
            "median_ms": statistics.median(values) if values else 0.0,
            "stdev_ms": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "sum_ms": sum(values),
        }

    component_rows = []
    all_buckets = sorted(set(list(acc.forward_ms.keys()) + list(acc.backward_ms.keys())))
    grand_total_ms = sum(wall_step_ms)
    for bucket in all_buckets:
        fwd = summarize(acc.forward_ms.get(bucket, []))
        bwd = summarize(acc.backward_ms.get(bucket, []))
        total_mean = fwd["mean_ms"] + bwd["mean_ms"]
        component_rows.append({
            "component": bucket,
            "forward_mean_ms": fwd["mean_ms"],
            "backward_mean_ms": bwd["mean_ms"],
            "total_mean_ms": total_mean,
            "pct_of_step": total_mean / (grand_total_ms / config.steps) * 100.0,
            "forward_sum_ms": fwd["sum_ms"],
            "backward_sum_ms": bwd["sum_ms"],
        })
    component_rows.sort(key=lambda r: -r["total_mean_ms"])

    other_mean = statistics.fmean(optimizer_ms)
    data_mean = statistics.fmean(data_ms)
    component_rows.append({
        "component": "optimizer_step",
        "forward_mean_ms": 0.0, "backward_mean_ms": 0.0, "total_mean_ms": other_mean,
        "pct_of_step": other_mean / (grand_total_ms / config.steps) * 100.0,
        "forward_sum_ms": 0.0, "backward_sum_ms": sum(optimizer_ms),
    })
    component_rows.append({
        "component": "h2d_data_copy",
        "forward_mean_ms": 0.0, "backward_mean_ms": 0.0, "total_mean_ms": data_mean,
        "pct_of_step": data_mean / (grand_total_ms / config.steps) * 100.0,
        "forward_sum_ms": 0.0, "backward_sum_ms": sum(data_ms),
    })

    accounted_mean = sum(r["total_mean_ms"] for r in component_rows)
    measured_mean_step = grand_total_ms / config.steps
    misc_overhead = measured_mean_step - accounted_mean

    result = {
        "config": {
            "manifest": str(manifest_path), "split": args.split, "seq_len": args.seq_len,
            "batch_size": args.batch_size, "warmup_steps": args.warmup_steps, "steps": args.steps,
            "d_model": args.d_model, "d_lexical": args.d_lexical, "layers": args.layers,
            "heads": args.heads, "precision": args.precision, "cutoffs": cutoffs,
        },
        "gpu": bench._gpu_identity(),
        "parameter_count": parameter_count,
        "fused_adamw": fused_optimizer,
        "tokens_per_second": tokens / max(total_wall_s, 1e-9),
        "effective_raw_mib_per_second": tokens * raw_bytes_per_token / (1 << 20) / max(total_wall_s, 1e-9),
        "measured_mean_step_ms": measured_mean_step,
        "measured_p50_step_ms": bench._percentile(wall_step_ms, 0.50),
        "measured_p95_step_ms": bench._percentile(wall_step_ms, 0.95),
        "mean_forward_wall_ms": statistics.fmean(forward_wall_ms),
        "mean_backward_wall_ms": statistics.fmean(backward_wall_ms),
        "accounted_component_mean_ms": accounted_mean,
        "misc_overhead_mean_ms": misc_overhead,
        "misc_overhead_pct": misc_overhead / measured_mean_step * 100.0,
        "components": component_rows,
        "torch_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1 << 20),
        "torch_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1 << 20),
        "peak_mem_trajectory_mib": {
            "first_measured_step": peak_mem_by_step[0] if peak_mem_by_step else None,
            "last_measured_step": peak_mem_by_step[-1] if peak_mem_by_step else None,
        },
        "nvidia_smi": telemetry,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
    }

    del optimizer, model, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return result


# --------------------------------------------------------------------------
# torch.profiler cross-check pass (op-level table, kernel count)
# --------------------------------------------------------------------------

def run_profiler_pass(args: argparse.Namespace, trace_out: Path) -> Dict[str, Any]:
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
        schedule, _ = bench._build_schedule(dataset, config.warmup_steps + 16, config.batch_size)
    finally:
        dataset.close()

    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # Use the ORIGINAL (un-patched) forward methods for this pass.
    bench.CausalTransformerBlock.forward = _ORIGINAL_BLOCK_FORWARD
    bench.AGLMBenchmarkLM.hidden = _ORIGINAL_HIDDEN
    bench.AGLMBenchmarkLM.loss = _ORIGINAL_LOSS

    model = bench.AGLMBenchmarkLM(config).to(device)
    optimizer, _ = bench._make_optimizer(model, config)
    dtype = bench._autocast_dtype(config.precision)
    scaler = torch.amp.GradScaler("cuda", enabled=config.precision == "fp16")

    def step(batch_index: int) -> None:
        cpu_inputs, cpu_targets = schedule[batch_index]
        inputs = cpu_inputs.to(device)
        targets = cpu_targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=config.precision != "fp32"):
            with record_function("backbone_region"):
                hidden = model.hidden(inputs).reshape(-1, config.d_model)
                targets_flat = targets.reshape(-1)
            with record_function("output_head_forward_region"):
                loss = model.output(hidden, targets_flat).loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    for batch_index in range(config.warmup_steps):
        step(batch_index)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    active_steps = 10
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for offset in range(active_steps):
            step(config.warmup_steps + offset)
            torch.cuda.synchronize(device)

    trace_out.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(trace_out))

    kernel_events = 0
    try:
        with open(trace_out, "r", encoding="utf-8") as handle:
            trace = json.load(handle)
        for event in trace.get("traceEvents", []):
            if event.get("cat") in ("kernel", "Kernel") or event.get("cat", "").startswith("cuda"):
                kernel_events += 1
    except Exception:
        kernel_events = -1

    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=40)
    peak_mib = torch.cuda.max_memory_allocated(device) / (1 << 20)
    allocated_mib = torch.cuda.memory_allocated(device) / (1 << 20)

    del optimizer, model, scaler
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "active_steps": active_steps,
        "total_kernel_events": kernel_events,
        "kernel_events_per_step": kernel_events / active_steps if kernel_events > 0 else None,
        "peak_allocated_mib": peak_mib,
        "allocated_mib_after": allocated_mib,
        "table_top40_by_cuda_time": table,
        "trace_path": str(trace_out),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-json", default="benchmark_results/phase1_component_profile.json")
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
    parser.add_argument("--skip-profiler-pass", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    print("=== PASS 1: CUDA-event component timing ===", flush=True)
    event_result = run_event_pass(args)

    profiler_result = None
    if not args.skip_profiler_pass:
        print("\n=== PASS 2: torch.profiler operator cross-check ===", flush=True)
        trace_out = Path(args.output_json).with_suffix("").parent / "phase1_trace.json"
        profiler_result = run_profiler_pass(args, trace_out)
        print(profiler_result["table_top40_by_cuda_time"], flush=True)

    payload = {
        "created_at_unix": time.time(),
        "event_pass": event_result,
        "profiler_pass": profiler_result,
    }
    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"\nWrote {output_json}", flush=True)

    print("\n=== COMPONENT SUMMARY (mean ms/step) ===")
    for row in event_result["components"]:
        print(f"{row['component']:32s} fwd={row['forward_mean_ms']:8.3f}ms  bwd={row['backward_mean_ms']:8.3f}ms  "
              f"total={row['total_mean_ms']:8.3f}ms  ({row['pct_of_step']:5.1f}%)")
    print(f"{'misc/overhead':32s} {'':8s}  {'':8s}  total={event_result['misc_overhead_mean_ms']:8.3f}ms  "
          f"({event_result['misc_overhead_pct']:5.1f}%)")
    print(f"\nmeasured mean step: {event_result['measured_mean_step_ms']:.3f} ms  "
          f"(forward wall {event_result['mean_forward_wall_ms']:.3f} ms + "
          f"backward wall {event_result['mean_backward_wall_ms']:.3f} ms + optimizer/data)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
