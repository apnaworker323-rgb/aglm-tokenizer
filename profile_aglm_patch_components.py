#!/usr/bin/env python3
"""Phase 10: Patch-1 vs Patch-2 component-level timing, same HierarchicalLatentLM
class and same hyperparameters as AGLM_LATENT_PATCHING_REPORT.md (d_model=160,
d_lexical=24, layers=6, heads=5, seq_len=256, batch=2, bf16), instrumented
(not modified) with CUDA-event checkpoints at four forward boundaries:

  patch_encode    -> lexical lookup + projection + patch-summary pooling
  global_backbone -> the causal Transformer stack over PATCH positions
  local_codec     -> the per-patch GRU that decodes individual tokens back out
  output_head     -> the real 1.55M-class AdaptiveLogSoftmaxWithLoss

plus one backward-total event pair. This isolates exactly which stage absorbs
the savings when the global (patch) sequence length is halved, answering why
Patch-2 measured ~0.98x end-to-end despite ~2x fewer global positions.

Instrumentation only: HierarchicalLatentLM's forward math is untouched;
methods are monkey-patched at call time to add event markers between existing
statements. Profiling only, no checkpoint is written.
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

import benchmark_aglm_gpu_training as bench
import aglm_latent_patching as patching

EXPECTED_VOCAB_SIZE = patching.VOCAB_SIZE

_ORIGINAL_TOKEN_HIDDEN = patching.HierarchicalLatentLM.token_hidden
_ORIGINAL_LOSS = patching.HierarchicalLatentLM.loss

_EVENTS: List[torch.cuda.Event] = []


def _mark(label: str) -> None:
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    _EVENTS.append((label, event))


def instrumented_token_hidden(self, patches, device):
    ids = patches.token_ids.to(device)
    mask = patches.token_mask.to(device)
    patch_mask = patches.patch_mask.to(device)
    batch, patch_count, width = ids.shape
    if patch_count > self.global_positions.shape[0]:
        raise ValueError("patch sequence exceeds configured global position table")

    local = self.lexical_projection(self.lexical(ids))
    local = local + self.slot_embedding[:width].view(1, 1, width, self.d_model)
    weights = mask.unsqueeze(-1).to(local.dtype)
    summaries = (local * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1)
    summaries = self.patch_projection(summaries)
    _mark("patch_encode")

    shifted = torch.empty_like(summaries)
    shifted[:, 0] = self.global_bos
    if patch_count > 1:
        shifted[:, 1:] = summaries[:, :-1]
    global_state = shifted + self.global_positions[:patch_count].unsqueeze(0)
    global_state = global_state * patch_mask.unsqueeze(-1).to(global_state.dtype)
    for block in self.blocks:
        global_state = block(global_state)
    conditions = self.global_norm(global_state)
    _mark("global_backbone")

    decoder_inputs = torch.empty_like(local)
    decoder_inputs[:, :, 0] = self.local_start
    if width > 1:
        decoder_inputs[:, :, 1:] = local[:, :, :-1]
    flat_inputs = decoder_inputs.reshape(batch * patch_count, width, self.d_model)
    initial = conditions.reshape(1, batch * patch_count, self.d_model).contiguous()
    decoded, _ = self.local_decoder(flat_inputs, initial)
    decoded = self.local_norm(decoded).reshape(batch, patch_count, width, self.d_model)
    _mark("local_codec")
    return decoded[mask], ids[mask]


def instrumented_loss(self, patches, device):
    hidden, targets = self.token_hidden(patches, device)
    out = self.output(hidden, targets).loss
    _mark("output_head")
    return out


def run_policy(policy_name: str, fixed_size: int, config: Dict[str, Any], real_ids_batches: List[torch.Tensor],
               device: torch.device, warmup: int, steps: int) -> Dict[str, Any]:
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    model = patching.HierarchicalLatentLM(
        d_model=config["d_model"], d_lexical=config["d_lexical"], n_layers=config["layers"],
        n_heads=config["heads"], max_global_positions=config["seq_len"], cutoffs=config["cutoffs"],
    ).to(device)
    model.train()

    patching.HierarchicalLatentLM.token_hidden = instrumented_token_hidden
    patching.HierarchicalLatentLM.loss = instrumented_loss

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.1,
                                   betas=(0.9, 0.95), fused=True)
    dtype = bench._autocast_dtype(config["precision"])
    patcher = patching.LosslessLatentPatcher()
    policy = patching.PatchPolicy(policy_name, "fixed", fixed_size=fixed_size)

    forward_sums: Dict[str, List[float]] = {}
    backward_ms: List[float] = []
    step_ms: List[float] = []
    global_position_counts: List[int] = []
    torch.cuda.reset_peak_memory_stats(device)

    total_batches = warmup + steps
    for batch_index in range(total_batches):
        ids = real_ids_batches[batch_index % len(real_ids_batches)]
        encoded = patcher.encode(ids, policy)
        global_position_counts.append(encoded.global_positions)

        optimizer.zero_grad(set_to_none=True)
        _EVENTS.clear()

        torch.cuda.synchronize(device)
        wall_started = time.perf_counter()
        step_start = torch.cuda.Event(enable_timing=True); step_start.record()
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=config["precision"] != "fp32"):
            loss = model.loss(encoded, device)
        bwd_start = torch.cuda.Event(enable_timing=True); bwd_start.record()
        loss.backward()
        bwd_end = torch.cuda.Event(enable_timing=True); bwd_end.record()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize(device)
        wall_elapsed = (time.perf_counter() - wall_started) * 1000.0

        if batch_index >= warmup:
            prev = step_start
            for label, event in _EVENTS:
                forward_sums.setdefault(label, []).append(prev.elapsed_time(event))
                prev = event
            backward_ms.append(bwd_start.elapsed_time(bwd_end))
            step_ms.append(wall_elapsed)

    patching.HierarchicalLatentLM.token_hidden = _ORIGINAL_TOKEN_HIDDEN
    patching.HierarchicalLatentLM.loss = _ORIGINAL_LOSS

    result = {
        "policy": policy_name,
        "mean_step_ms": statistics.fmean(step_ms),
        "mean_backward_ms": statistics.fmean(backward_ms),
        "mean_global_positions": statistics.fmean(global_position_counts),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1 << 20),
        "components_forward_mean_ms": {k: statistics.fmean(v) for k, v in forward_sums.items()},
    }
    del model, optimizer
    gc.collect(); torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-json", default="benchmark_results/phase10/patch_component_profile.json")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--d-model", type=int, default=160)
    parser.add_argument("--d-lexical", type=int, default=24)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=5)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--cutoffs", default="16384,131072,524288")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    manifest_path = Path(args.manifest).expanduser().resolve()
    cutoffs = tuple(int(v) for v in args.cutoffs.split(","))
    config = {
        "seed": 20260815, "seq_len": args.seq_len, "d_model": args.d_model, "d_lexical": args.d_lexical,
        "layers": args.layers, "heads": args.heads, "precision": args.precision, "cutoffs": cutoffs,
        "learning_rate": 3e-4,
    }

    dataset = bench.AGLMShardedDataset(manifest=str(manifest_path), split="train", seq_len=args.seq_len, seed=20260815)
    try:
        real_ids_batches = []
        n_batches = args.warmup + args.steps
        for batch_index in range(min(n_batches, 20)):
            inputs, _, _ = dataset.get_batch(batch_size=args.batch_size, start_index=batch_index * args.batch_size)
            real_ids_batches.append(inputs)
    finally:
        dataset.close()

    results = {}
    for policy_name, fixed_size in (("fixed-1-control", 1), ("fixed-2", 2)):
        print(f"\n=== {policy_name} (fixed_size={fixed_size}) ===", flush=True)
        results[policy_name] = run_policy(policy_name, fixed_size, config, real_ids_batches, device, args.warmup, args.steps)
        r = results[policy_name]
        print(f"mean_step_ms={r['mean_step_ms']:.3f}  mean_global_positions={r['mean_global_positions']:.1f}  "
              f"peak_vram={r['peak_allocated_mib']:.1f}MiB")
        for label, ms in r["components_forward_mean_ms"].items():
            print(f"  fwd.{label:16s} {ms:.3f} ms")
        print(f"  bwd.total{'':10s} {r['mean_backward_ms']:.3f} ms")

    speedup = results["fixed-1-control"]["mean_step_ms"] / results["fixed-2"]["mean_step_ms"]
    print(f"\nfixed-2 / fixed-1 step-time ratio: {results['fixed-2']['mean_step_ms']/results['fixed-1-control']['mean_step_ms']:.4f}x "
          f"(>1.0 means fixed-2 was SLOWER)")

    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"\nWrote {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
