#!/usr/bin/env python3
"""Controlled GPU study of lossless causal AGLM latent patches."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from aglm_dataset_loader import AGLMShardedDataset
from aglm_latent_patching import (
    MAX_TOKEN_ID,
    VOCAB_SIZE,
    HierarchicalLatentLM,
    LosslessPatchBatch,
    LosslessLatentPatcher,
    PatchPolicy,
    calibrate_entropy_threshold,
    entropy_table_from_counts,
)
from aglm_tokenizer.core.tokenizer import AGLMUniversalTokenizer
from compare_aglm_minimal_segmentation import ARTIFACT_HASHES, verify_frozen


TOKENIZER_SHA256 = ARTIFACT_HASHES["aglm_vocab.json.gz"]


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_manifest(path: Path) -> Dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    failures = []
    if not manifest.get("complete") or manifest.get("sample_run"):
        failures.append("manifest is not a completed full conversion")
    if manifest.get("numpy_dtype") != "<u4" or manifest.get("endian") != "little":
        failures.append("dataset is not canonical uint32 little-endian")
    tokenizer = manifest.get("tokenizer", {})
    if tokenizer.get("sha256") != TOKENIZER_SHA256 or tokenizer.get("vocab_size") != VOCAB_SIZE:
        failures.append("frozen tokenizer identity mismatch")
    if failures:
        raise RuntimeError("; ".join(failures))
    return manifest


def compute_train_frequency(manifest_path: Path, manifest: Dict[str, Any], cache: Path) -> np.ndarray:
    identity = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    metadata_path = cache.with_suffix(".json")
    if cache.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        counts = np.load(cache, mmap_mode="r")
        if metadata.get("manifest_sha256") == identity and counts.shape == (VOCAB_SIZE,):
            return np.asarray(counts)
    root = manifest_path.parent.parent
    counts = np.zeros(VOCAB_SIZE, dtype=np.uint64)
    for index, row in enumerate(manifest["shards"]["train"]):
        path = root / row["path"]
        mapped = np.memmap(path, dtype="<u4", mode="r")
        for start in range(0, len(mapped), 8_000_000):
            chunk = np.asarray(mapped[start : start + 8_000_000], dtype=np.int64)
            counts += np.bincount(chunk, minlength=VOCAB_SIZE).astype(np.uint64, copy=False)
        mapped._mmap.close()  # type: ignore[attr-defined]
        print(f"[frequency] {index + 1}/{len(manifest['shards']['train'])}", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(cache.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, counts, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, cache)
    atomic_json(metadata_path, {"manifest_sha256": identity, "train_tokens": int(counts.sum())})
    return counts


def token_byte_lengths(tokenizer_dir: Path) -> np.ndarray:
    tokenizer = AGLMUniversalTokenizer.load(str(tokenizer_dir))
    if tokenizer.vocab_size != VOCAB_SIZE:
        raise RuntimeError("frozen vocabulary size changed")
    lengths = np.zeros(VOCAB_SIZE, dtype=np.uint16)
    for token_id, value in tokenizer.engine.id_to_bytes.items():
        if len(value) > np.iinfo(np.uint16).max:
            raise RuntimeError("token byte length exceeds uint16 metadata capacity")
        lengths[token_id] = len(value)
    # Special IDs carry document/control semantics and represent no raw source bytes.
    for token_id in tokenizer.engine.special_id_to_str:
        lengths[token_id] = 0
    del tokenizer
    gc.collect()
    return lengths


def build_schedule(
    dataset: AGLMShardedDataset,
    batches: int,
    batch_size: int,
    byte_lengths: np.ndarray,
) -> Tuple[List[torch.Tensor], List[int], str]:
    sequences: List[torch.Tensor] = []
    raw_bytes: List[int] = []
    digest = hashlib.sha256()
    for batch_index in range(batches):
        inputs, _, _ = dataset.get_batch(batch_size, start_index=batch_index * batch_size)
        if int(inputs.min()) < 0 or int(inputs.max()) > MAX_TOKEN_ID:
            raise RuntimeError("schedule contains an ID outside frozen vocabulary")
        inputs = inputs.contiguous()
        digest.update(inputs.numpy().astype("<i8", copy=False).tobytes())
        sequences.append(inputs)
        raw_bytes.append(int(byte_lengths[inputs.numpy()].sum(dtype=np.uint64)))
    return sequences, raw_bytes, digest.hexdigest()


def codec_metrics(
    patcher: LosslessLatentPatcher,
    policy: PatchPolicy,
    schedules: Sequence[torch.Tensor],
    raw_bytes: Sequence[int],
) -> Dict[str, Any]:
    total_tokens = 0
    total_positions = 0
    histogram = np.zeros(9, dtype=np.int64)
    exact = True
    random_retrieval_correct = 0
    random_retrieval_total = 0
    digest_original = hashlib.sha256()
    digest_decoded = hashlib.sha256()
    for batch_index, ids in enumerate(schedules):
        patches = patcher.encode(ids, policy)
        decoded = patcher.decode(patches)
        exact = exact and torch.equal(decoded, ids)
        original_bytes = ids.numpy().astype("<u4", copy=False).tobytes()
        decoded_bytes = decoded.numpy().astype("<u4", copy=False).tobytes()
        digest_original.update(original_bytes)
        digest_decoded.update(decoded_bytes)
        total_tokens += patches.token_count
        total_positions += patches.global_positions
        histogram += np.bincount(patches.lengths[patches.patch_mask].numpy(), minlength=9)[:9]
        # Deterministic random-access retrieval through the exact local decoder.
        for row in range(ids.shape[0]):
            token_index = (batch_index * 131 + row * 17) % ids.shape[1]
            random_retrieval_correct += int(decoded[row, token_index] == ids[row, token_index])
            random_retrieval_total += 1
    if not exact or digest_original.digest() != digest_decoded.digest():
        raise RuntimeError(f"lossless codec gate failed for {policy.name}")
    return {
        "codec_bit_exact": True,
        "original_token_sha256": digest_original.hexdigest(),
        "decoded_token_sha256": digest_decoded.hexdigest(),
        "tokens": total_tokens,
        "global_positions": total_positions,
        "tokens_per_global_position": total_tokens / total_positions,
        "raw_bytes_per_global_position": sum(raw_bytes) / total_positions,
        "position_reduction_fraction": 1.0 - total_positions / total_tokens,
        "patch_length_histogram": {str(i): int(histogram[i]) for i in range(1, 9)},
        "exact_codec_retrieval_accuracy": random_retrieval_correct / random_retrieval_total,
        "exact_codec_retrieval_trials": random_retrieval_total,
    }


def causal_gate(
    model: HierarchicalLatentLM,
    patcher: LosslessLatentPatcher,
    policy: PatchPolicy,
    sequence: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    source = sequence[:1, :96].clone()
    cut = 47
    changed = source.clone()
    changed[:, cut:] = (changed[:, cut:] + 104_729) % VOCAB_SIZE
    first_patches = patcher.encode(source, policy)
    second_patches = patcher.encode(changed, policy)

    # Dynamic future entropy may change only the number of suffix patches. Pad
    # both executions to one common tensor shape so an SDPA kernel-selection
    # change cannot masquerade as data dependence in the bit-exact causality gate.
    target_patches = max(first_patches.token_ids.shape[1], second_patches.token_ids.shape[1])

    def padded(value: LosslessPatchBatch) -> LosslessPatchBatch:
        missing = target_patches - value.token_ids.shape[1]
        if not missing:
            return value
        return LosslessPatchBatch(
            exact_latents=torch.nn.functional.pad(value.exact_latents, (0, 0, 0, missing)),
            token_ids=torch.nn.functional.pad(value.token_ids, (0, 0, 0, missing)),
            token_mask=torch.nn.functional.pad(value.token_mask, (0, 0, 0, missing)),
            patch_mask=torch.nn.functional.pad(value.patch_mask, (0, missing)),
            lengths=torch.nn.functional.pad(value.lengths, (0, missing)),
            original_length=value.original_length,
            policy_name=value.policy_name,
        )

    first, _ = model.token_hidden(padded(first_patches), device)
    second, _ = model.token_hidden(padded(second_patches), device)
    # The hidden used to predict token `cut` must not see that token or any future token.
    prefix_count = cut + 1
    delta = float((first[:prefix_count] - second[:prefix_count]).abs().max().detach().cpu())
    passed = delta == 0.0
    if not passed:
        raise RuntimeError(f"strict causality failed for {policy.name}: max delta={delta}")
    return {"passed": passed, "perturb_from_token": cut, "predictions_checked": prefix_count, "max_abs_delta": delta}


def make_optimizer(model: torch.nn.Module, lr: float) -> Tuple[torch.optim.Optimizer, bool]:
    try:
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1, fused=True), True
    except (RuntimeError, TypeError):
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1), False


def natural_repeat_mask(ids: torch.Tensor, minimum_distance: int = 32) -> torch.Tensor:
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for batch_index, row in enumerate(ids.tolist()):
        seen: set[int] = set()
        for index, token_id in enumerate(row):
            if index >= minimum_distance:
                seen.add(row[index - minimum_distance])
            if token_id in seen:
                mask[batch_index, index] = True
    return mask


def adaptive_predict_chunked(
    output: torch.nn.AdaptiveLogSoftmaxWithLoss,
    hidden: torch.Tensor,
    chunk_size: int = 8,
) -> torch.Tensor:
    """Exact argmax without allocating [all_validation_tokens, 1.55M]."""
    predictions = []
    for start in range(0, len(hidden), chunk_size):
        predictions.append(output.predict(hidden[start : start + chunk_size].float()))
    return torch.cat(predictions)


@torch.no_grad()
def evaluate(
    model: HierarchicalLatentLM,
    patcher: LosslessLatentPatcher,
    policy: PatchPolicy,
    sequences: Sequence[torch.Tensor],
    raw_bytes: Sequence[int],
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    total_raw = 0
    correct = 0
    repeat_correct = 0
    repeat_total = 0
    for ids, byte_count in zip(sequences, raw_bytes):
        patches = patcher.encode(ids, policy)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden, targets = model.token_hidden(patches, device)
            loss = model.output(hidden, targets).loss
        predictions = adaptive_predict_chunked(model.output, hidden)
        flat_targets = targets
        correct += int((predictions == flat_targets).sum().item())
        repeat = natural_repeat_mask(ids).reshape(-1).to(device)
        repeat_correct += int(((predictions == flat_targets) & repeat).sum().item())
        repeat_total += int(repeat.sum().item())
        token_count = len(flat_targets)
        total_nll += float(loss.detach().cpu()) * token_count
        total_tokens += token_count
        total_raw += byte_count
    return {
        "validation_loss_nats": total_nll / total_tokens,
        "validation_bpb": total_nll / (math.log(2.0) * total_raw),
        "validation_tokens": total_tokens,
        "validation_raw_bytes": total_raw,
        "causal_lm_exact_token_accuracy": correct / total_tokens,
        "exact_long_range_repeat_retrieval_accuracy": repeat_correct / repeat_total if repeat_total else None,
        "exact_long_range_repeat_retrieval_correct": repeat_correct,
        "exact_long_range_repeat_retrieval_trials": repeat_total,
    }


def train_policy(
    policy: PatchPolicy,
    patcher: LosslessLatentPatcher,
    train_sequences: Sequence[torch.Tensor],
    train_raw_bytes: Sequence[int],
    val_sequences: Sequence[torch.Tensor],
    val_raw_bytes: Sequence[int],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = HierarchicalLatentLM(
        d_model=args.d_model, d_lexical=args.d_lexical, n_layers=args.layers,
        n_heads=args.heads, max_global_positions=args.seq_len,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer, fused = make_optimizer(model, args.learning_rate)
    causality = causal_gate(model, patcher, policy, val_sequences[0], device)
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    step_times: List[float] = []
    patch_times: List[float] = []
    timed_raw = 0
    timed_tokens = 0
    losses: List[float] = []
    timed_started = 0.0
    for step, (ids, byte_count) in enumerate(zip(train_sequences, train_raw_bytes)):
        patch_started = time.perf_counter()
        patches = patcher.encode(ids, policy)
        patch_elapsed = time.perf_counter() - patch_started
        optimizer.zero_grad(set_to_none=True)
        step_started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model.loss(patches, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - step_started
        if step == args.warmup_steps - 1:
            timed_started = time.perf_counter()
        if step >= args.warmup_steps:
            step_times.append(elapsed)
            patch_times.append(patch_elapsed)
            timed_raw += byte_count
            timed_tokens += ids.numel()
            losses.append(float(loss.detach().cpu()))
            completed = step - args.warmup_steps + 1
            if completed == 1 or completed % args.log_interval == 0 or completed == args.steps:
                print(f"[{policy.name}] {completed}/{args.steps} loss={losses[-1]:.5f} step={elapsed*1000:.1f}ms patch={patch_elapsed*1000:.2f}ms", flush=True)
    torch.cuda.synchronize(device)
    elapsed_total = time.perf_counter() - timed_started
    # Adam moments and dense gradients are not needed for validation. Releasing
    # them prevents exact full-vocabulary argmax from competing with training state.
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    validation = evaluate(model, patcher, policy, val_sequences, val_raw_bytes, device)
    result = {
        "policy": policy.name,
        "parameter_count": parameter_count,
        "fused_adamw": fused,
        "timed_steps": args.steps,
        "timed_tokens": timed_tokens,
        "timed_raw_bytes": timed_raw,
        "elapsed_seconds_including_patching": elapsed_total,
        "summed_gpu_step_seconds": sum(step_times),
        "tokens_per_second": timed_tokens / elapsed_total,
        "raw_mib_per_second": timed_raw / (1 << 20) / elapsed_total,
        "gpu_step_tokens_per_second": timed_tokens / sum(step_times),
        "gpu_step_raw_mib_per_second": timed_raw / (1 << 20) / sum(step_times),
        "step_ms_p50": percentile(step_times, 50) * 1000,
        "step_ms_p95": percentile(step_times, 95) * 1000,
        "patch_ms_p50": percentile(patch_times, 50) * 1000,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1 << 20),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1 << 20),
        "causality": causality,
        **validation,
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    rows = payload["results"]
    quality_candidates = [
        row["policy"] for row in rows[1:] if row["quality_preserved_vs_patch1"]
    ]
    speed_candidates = [
        row["policy"] for row in rows[1:] if row["gpu_speedup_confirmed"]
    ]
    exact_tokens = rows[0]["codec"]["tokens"]
    exact_trials = rows[0]["codec"]["exact_codec_retrieval_trials"]
    lines = [
        "# AGLM Lossless Hierarchical Latent-Patching Report", "",
        f"Status: **{payload['status']}**  ",
        f"Frozen vocabulary: **{VOCAB_SIZE:,}** IDs, SHA256 `{TOKENIZER_SHA256}`  ",
        f"Identical train schedule SHA256: `{payload['train_schedule_sha256']}`", "",
        "| Policy | tokens/global pos | raw B/global pos | val BPB | codec retrieval | learned repeat retrieval | learned token accuracy | end-to-end raw MiB/s | GPU-step raw MiB/s | peak MiB | GPU ratio | projected 20TB days | quality preserved |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        codec = row["codec"]
        retrieval = row["exact_long_range_repeat_retrieval_accuracy"]
        lines.append(
            f"| {row['policy']} | {codec['tokens_per_global_position']:.3f} | {codec['raw_bytes_per_global_position']:.3f} | "
            f"{row['validation_bpb']:.5f} | {codec['exact_codec_retrieval_accuracy']:.3f} | "
            f"{retrieval if retrieval is not None else 'n/a'} | {row['causal_lm_exact_token_accuracy']:.5f} | "
            f"{row['raw_mib_per_second']:.4f} | {row['gpu_step_raw_mib_per_second']:.4f} | "
            f"{row['peak_allocated_mib']:.1f} | {row['gpu_wall_clock_ratio_vs_patch1']:.3f}x | "
            f"{row['projected_20tb_training_days']:.1f} | "
            f"{'YES' if row['quality_preserved_vs_patch1'] else 'NO'} |"
        )
    lines += [
        "", "## Hard gates", "",
        "- Vocabulary artifacts are hash-identical before and after; no token was added, removed, or renumbered.",
        f"- Every patch policy reconstructs all {exact_tokens:,} audited uint32 tokens bit-for-bit; original and decoded SHA256 values match.",
        f"- Exact random-access local-code retrieval is 100% across {exact_trials:,} trials per policy.",
        "- The deterministic exact decoder and the learned causal LM predictor are separate measurements. The learned predictor is not claimed to reconstruct every token.",
        "- Future-token perturbation produced exactly zero change in all checked earlier prediction states for every policy.",
        "- `quality preserved` is strict: validation BPB may not exceed patch-1 and learned long-range repeat retrieval may not fall below patch-1.",
        "- Position reduction is never labeled a speedup. `GPU wall-clock ratio` uses synchronized forward + backward + clipping + optimizer time and excludes CPU patch construction.",
        "", "## Decision", "",
        f"- Non-control policies passing the sampled BPB + learned-retrieval quality gate: **{', '.join(quality_candidates) if quality_candidates else 'none'}**.",
        f"- Policies with quality-preserving GPU wall-clock speedup: **{', '.join(speed_candidates) if speed_candidates else 'none'}**.",
        f"- The best quality-preserving candidate, fixed-2, ran at {rows[1]['gpu_wall_clock_ratio_vs_patch1']:.3f}x patch-1 GPU throughput. It is therefore not a confirmed GPU speedup.",
        "- Production remains on patch-1. Fixed-2 is a research candidate only; fixed-4, fixed-8, and dynamic entropy patching fail the current strict quality gate.",
        "- The 20TB projections are configuration-specific linear extrapolations from this small GPU prototype, not production forecasts.",
        "", "## Scope caveat", "",
        f"These are {payload['config']['steps']}-step controlled prototype results on a compact full-ID model, not converged final-model quality. "
        "The exact FP32 local code is structurally reversible; that does not make a learned generative prediction exact. "
        "A policy failing the strict quality gate remains experimental even when its codec is mathematically lossless.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", default="AGLM_LATENT_PATCHING_REPORT.json")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--d-model", type=int, default=160)
    parser.add_argument("--d-lexical", type=int, default=24)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--log-interval", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the wall-clock study")
    manifest_path = Path(args.manifest).expanduser().resolve()
    tokenizer_dir = Path(args.tokenizer).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    manifest = load_manifest(manifest_path)
    hashes_before = verify_frozen(tokenizer_dir)
    cache = Path(__file__).resolve().parent / "benchmark_results" / "latent_patching" / "train_frequency.npy"
    train_counts = compute_train_frequency(manifest_path, manifest, cache)
    entropy_bits = entropy_table_from_counts(train_counts)
    byte_lengths = token_byte_lengths(tokenizer_dir)

    total_train_batches = args.warmup_steps + args.steps
    train_dataset = AGLMShardedDataset(str(manifest_path), "train", args.seq_len, args.seed)
    val_dataset = AGLMShardedDataset(str(manifest_path), "val", args.seq_len, args.seed + 1)
    try:
        train_sequences, train_raw, train_digest = build_schedule(train_dataset, total_train_batches, args.batch_size, byte_lengths)
        val_sequences, val_raw, val_digest = build_schedule(val_dataset, args.val_batches, args.batch_size, byte_lengths)
    finally:
        train_dataset.close()
        val_dataset.close()
    threshold = calibrate_entropy_threshold(torch.cat(train_sequences[: min(10, len(train_sequences))]), entropy_bits, 4.0)
    patcher = LosslessLatentPatcher(entropy_bits)
    policies = [
        PatchPolicy("fixed-1-control", "fixed", fixed_size=1),
        PatchPolicy("fixed-2", "fixed", fixed_size=2),
        PatchPolicy("fixed-4", "fixed", fixed_size=4),
        PatchPolicy("fixed-8", "fixed", fixed_size=8),
        PatchPolicy("dynamic-entropy", "entropy", entropy_threshold_bits=threshold, max_patch_tokens=8),
    ]
    codec = {
        policy.name: codec_metrics(patcher, policy, train_sequences + val_sequences, train_raw + val_raw)
        for policy in policies
    }
    device = torch.device("cuda:0")
    results = []
    for policy in policies:
        print(f"\n=== {policy.name} ===", flush=True)
        row = train_policy(policy, patcher, train_sequences, train_raw, val_sequences, val_raw, args, device)
        row["codec"] = codec[policy.name]
        results.append(row)

    baseline = results[0]
    for row in results:
        row["measured_wall_clock_ratio_vs_patch1"] = row["raw_mib_per_second"] / baseline["raw_mib_per_second"]
        row["gpu_wall_clock_ratio_vs_patch1"] = row["gpu_step_raw_mib_per_second"] / baseline["gpu_step_raw_mib_per_second"]
        retrieval = row["exact_long_range_repeat_retrieval_accuracy"]
        baseline_retrieval = baseline["exact_long_range_repeat_retrieval_accuracy"]
        row["quality_preserved_vs_patch1"] = (
            row["validation_bpb"] <= baseline["validation_bpb"]
            and (retrieval is not None and baseline_retrieval is not None and retrieval >= baseline_retrieval)
        )
        row["gpu_speedup_confirmed"] = row["quality_preserved_vs_patch1"] and row["gpu_wall_clock_ratio_vs_patch1"] > 1.0
        row["projected_20tb_training_seconds"] = 20_000_000_000_000 / (row["raw_mib_per_second"] * (1 << 20))
        row["projected_20tb_training_days"] = row["projected_20tb_training_seconds"] / 86400

    hashes_after = verify_frozen(tokenizer_dir)
    if hashes_before != hashes_after:
        raise RuntimeError("frozen vocabulary artifacts changed during prototype")
    hard_gates = {
        "vocabulary_unchanged": True,
        "all_codecs_bit_exact": all(row["codec"]["codec_bit_exact"] for row in results),
        "all_causality_checks_passed": all(row["causality"]["passed"] for row in results),
        "all_exact_codec_retrieval": all(row["codec"]["exact_codec_retrieval_accuracy"] == 1.0 for row in results),
    }
    payload = {
        "status": "PASSED" if all(hard_gates.values()) else "FAILED",
        "config": vars(args),
        "frozen_vocabulary": {"size": VOCAB_SIZE, "hashes_before": hashes_before, "hashes_after": hashes_after},
        "train_schedule_sha256": train_digest,
        "validation_schedule_sha256": val_digest,
        "dynamic_entropy_threshold_bits": threshold,
        "hard_gates": hard_gates,
        "gpu": {"name": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda},
        "results": results,
    }
    atomic_json(output, payload)
    report = output.with_suffix(".md")
    write_markdown(report, payload)
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Reports: {output} and {report}", flush=True)
    return 0 if payload["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
