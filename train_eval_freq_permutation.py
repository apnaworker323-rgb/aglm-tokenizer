#!/usr/bin/env python3
"""Quality-gate steps 4-9: controlled A/B training comparison between the
current production output head (A, real AGLM IDs) and the frequency-permuted
output head (B, exact bijection applied only at the loss). Same architecture,
same initialization seed, same deterministic batch sequence (both variants
call dataset.get_batch(start_index=step*batch_size) - a pure function of step
index, so A and B see bit-identical input batches in the same order).

This is a BOUNDED controlled experiment (thousands of steps, a small slice of
the 1.47B-token corpus), not full-corpus training. No production checkpoint is
written; state lives in-process for the run's own post-training evaluation.

Run once per variant/experiment (separate process = clean GPU memory):
  --variant baseline  --steps N                     Experiment I, variant A
  --variant remapped  --steps N                      Experiment I, variant B
  --variant remapped  --time-budget-seconds T        Experiment II, variant B
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

import benchmark_aglm_gpu_training as bench

VOCAB_SIZE = bench.EXPECTED_VOCAB_SIZE
ROOT = Path(__file__).parent
PERMUTATION_PATH = ROOT / "benchmark_results" / "phase3" / "freq_rank_permutation.npy"
MANIFEST_PATH = Path("/run/media/akash/18FAA791FAA76A28/aglm_project/aglm_tokenized_dataset/metadata/dataset_manifest.json")
TOKENIZER_PATH = ROOT / "exported_tokenizers" / "aglm_universal_max"
EOS_ID = 258
LN2 = math.log(2.0)


def nats_to_bpb(mean_nll_nats: float, raw_bytes_per_token: float) -> float:
    return mean_nll_nats / (LN2 * raw_bytes_per_token)


def build_val_batches(dataset: bench.AGLMShardedDataset, n_batches: int, batch_size: int, seed_offset: int = 999_000):
    batches = []
    for i in range(n_batches):
        inputs, targets, _ = dataset.get_batch(batch_size=batch_size, start_index=(seed_offset + i) * batch_size)
        batches.append((inputs, targets))
    return batches


class Variant:
    def __init__(self, name: str, remap: bool, permutation: Optional[torch.Tensor], inverse: Optional[torch.Tensor]):
        self.name = name
        self.remap = remap
        self.permutation = permutation
        self.inverse = inverse

    def targets_for_loss(self, real_targets_flat: torch.Tensor) -> torch.Tensor:
        if not self.remap:
            return real_targets_flat
        return self.permutation[real_targets_flat]

    def predicted_real_ids(self, predicted_classes: torch.Tensor) -> torch.Tensor:
        if not self.remap:
            return predicted_classes
        return self.inverse[predicted_classes]


def per_token_nll(model: bench.AGLMBenchmarkLM, variant: Variant, input_ids: torch.Tensor,
                   real_targets: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Real-valued per-token NLL (nats), one value per (batch*seq) position, no grad."""
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        hidden = model.hidden(input_ids).reshape(-1, model.final_norm.weight.numel())
        loss_targets = variant.targets_for_loss(real_targets.reshape(-1))
        result = model.output(hidden.float(), loss_targets)
    return (-result.output).float()


def evaluate_val_bpb(model, variant: Variant, val_batches, device, dtype, raw_bytes_per_token) -> Dict[str, float]:
    all_nll = []
    for inputs, targets in val_batches:
        inputs = inputs.to(device)
        targets = targets.to(device)
        nll = per_token_nll(model, variant, inputs, targets, dtype)
        all_nll.append(nll.cpu())
    flat = torch.cat(all_nll)
    mean_nll = float(flat.mean())
    return {
        "val_mean_nll_nats": mean_nll,
        "val_bpb": nats_to_bpb(mean_nll, raw_bytes_per_token),
        "val_tokens": int(flat.numel()),
    }


def frequency_band_bpb(model, variant: Variant, val_batches, device, dtype, raw_bytes_per_token,
                        permutation_np: np.ndarray) -> List[Dict[str, Any]]:
    bands = [(0, 1_000, "top_1K"), (1_000, 10_000, "1K_10K"), (10_000, 100_000, "10K_100K"),
             (100_000, 500_000, "100K_500K"), (500_000, VOCAB_SIZE, "500K_1.55M")]
    nll_by_band: Dict[str, List[float]] = {name: [] for _, _, name in bands}
    for inputs, targets in val_batches:
        inputs_d = inputs.to(device)
        targets_d = targets.to(device)
        nll = per_token_nll(model, variant, inputs_d, targets_d, dtype).cpu().numpy()
        real_targets_flat = targets.reshape(-1).numpy()
        true_ranks = permutation_np[real_targets_flat]
        for lo, hi, name in bands:
            mask = (true_ranks >= lo) & (true_ranks < hi)
            if mask.any():
                nll_by_band[name].extend(nll[mask].tolist())
    result = []
    for lo, hi, name in bands:
        values = nll_by_band[name]
        if values:
            mean_nll = statistics.fmean(values)
            result.append({"band": name, "rank_range": [lo, hi], "n_tokens": len(values),
                            "mean_nll_nats": mean_nll, "bpb": nats_to_bpb(mean_nll, raw_bytes_per_token)})
        else:
            result.append({"band": name, "rank_range": [lo, hi], "n_tokens": 0, "mean_nll_nats": None, "bpb": None})
    return result


def quality_battery(model, variant: Variant, tok, device, dtype, raw_bytes_per_token, seq_len: int) -> Dict[str, Any]:
    probes = {
        "english": "The committee reviewed the quarterly budget report before approving the new infrastructure spending plan for next year.",
        "hindi_devanagari": "भारत की राजधानी नई दिल्ली है और यह देश का सबसे महत्वपूर्ण राजनीतिक केंद्र है। यहाँ की जनसंख्या करोड़ों में है।",
        "romanized_hindi": "aaj mausam bahut accha hai aur hum log park mein ghumne ja rahe hain shaam ko",
        "chinese": "北京是中华人民共和国的首都，也是全国的政治、文化和国际交流中心，拥有悠久的历史。",
        "arabic": "القاهرة هي عاصمة جمهورية مصر العربية وأكبر مدنها من حيث عدد السكان والمساحة الجغرافية",
        "code": "class LinearLayer(nn.Module):\n    def __init__(self, in_features, out_features):\n        super().__init__()\n        self.weight = nn.Parameter(torch.randn(out_features, in_features))",
        "math": "The derivative of sin(x)cos(x) with respect to x is cos(2x), obtained via the product rule and double angle identity.",
        "numbers": "Revenue increased from $4,281,930 to $7,652,104 over three fiscal quarters, representing a growth rate of 78.7 percent annually.",
        "uuids": "The transaction reference 3fa85f64-5717-4562-b3fc-2c963f66afa6 was logged alongside session token 9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d.",
        "urls": "Documentation is available at https://docs.example.org/api/v2/reference and the changelog at https://github.com/example/repo/blob/main/CHANGELOG.md",
        "rare_tokens": "The xerothermic mesoclimate favored sclerophyllous vegetation, dominated by pyrophytic shrubs adapted to recurrent anthropogenic fire regimes.",
    }
    results = {}
    for name, text in probes.items():
        ids = tok.encode(text)[:seq_len - 1]
        if len(ids) < 4:
            continue
        input_ids = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
        target_ids = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)
        nll = per_token_nll(model, variant, input_ids, target_ids, dtype)
        mean_nll = float(nll.mean())
        results[name] = {"n_tokens": len(ids) - 1, "mean_nll_nats": mean_nll,
                          "bpb": nats_to_bpb(mean_nll, raw_bytes_per_token)}

    # exact copy: does the SECOND occurrence of a duplicated span get a lower loss than the first?
    base_ids = tok.encode("The satellite transmitted telemetry data through the encrypted uplink channel before losing contact.")
    copy_ids = (base_ids + base_ids)[:seq_len]
    input_ids = torch.tensor(copy_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    target_ids = torch.tensor(copy_ids[1:], dtype=torch.long, device=device).unsqueeze(0)
    nll = per_token_nll(model, variant, input_ids, target_ids, dtype).cpu().numpy()
    n = len(base_ids)
    first_occurrence = float(nll[: n - 1].mean())
    second_start = n - 1
    second_occurrence = float(nll[second_start: second_start + n].mean()) if second_start < len(nll) else None
    results["exact_copy_induction"] = {
        "first_occurrence_mean_nll_nats": first_occurrence,
        "second_occurrence_mean_nll_nats": second_occurrence,
        "improvement_ratio": (first_occurrence / second_occurrence) if second_occurrence else None,
        "note": "ratio > 1 means the model found the SECOND (repeated) occurrence easier to predict than the first -- weak evidence of induction/copying capability",
    }

    # passkey retrieval: "the code is D1 D2 D3 D4 D5 . <filler> . the code is" -> measure NLL at the 5 digit positions
    filler = tok.encode(" The weather forecast predicted scattered clouds with a gentle breeze from the northwest throughout the afternoon and into the evening hours.")
    digits_text = "The secret access code is 4 8 2 1 3 ."
    digit_ids = tok.encode(digits_text)
    prompt_ids = digit_ids + filler * 3 + tok.encode(" The secret access code is 4 8 2 1 3")
    prompt_ids = prompt_ids[:seq_len]
    if len(prompt_ids) > 10:
        input_ids = torch.tensor(prompt_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
        target_ids = torch.tensor(prompt_ids[1:], dtype=torch.long, device=device).unsqueeze(0)
        nll = per_token_nll(model, variant, input_ids, target_ids, dtype).cpu().numpy()
        tail_n = min(6, len(nll))
        passkey_region_nll = float(nll[-tail_n:].mean())
        results["passkey_retrieval"] = {
            "prompt_tokens": len(prompt_ids),
            "mean_nll_nats_at_repeated_passkey_digits": passkey_region_nll,
            "bpb": nats_to_bpb(passkey_region_nll, raw_bytes_per_token),
            "note": "NLL at the digit positions in the SECOND (query) occurrence of the passkey, after filler text; lower is better retrieval signal",
        }
    return results


def eos_generation_stopping_check(model, variant: Variant, tok, device, dtype) -> Dict[str, Any]:
    prompt = "The report concluded with a summary of key findings and recommendations."
    ids = tok.encode(prompt)
    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        hidden = model.hidden(input_ids).reshape(-1, model.final_norm.weight.numel())
        last_hidden = hidden[-1:].float()
        predicted_class = model.output.predict(last_hidden)
    predicted_class_int = int(predicted_class.item())
    real_id_correct = variant.predicted_real_ids(predicted_class)
    real_id_correct_int = int(real_id_correct.item())
    # Demonstrate the bug this test guards against: comparing the RAW predicted class
    # (permuted-rank space, for variant B) directly against the real EOS id is wrong.
    naive_eos_check = (predicted_class_int == EOS_ID)
    correct_eos_check = (real_id_correct_int == EOS_ID)
    return {
        "prompt": prompt,
        "raw_predicted_class": predicted_class_int,
        "correctly_inverse_mapped_real_id": real_id_correct_int,
        "naive_check_predicted_class_equals_EOS_id_WRONG_for_remapped": naive_eos_check,
        "correct_check_inverse_mapped_id_equals_EOS_id": correct_eos_check,
        "in_range": 0 <= real_id_correct_int < VOCAB_SIZE,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device("cuda:0")
    manifest, raw_bytes_per_token = bench._load_and_validate_manifest(MANIFEST_PATH, "train")
    permutation_np = np.load(PERMUTATION_PATH)
    inverse_np = np.argsort(permutation_np)
    permutation_t = torch.from_numpy(permutation_np).to(device)
    inverse_t = torch.from_numpy(inverse_np).to(device)

    remap = args.variant == "remapped"
    variant = Variant(args.variant, remap, permutation_t if remap else None, inverse_t if remap else None)

    config = bench.BenchmarkConfig(
        manifest=str(MANIFEST_PATH), split="train", seq_len=args.seq_len, batch_size=args.batch_size,
        warmup_steps=0, steps=args.steps, seed=args.seed, d_model=192, d_lexical=32, n_layers=4, n_heads=6,
        ffn_multiple=8 / 3, precision="bf16", learning_rate=args.learning_rate, weight_decay=0.1, clip_grad=1.0,
        cutoffs=(16384, 131072, 524288),
    )
    dtype = bench._autocast_dtype(config.precision)

    train_dataset = bench.AGLMShardedDataset(manifest=str(MANIFEST_PATH), split="train", seq_len=args.seq_len, seed=args.seed)
    val_dataset = bench.AGLMShardedDataset(manifest=str(MANIFEST_PATH), split="val", seq_len=args.seq_len, seed=args.seed + 1)
    val_batches = build_val_batches(val_dataset, args.val_batches, args.batch_size)

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    model = bench.AGLMBenchmarkLM(config).to(device)
    model.train()
    optimizer, fused = bench._make_optimizer(model, config)

    history: List[Dict[str, Any]] = []
    nan_or_inf_steps: List[int] = []
    torch.cuda.reset_peak_memory_stats(device)

    step = 0
    train_started = time.time()
    time_budget = args.time_budget_seconds
    max_steps = args.steps if time_budget is None else 10 ** 9

    while step < max_steps:
        if time_budget is not None and (time.time() - train_started) >= time_budget:
            break
        inputs, targets, _ = train_dataset.get_batch(batch_size=args.batch_size, start_index=step * args.batch_size)
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=dtype):
            hidden = model.hidden(inputs).reshape(-1, config.d_model)
            loss_targets = variant.targets_for_loss(targets.reshape(-1))
            loss = model.output(hidden, loss_targets).loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        is_bad = not math.isfinite(loss_value)
        if is_bad:
            nan_or_inf_steps.append(step)

        if step > 0 and step % args.empty_cache_every == 0:
            # AdaptiveLogSoftmaxWithLoss's tail-cluster intermediates vary in size
            # batch-to-batch (different corpus tokens land in different clusters),
            # which fragments the CUDA caching allocator over thousands of steps on
            # a tight 6GB budget. Periodic release keeps long runs from OOMing on
            # fragmentation despite peak *allocated* memory being well under budget.
            gc.collect()
            torch.cuda.empty_cache()

        if step % args.log_every == 0 or step == max_steps - 1:
            train_bpb = nats_to_bpb(loss_value, raw_bytes_per_token)
            entry = {"step": step, "wall_s": time.time() - train_started, "train_loss_nats": loss_value,
                      "train_bpb": train_bpb, "grad_norm": float(grad_norm), "nan_or_inf": is_bad}
            if step % args.eval_every == 0 or step == max_steps - 1:
                model.eval()
                val = evaluate_val_bpb(model, variant, val_batches, device, dtype, raw_bytes_per_token)
                model.train()
                entry.update(val)
            history.append(entry)
            print(f"[{args.variant}] step={step:>5} wall={entry['wall_s']:.1f}s train_bpb={train_bpb:.4f} "
                  f"grad_norm={float(grad_norm):.3f}" + (f" val_bpb={entry.get('val_bpb')}" if "val_bpb" in entry else ""),
                  flush=True)
            if step % args.eval_every == 0:
                partial_path = Path(args.output_json).expanduser().resolve().with_suffix(".partial.json")
                partial_path.parent.mkdir(parents=True, exist_ok=True)
                partial_path.write_text(json.dumps({"variant": args.variant, "completed_steps": step,
                                                      "history": history}, indent=2, default=str), encoding="utf-8")
        step += 1

    train_elapsed = time.time() - train_started
    completed_steps = step
    tokens_processed = completed_steps * args.batch_size * args.seq_len
    tok_s = tokens_processed / max(train_elapsed, 1e-9)
    raw_mib_s = tokens_processed * raw_bytes_per_token / (1 << 20) / max(train_elapsed, 1e-9)

    model.eval()
    final_val = evaluate_val_bpb(model, variant, val_batches, device, dtype, raw_bytes_per_token)
    band_bpb = frequency_band_bpb(model, variant, val_batches, device, dtype, raw_bytes_per_token, permutation_np)

    print("Loading tokenizer for quality battery + generation check...", flush=True)
    sys.path.insert(0, str(ROOT))
    from aglm_tokenizer.core.tokenizer import AGLMUniversalTokenizer
    tok = AGLMUniversalTokenizer.load(str(TOKENIZER_PATH))
    battery = quality_battery(model, variant, tok, device, dtype, raw_bytes_per_token, args.seq_len)
    eos_check = eos_generation_stopping_check(model, variant, tok, device, dtype)

    result = {
        "variant": args.variant,
        "experiment": "time_budget" if time_budget is not None else "step_matched",
        "requested_steps": args.steps if time_budget is None else None,
        "requested_time_budget_seconds": time_budget,
        "completed_steps": completed_steps,
        "train_elapsed_seconds": train_elapsed,
        "tokens_processed": tokens_processed,
        "tokens_per_second": tok_s,
        "raw_mib_per_second": raw_mib_s,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1 << 20),
        "nan_or_inf_step_count": len(nan_or_inf_steps),
        "nan_or_inf_steps": nan_or_inf_steps[:50],
        "final_val": final_val,
        "history": history,
        "frequency_band_bpb": band_bpb,
        "quality_battery": battery,
        "eos_generation_stopping_check": eos_check,
        "seed": config.seed,
        "raw_bytes_per_token": raw_bytes_per_token,
    }

    train_dataset.close()
    val_dataset.close()
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("baseline", "remapped"), required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--time-budget-seconds", type=float, default=None)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--val-batches", type=int, default=30)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--empty-cache-every", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    result = run(args)
    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\ncompleted_steps={result['completed_steps']} elapsed={result['train_elapsed_seconds']:.1f}s "
          f"final_val_bpb={result['final_val']['val_bpb']:.4f} tok/s={result['tokens_per_second']:.1f}")
    print(f"Wrote {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
