#!/usr/bin/env python3
"""Quality-gate step 1-3: prove the output-side frequency permutation is an
exact bijection over the full frozen 1,551,017-ID vocabulary, verify special
tokens survive it correctly, and verify real multilingual/code/number/URL/UUID
text survives an encode -> permute -> inverse-permute -> decode round trip
bit-for-bit. No GPU, no model, no training involved.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

VOCAB_SIZE = 1_551_017
ROOT = Path(__file__).parent
PERMUTATION_PATH = ROOT / "benchmark_results" / "phase3" / "freq_rank_permutation.npy"
MANIFEST_PATH = Path("/run/media/akash/18FAA791FAA76A28/aglm_project/aglm_tokenized_dataset/metadata/dataset_manifest.json")
FREQ_PATH = Path("/run/media/akash/18FAA791FAA76A28/aglm_project/aglm_tokenized_dataset/metadata/token_frequency.npy")
TOKENIZER_PATH = ROOT / "exported_tokenizers" / "aglm_universal_max"

report = {"checks": [], "roundtrip_tests": [], "special_tokens": []}


def check(name: str, passed: bool, detail: str = "") -> None:
    report["checks"].append({"name": name, "passed": bool(passed), "detail": detail})
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}  {detail}")
    if not passed:
        raise AssertionError(f"CORRECTNESS GATE FAILED: {name}  {detail}")


def main() -> int:
    print("=== Step 1: bijection proof over all 1,551,017 IDs ===")
    permutation = np.load(PERMUTATION_PATH)  # permutation[token_id] = frequency_rank
    check("permutation shape", permutation.shape == (VOCAB_SIZE,), f"shape={permutation.shape}")
    check("permutation dtype is integer", np.issubdtype(permutation.dtype, np.integer), f"dtype={permutation.dtype}")

    inverse = np.argsort(permutation)  # inverse[rank] = token_id
    check("inverse shape", inverse.shape == (VOCAB_SIZE,), f"shape={inverse.shape}")

    # No collisions, no missing IDs, full range represented: a permutation of range(V)
    # is exactly characterized by "sorted(permutation) == arange(V)".
    sorted_perm = np.sort(permutation)
    check("no collisions / no missing IDs / full range covered",
          np.array_equal(sorted_perm, np.arange(VOCAB_SIZE)),
          "sorted(permutation) == arange(V)")

    ids = np.arange(VOCAB_SIZE, dtype=np.int64)
    check("inverse_permutation[permutation[id]] == id for ALL ids",
          np.array_equal(inverse[permutation], ids),
          "checked vectorized over all 1,551,017 IDs")
    check("permutation[inverse_permutation[rank]] == rank for ALL ranks",
          np.array_equal(permutation[inverse], ids),
          "checked vectorized over all 1,551,017 ranks (other direction)")

    unique_count = np.unique(permutation).shape[0]
    check("exactly V unique output ranks", unique_count == VOCAB_SIZE, f"unique={unique_count}")

    print("\n=== Step 2 (documentation, not a runtime check): computation graph ===")
    print("""
  input_ids  ---------------------------------------------------> lexical embedding lookup -> backbone -> hidden
  (real AGLM IDs, UNCHANGED)

  target_ids -> permutation[target_ids] -> remapped_targets -----> AdaptiveLogSoftmaxWithLoss(hidden, remapped_targets) -> loss
  (real AGLM IDs)   (frequency-rank space, ONLY here)

  Only the tensor passed as `target` to the adaptive-softmax loss is remapped. `input_ids` never touches the
  permutation table anywhere in the forward pass: the embedding lookup (`self.lexical(input_ids)`), the
  projection, and every backbone block operate on real, un-permuted AGLM IDs exactly as in the unmodified
  production model. This is enforced by code structure, not by convention: see
  profile_aglm_freq_remap_full_model.py's forward_fn -- `hidden = model.hidden(inputs)...; remapped_targets =
  permutation[targets.reshape(-1)]; model.output(hidden, remapped_targets)` -- inputs never appear on the
  right-hand side of any permutation lookup.
""")

    print("=== Step 8: special-token audit ===")
    manifest = json.loads(MANIFEST_PATH.read_text())
    special_ids = manifest["tokenizer"]["special_token_ids"]
    freq = np.load(FREQ_PATH)
    seen_ranks = set()
    for name, tid in special_ids.items():
        rank = int(permutation[tid])
        count = int(freq[tid])
        cutoffs = [0, 16384, 131072, 524288, VOCAB_SIZE]
        cluster = next(i for i in range(4) if cutoffs[i] <= rank < cutoffs[i + 1])
        cluster_name = ["head", "tail0", "tail1", "tail2"][cluster]
        collision = rank in seen_ranks
        seen_ranks.add(rank)
        entry = {"name": name, "token_id": tid, "corpus_count": count, "permuted_rank": rank,
                  "cluster": cluster_name, "collides_with_another_special_token": collision}
        report["special_tokens"].append(entry)
        print(f"  {name:20s} id={tid:>7}  corpus_count={count:>12,}  permuted_rank={rank:>9,}  "
              f"cluster={cluster_name:6s}  collision={collision}")
        check(f"special token {name} has a unique rank (no collision)", not collision)
    check("all 9 special tokens produced distinct ranks", len(seen_ranks) == len(special_ids))

    print("\n=== Step 3: real tokenizer encode -> permute -> inverse -> decode round trip ===")
    sys.path.insert(0, str(ROOT))
    from aglm_tokenizer.core.tokenizer import AGLMUniversalTokenizer
    print(f"Loading tokenizer from {TOKENIZER_PATH} (may take ~15s)...")
    started = time.time()
    tok = AGLMUniversalTokenizer.load(str(TOKENIZER_PATH))
    print(f"Loaded in {time.time()-started:.1f}s, vocab_size={tok.vocab_size}")

    probes = {
        "english": "The quick brown fox jumps over the lazy dog. Machine learning models require careful evaluation.",
        "hindi_devanagari": "मशीन लर्निंग एक दिलचस्प क्षेत्र है जो तेजी से विकसित हो रहा है।",
        "romanized_hindi": "yeh ek accha din hai aur mujhe bahut khushi ho rahi hai",
        "tamil": "இயந்திர கற்றல் என்பது ஒரு சுவாரஸ்யமான துறையாகும்",
        "chinese": "机器学习是人工智能的一个重要分支，近年来发展迅速。",
        "japanese": "機械学習は人工知能の重要な分野であり、急速に発展しています。",
        "arabic": "التعلم الآلي هو مجال مثير للاهتمام يتطور بسرعة كبيرة جدا",
        "code_python": "def forward(self, x: torch.Tensor) -> torch.Tensor:\n    return self.linear(x) + residual",
        "code_json": '{"id": 48213, "active": true, "tags": ["a", "b", null], "ratio": 3.14159}',
        "math": "Let f(x) = 3x^2 + 2x - 7. Then f'(x) = 6x + 2 and the integral is x^3 + x^2 - 7x + C.",
        "numbers": "The population grew from 1,234,567 to 8,901,234 between 1995 and 2023, a 621.4% increase.",
        "uuid": "Request ID: 550e8400-e29b-41d4-a716-446655440000 correlates with session f47ac10b-58cc-4372-a567-0e02b2c3d479",
        "url": "See https://example.com/path/to/resource?query=value&other=123#fragment for details, or ftp://mirror.example.org/pub/file.tar.gz",
        "mixed_emoji_unicode": "Status: ✅ done 🎉 — café, naïve, Zürich, Москва, 東京都, ½ ± √2 ≈ 1.41421",
        "repeated_structure": "AAAA BBBB AAAA BBBB AAAA BBBB CCCC DDDD CCCC DDDD",
    }

    all_ok = True
    for name, text in probes.items():
        real_ids = tok.encode(text)
        ids_arr = np.array(real_ids, dtype=np.int64)
        out_of_range = (ids_arr < 0) | (ids_arr >= VOCAB_SIZE)
        if out_of_range.any():
            check(f"roundtrip[{name}]: all encoded IDs within frozen vocab range", False,
                  f"{int(out_of_range.sum())} IDs out of range")
        remapped = permutation[ids_arr]
        recovered = inverse[remapped]
        exact_id_match = np.array_equal(recovered, ids_arr)
        decoded_original = tok.decode(real_ids)
        decoded_recovered = tok.decode(recovered.tolist())
        exact_text_match = decoded_original == decoded_recovered
        source_roundtrip_match = decoded_original == text
        ok = exact_id_match and exact_text_match
        all_ok = all_ok and ok
        report["roundtrip_tests"].append({
            "name": name, "n_tokens": len(real_ids), "exact_id_match": bool(exact_id_match),
            "exact_text_match_after_permute_inverse": bool(exact_text_match),
            "tokenizer_source_roundtrip_match": bool(source_roundtrip_match),
        })
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name:22s} tokens={len(real_ids):>4}  id_match={exact_id_match}  "
              f"text_match_after_permute={exact_text_match}  tokenizer_own_roundtrip={source_roundtrip_match}")
        check(f"roundtrip[{name}]: encode->permute->inverse->decode == encode->decode", ok)

    print(f"\nAll {len(probes)} probe strings survived encode->permute->inverse->decode exactly: {all_ok}")

    output_path = ROOT / "benchmark_results" / "phase13" / "permutation_correctness.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
