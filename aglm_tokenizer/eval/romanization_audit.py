"""
Romanization Stress-Test and Spelling-Variant Fragmentation Auditor.
Implements Section 15 of Mandatory Specifications:
- Evaluates 10,000+ romanized / transliterated examples.
- Measures token inflation across spelling variants.
- Calculates exact lossless reconstruction and fragmentation indices.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from tabulate import tabulate
from aglm_tokenizer.corpus.romanization_dataset import RomanizationDatasetGenerator


class RomanizationAuditor:
    """Executes full Romanization stress tests across multiple tokenizers."""

    def __init__(self, target_sample_count: int = 10000):
        self.dataset = RomanizationDatasetGenerator.generate_romanized_stress_dataset(target_sample_count)

    def run_audit(self, tokenizer_name: str, tokenizer_obj: Any) -> Dict[str, Any]:
        """Runs the 10,000 example stress test on the tokenizer."""
        per_lang_stats: Dict[str, Dict[str, Any]] = {}
        total_tokens = 0
        total_bytes = 0
        total_words = 0
        lossless_failures = 0

        # Group by base sentence to measure variant fragmentation
        variant_groups: Dict[str, List[int]] = {}

        for item in self.dataset:
            lang = item["language"]
            text = item["text"]
            base_key = f"{lang}:{item['base_text']}"
            raw_bytes = text.encode("utf-8")
            words = text.split()

            if lang not in per_lang_stats:
                per_lang_stats[lang] = {
                    "count": 0,
                    "bytes": 0,
                    "tokens": 0,
                    "words": 0,
                    "tokens_per_word": [],
                    "bytes_per_token": []
                }

            tokens = tokenizer_obj.encode(text)
            n_toks = len(tokens)
            n_bytes = len(raw_bytes)
            n_words = max(1, len(words))

            # Exact roundtrip check
            reconstructed = tokenizer_obj.decode(tokens)
            if reconstructed != text:
                lossless_failures += 1

            per_lang_stats[lang]["count"] += 1
            per_lang_stats[lang]["bytes"] += n_bytes
            per_lang_stats[lang]["tokens"] += n_toks
            per_lang_stats[lang]["words"] += n_words
            per_lang_stats[lang]["tokens_per_word"].append(n_toks / n_words)
            per_lang_stats[lang]["bytes_per_token"].append(n_bytes / n_toks if n_toks > 0 else 0)

            total_tokens += n_toks
            total_bytes += n_bytes
            total_words += n_words

            variant_groups.setdefault(base_key, []).append(n_toks)

        # Compute variant fragmentation variance (token length variance across spelling variants of the same base sentence)
        fragmentation_variances = []
        for base_key, tok_lens in variant_groups.items():
            if len(tok_lens) > 1:
                fragmentation_variances.append(float(np.var(tok_lens)))

        mean_fragmentation_var = float(np.mean(fragmentation_variances)) if fragmentation_variances else 0.0
        overall_bpt = (total_bytes / total_tokens) if total_tokens > 0 else 0.0
        overall_tpw = (total_tokens / total_words) if total_words > 0 else 0.0

        # Language summary table
        table_rows = []
        for lang, st in sorted(per_lang_stats.items()):
            bpt = st["bytes"] / st["tokens"] if st["tokens"] > 0 else 0.0
            tpw = st["tokens"] / st["words"] if st["words"] > 0 else 0.0
            table_rows.append([
                lang,
                f"{st['count']:,}",
                f"{st['bytes'] / 1024:.1f} KB",
                f"{st['tokens']:,}",
                f"{bpt:.2f}",
                f"{tpw:.2f}"
            ])

        return {
            "tokenizer_name": tokenizer_name,
            "sample_count": len(self.dataset),
            "total_tokens": total_tokens,
            "total_bytes": total_bytes,
            "overall_bytes_per_token": overall_bpt,
            "overall_tokens_per_word": overall_tpw,
            "mean_spelling_fragmentation_variance": mean_fragmentation_var,
            "lossless_failures": lossless_failures,
            "is_100_percent_lossless": (lossless_failures == 0),
            "language_table": tabulate(table_rows, headers=["Language", "Samples", "Raw Size", "Tokens", "Bytes/Token", "Tokens/Word"], tablefmt="github")
        }
