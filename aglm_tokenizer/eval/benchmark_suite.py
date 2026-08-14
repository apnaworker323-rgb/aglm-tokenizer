"""
Multilingual Tokenizer Benchmark Suite.
Implements Section 8 and 14 of Mandatory Specifications:
Produces the exact required Language-by-Language Benchmark Table:
| Language | Script | Raw MB | Tokens | Bytes/Token | Tokens/Word | Byte Fallback % | Encode MB/s |
Along with fairness metrics, macro-averages, worst/best 10, and cross-tokenizer comparison.
"""

from typing import Dict, List, Any, Optional, Tuple
import time
from tabulate import tabulate
from aglm_tokenizer.corpus.language_registry import LANGUAGES, LanguageSpec
from aglm_tokenizer.corpus.multilingual_corpus import MultilingualCorpusManager
from aglm_tokenizer.eval.metrics import LanguageEvalResult, FairnessMetrics, MetricsCalculator
from aglm_tokenizer.eval.public_tokenizers import PublicTokenizerWrapper


class MultilingualBenchmarkSuite:
    """Runs rigorous, standardized multilingual benchmarks across all languages and tokenizers."""

    def __init__(self, languages: Optional[List[str]] = None):
        if languages is None:
            self.languages = list(LANGUAGES.keys())
        else:
            self.languages = languages

    def evaluate_tokenizer(self, tokenizer_name: str, tokenizer_obj: Any) -> Tuple[List[LanguageEvalResult], FairnessMetrics]:
        """
        Evaluates a single tokenizer (AGLM or Public) on the strictly held-out multilingual benchmark set.
        """
        results: List[LanguageEvalResult] = []

        for lang_code in self.languages:
            lang_spec = LANGUAGES.get(lang_code)
            if not lang_spec:
                continue

            eval_text = MultilingualCorpusManager.get_held_out_corpus(lang_code)
            raw_bytes = eval_text.encode("utf-8")
            raw_words = len(eval_text.split())

            t0 = time.perf_counter()
            if hasattr(tokenizer_obj, "encode_with_stats"):
                stats = tokenizer_obj.encode_with_stats(eval_text)
                encode_time = time.perf_counter() - t0
                tokens = stats["tokens"]
                num_tokens = stats["num_tokens"]
                byte_fallbacks = stats["byte_fallback_count"]
                byte_fallback_pct = stats["byte_fallback_ratio"] * 100.0
                is_lossless = stats["is_lossless"]
            else:
                tokens = tokenizer_obj.encode(eval_text)
                encode_time = time.perf_counter() - t0
                num_tokens = len(tokens)
                byte_fallbacks = 0  # public tokenizers internal fallback
                byte_fallback_pct = 0.0
                reconstructed = tokenizer_obj.decode(tokens)
                is_lossless = (reconstructed == eval_text)

            bpt = (len(raw_bytes) / num_tokens) if num_tokens > 0 else 0.0
            tpw = (num_tokens / max(1, raw_words))
            throughput = (len(raw_bytes) / (1024 * 1024)) / encode_time if encode_time > 0 else 0.0

            results.append(LanguageEvalResult(
                language_code=lang_code,
                language_name=lang_spec.name,
                script=lang_spec.script.value,
                family=lang_spec.family.value,
                raw_bytes=len(raw_bytes),
                raw_words=raw_words,
                token_count=num_tokens,
                bytes_per_token=bpt,
                tokens_per_word=tpw,
                byte_fallback_count=byte_fallbacks,
                byte_fallback_percent=byte_fallback_pct,
                encode_time_sec=encode_time,
                encode_throughput_mb_s=throughput,
                is_lossless=is_lossless
            ))

        fairness = MetricsCalculator.compute_fairness_summary(results)
        return results, fairness

    @staticmethod
    def format_markdown_table(results: List[LanguageEvalResult], tokenizer_name: str) -> str:
        """Formats the results as a standard GitHub Markdown table as specified in Section 8."""
        headers = ["Language", "Script", "Raw KB", "Tokens", "Bytes/Token", "Tokens/Word", "Byte Fallback %", "Encode MB/s"]
        rows = []
        for r in results:
            rows.append([
                f"{r.language_name} ({r.language_code})",
                r.script,
                f"{r.raw_bytes / 1024:.2f}",
                f"{r.token_count:,}",
                f"{r.bytes_per_token:.2f}",
                f"{r.tokens_per_word:.2f}",
                f"{r.byte_fallback_percent:.1f}%",
                f"{r.encode_throughput_mb_s:.2f}"
            ])
        return tabulate(rows, headers=headers, tablefmt="github")

    @staticmethod
    def format_fairness_summary(fairness: FairnessMetrics, tokenizer_name: str) -> str:
        """Formats fairness and inequality metrics in markdown."""
        lines = [
            f"### Fairness & Imbalance Summary: `{tokenizer_name}`",
            f"- **Best-Language Bytes/Token**: `{fairness.best_bytes_per_token:.2f}`",
            f"- **Median-Language Bytes/Token**: `{fairness.median_bytes_per_token:.2f}`",
            f"- **Worst-Language Bytes/Token**: `{fairness.worst_bytes_per_token:.2f}`",
            f"- **P10 / P50 / P90**: `{fairness.p10:.2f} / {fairness.p50:.2f} / {fairness.p90:.2f}`",
            f"- **Macro Mean ± Std**: `{fairness.macro_mean:.2f} ± {fairness.macro_std:.2f}`",
            f"- **Coefficient of Variation (CV)**: `{fairness.coefficient_of_variation:.3f}`",
            f"- **Gini Inequality Index**: `{fairness.gini_index:.3f}`",
            f"- **Theil Inequality Index**: `{fairness.theil_index:.3f}`",
            f"- **Max-to-Min Ratio**: `{fairness.max_to_min_ratio:.2f}x`",
            f"- **Macro Patches / GB**: `{fairness.macro_patches_per_gb:,.0f} tokens/GB`",
            "",
            "**Top 5 Best Compressing Languages:** " + ", ".join([f"{code} ({val:.2f} B/T)" for code, val in fairness.best_10_languages[:5]]),
            "**Bottom 5 Worst Compressing Languages:** " + ", ".join([f"{code} ({val:.2f} B/T)" for code, val in fairness.worst_10_languages[:5]])
        ]
        return "\n".join(lines)
