"""
Multilingual Tokenizer Evaluation Metrics and Inequality / Fairness Indices.
Implements Sections 8, 9, and 16 of Mandatory Specifications:
- Language-by-language performance table metrics:
  - Raw MB, Token Count, Bytes/Token, Tokens/Word, Byte Fallback %, Encode MB/s
- Macro / Micro aggregations and summary statistics:
  - Macro Patches / GB
  - Best 10 languages, Worst 10 languages
  - Variance and Standard Deviation
- Inequality & Fairness metrics:
  - Best, Median, Worst B/T
  - Percentiles: P10, P50, P90
  - Coefficient of Variation (CV = std / mean)
  - Gini Index of compression disparity
  - Theil Index
  - Max-to-Min ratio
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
import math


@dataclass
class LanguageEvalResult:
    language_code: str
    language_name: str
    script: str
    family: str
    raw_bytes: int
    raw_words: int
    token_count: int
    bytes_per_token: float
    tokens_per_word: float
    byte_fallback_count: int
    byte_fallback_percent: float
    encode_time_sec: float
    encode_throughput_mb_s: float
    is_lossless: bool


@dataclass
class FairnessMetrics:
    best_bytes_per_token: float
    median_bytes_per_token: float
    worst_bytes_per_token: float
    p10: float
    p50: float
    p90: float
    macro_mean: float
    macro_std: float
    coefficient_of_variation: float  # CV = std / mean
    gini_index: float
    theil_index: float
    max_to_min_ratio: float
    worst_10_languages: List[Tuple[str, float]]
    best_10_languages: List[Tuple[str, float]]
    macro_patches_per_gb: float


class MetricsCalculator:
    """Calculates granular language-level and global inequality metrics."""

    @staticmethod
    def compute_gini(values: List[float]) -> float:
        """Computes the Gini coefficient of inequality."""
        if not values or len(values) == 1:
            return 0.0
        arr = np.sort(np.array(values, dtype=np.float64))
        n = len(arr)
        index = np.arange(1, n + 1)
        return float((np.sum((2 * index - n - 1) * arr)) / (n * np.sum(arr)))

    @staticmethod
    def compute_theil(values: List[float]) -> float:
        """Computes Theil index of inequality (T_T)."""
        arr = np.array(values, dtype=np.float64)
        mean_val = np.mean(arr)
        if mean_val <= 0:
            return 0.0
        ratios = arr / mean_val
        # Avoid log(0)
        ratios = np.where(ratios > 0, ratios, 1e-12)
        return float(np.mean(ratios * np.log(ratios)))

    @classmethod
    def compute_fairness_summary(cls, results: List[LanguageEvalResult]) -> FairnessMetrics:
        """Computes full fairness, percentiles, and inequality metrics across all evaluated languages."""
        if not results:
            return FairnessMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [], [], 0)

        bpt_values = [r.bytes_per_token for r in results]
        arr = np.array(bpt_values, dtype=np.float64)

        best_bpt = float(np.max(arr))
        median_bpt = float(np.median(arr))
        worst_bpt = float(np.min(arr))

        p10 = float(np.percentile(arr, 10))
        p50 = float(np.percentile(arr, 50))
        p90 = float(np.percentile(arr, 90))

        macro_mean = float(np.mean(arr))
        macro_std = float(np.std(arr))
        cv = float(macro_std / macro_mean) if macro_mean > 0 else 0.0

        gini = cls.compute_gini(bpt_values)
        theil = cls.compute_theil(bpt_values)
        max_to_min = float(best_bpt / worst_bpt) if worst_bpt > 0 else 999.0

        # Best and worst 10 languages by Bytes/Token
        sorted_by_bpt = sorted(results, key=lambda r: r.bytes_per_token, reverse=True)
        best_10 = [(r.language_code, r.bytes_per_token) for r in sorted_by_bpt[:10]]
        worst_10 = [(r.language_code, r.bytes_per_token) for r in sorted_by_bpt[-10:]]

        # Macro patches / GB: 1 GB / (macro_mean bytes/token * patch_size_tokens)
        # Standard patch unit: tokens required to represent 1 GB
        tokens_per_gb = (1024 * 1024 * 1024) / macro_mean if macro_mean > 0 else 0
        macro_patches_per_gb = tokens_per_gb

        return FairnessMetrics(
            best_bytes_per_token=best_bpt,
            median_bytes_per_token=median_bpt,
            worst_bytes_per_token=worst_bpt,
            p10=p10,
            p50=p50,
            p90=p90,
            macro_mean=macro_mean,
            macro_std=macro_std,
            coefficient_of_variation=cv,
            gini_index=gini,
            theil_index=theil,
            max_to_min_ratio=max_to_min,
            worst_10_languages=worst_10,
            best_10_languages=best_10,
            macro_patches_per_gb=macro_patches_per_gb
        )
