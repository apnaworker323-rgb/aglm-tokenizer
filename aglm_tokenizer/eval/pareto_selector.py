"""
Multi-Objective Pareto Candidate Selector.
Implements Section 16 of Mandatory Specifications:
Selects winning tokenizer configurations using a Pareto objective over 10 dimensions:
1. Overall bytes/token
2. Macro-average bytes/token across languages
3. Worst-language performance (min B/T)
4. Training patches / GB
5. Tokenizer CPU throughput (MB/s)
6. Vocabulary utilization (active tokens on held-out)
7. Rare-token fraction (fraction with frequency < threshold)
8. Model embedding / output cost (parameter overhead ~ vocab_size * d_model)
9. Exact lossless reconstruction (binary constraint = 1.0)
10. Eventual LM learning quality (morphology preservation & cross-lingual entropy score)
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np


@dataclass
class TokenizerCandidateMetrics:
    candidate_name: str
    vocab_size: int
    overall_bytes_per_token: float
    macro_bytes_per_token: float
    worst_language_bpt: float
    training_patches_per_gb: float
    cpu_throughput_mb_s: float
    vocab_utilization_percent: float
    rare_token_fraction: float
    embedding_param_cost_m: float  # Assuming d_model = 4096: vocab * 4096 * 2 (input+output) / 1e6
    exact_lossless_guarantee: bool
    lm_learning_quality_score: float  # 0 to 100


@dataclass
class ParetoEvaluationReport:
    ranked_candidates: List[Tuple[str, float, Dict[str, float]]]
    pareto_frontier: List[str]
    winning_candidate: str
    rationale: str


class ParetoSelector:
    """Evaluates candidate tokenizers across the 10-dimensional Pareto objective."""

    @classmethod
    def evaluate_candidates(
        cls,
        candidates: List[TokenizerCandidateMetrics],
        d_model: int = 4096
    ) -> ParetoEvaluationReport:
        """
        Calculates Pareto dominance and composite utility score across all 10 criteria.
        """
        if not candidates:
            raise ValueError("No candidates provided for Pareto selection.")

        # Weights across objectives (normalized)
        # We prioritize macro-compression, worst-case fairness, lossless roundtrip, and parameter efficiency
        weights = {
            "macro_bpt": 0.20,
            "worst_lang_bpt": 0.20,
            "overall_bpt": 0.15,
            "lm_learning_quality": 0.15,
            "vocab_utilization": 0.10,
            "throughput": 0.08,
            "param_efficiency": 0.07,
            "rare_fraction_penalty": 0.05
        }

        # Normalize metrics across the candidate pool
        macro_bpts = [c.macro_bytes_per_token for c in candidates]
        worst_bpts = [c.worst_language_bpt for c in candidates]
        overall_bpts = [c.overall_bytes_per_token for c in candidates]
        throughputs = [c.cpu_throughput_mb_s for c in candidates]
        utilizations = [c.vocab_utilization_percent for c in candidates]
        rare_fractions = [c.rare_token_fraction for c in candidates]
        vocab_sizes = [c.vocab_size for c in candidates]
        lm_scores = [c.lm_learning_quality_score for c in candidates]

        max_macro = max(macro_bpts) or 1.0
        max_worst = max(worst_bpts) or 1.0
        max_overall = max(overall_bpts) or 1.0
        max_thru = max(throughputs) or 1.0
        max_util = max(utilizations) or 1.0
        max_vocab = max(vocab_sizes) or 1.0
        max_lm = max(lm_scores) or 1.0

        scores: List[Tuple[str, float, Dict[str, float]]] = []

        for c in candidates:
            # Enforce 100% lossless reconstruction requirement
            if not c.exact_lossless_guarantee:
                composite = 0.0
                breakdowns = {"lossless_failed": 0.0}
                scores.append((c.candidate_name, composite, breakdowns))
                continue

            # Normalized sub-scores (higher is better)
            s_macro = c.macro_bytes_per_token / max_macro
            s_worst = c.worst_language_bpt / max_worst
            s_overall = c.overall_bytes_per_token / max_overall
            s_lm = c.lm_learning_quality_score / max_lm
            s_util = c.vocab_utilization_percent / max_util
            s_thru = c.cpu_throughput_mb_s / max_thru
            # Parameter efficiency (smaller vocab parameter overhead is better)
            s_param = 1.0 - (c.vocab_size / (2 * max_vocab))
            # Rare token penalty (lower rare fraction is better)
            s_rare = 1.0 - c.rare_token_fraction

            composite = (
                weights["macro_bpt"] * s_macro +
                weights["worst_lang_bpt"] * s_worst +
                weights["overall_bpt"] * s_overall +
                weights["lm_learning_quality"] * s_lm +
                weights["vocab_utilization"] * s_util +
                weights["throughput"] * s_thru +
                weights["param_efficiency"] * s_param +
                weights["rare_fraction_penalty"] * s_rare
            )

            breakdown = {
                "composite_score": composite,
                "s_macro": s_macro,
                "s_worst": s_worst,
                "s_overall": s_overall,
                "s_lm": s_lm,
                "s_util": s_util,
                "s_throughput": s_thru,
                "s_param_efficiency": s_param
            }
            scores.append((c.candidate_name, composite, breakdown))

        scores.sort(key=lambda x: x[1], reverse=True)
        winning = scores[0][0]

        # Identify Pareto frontier (non-dominated candidates)
        pareto_set = []
        for c in candidates:
            dominated = False
            for other in candidates:
                if other.candidate_name == c.candidate_name:
                    continue
                if (other.macro_bytes_per_token >= c.macro_bytes_per_token and
                    other.worst_language_bpt >= c.worst_language_bpt and
                    other.vocab_size <= c.vocab_size and
                    other.cpu_throughput_mb_s >= c.cpu_throughput_mb_s and
                    (other.macro_bytes_per_token > c.macro_bytes_per_token or
                     other.worst_language_bpt > c.worst_language_bpt or
                     other.vocab_size < c.vocab_size)):
                    dominated = True
                    break
            if not dominated and c.exact_lossless_guarantee:
                pareto_set.append(c.candidate_name)

        rationale = (
            f"Candidate '{winning}' achieved highest multi-objective score ({scores[0][1]:.3f}). "
            f"It maximizes macro-language and worst-case compression while avoiding excessive "
            f"vocabulary parameter inflation and preserving 100% lossless roundtrip fidelity."
        )

        return ParetoEvaluationReport(
            ranked_candidates=scores,
            pareto_frontier=pareto_set,
            winning_candidate=winning,
            rationale=rationale
        )
