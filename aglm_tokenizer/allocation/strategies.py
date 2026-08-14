"""
Language-Balanced Vocabulary Allocation Strategies.
Implements Section 4 of Mandatory Specifications:
A. Raw frequency allocation
B. Temperature-smoothed language allocation (p_i ~ f_i^(1/T))
C. Minimum guaranteed allocation per language/script
D. Utility-weighted allocation: frequency * positions_saved
E. Fairness-aware utility: language_weight * frequency * positions_saved
"""

from enum import Enum
import math
from typing import Dict, List, Set, Tuple
from aglm_tokenizer.allocation.candidate_miner import CandidateStats
from aglm_tokenizer.core.script_handlers import ScriptType


class AllocationStrategyType(str, Enum):
    A_RAW_FREQUENCY = "Strategy_A_Raw_Frequency"
    B_TEMP_SMOOTHED = "Strategy_B_Temp_Smoothed"
    C_MIN_GUARANTEE = "Strategy_C_Min_Guarantee"
    D_UTILITY_WEIGHTED = "Strategy_D_Utility_Weighted"
    E_FAIRNESS_AWARE_UTILITY = "Strategy_E_Fairness_Aware_Utility"


class VocabAllocator:
    """Implements and benchmarks vocabulary allocation algorithms."""

    @staticmethod
    def allocate_vocab(
        candidates: Dict[str, CandidateStats],
        target_vocab_size: int,
        strategy: AllocationStrategyType,
        temperature: float = 2.5,
        min_guarantee_per_script: int = 500,
        reserved_base_tokens: int = 256,  # 256 raw bytes guaranteed
        all_languages: List[str] = None
    ) -> List[CandidateStats]:
        """
        Allocates vocabulary slots to candidate tokens based on the chosen strategy.
        Guarantees inclusion of the 256 byte-level fallbacks first.
        """
        all_cands = list(candidates.values())
        available_slots = max(1, target_vocab_size - reserved_base_tokens)

        # -------------------------------------------------------------
        # Strategy A: Raw frequency allocation
        # -------------------------------------------------------------
        if strategy == AllocationStrategyType.A_RAW_FREQUENCY:
            sorted_cands = sorted(all_cands, key=lambda c: c.total_corpus_freq, reverse=True)
            return sorted_cands[:available_slots]

        # -------------------------------------------------------------
        # Strategy B: Temperature-smoothed language allocation
        # -------------------------------------------------------------
        elif strategy == AllocationStrategyType.B_TEMP_SMOOTHED:
            # Aggregate total corpus mass per language
            lang_totals: Dict[str, int] = {}
            for c in all_cands:
                for lang, freq in c.lang_frequencies.items():
                    lang_totals[lang] = lang_totals.get(lang, 0) + freq

            # Compute smoothed language weights: w_lang = (F_lang / F_total) ^ (1/T)
            total_mass = sum(lang_totals.values()) or 1
            smooth_weights: Dict[str, float] = {}
            for lang, mass in lang_totals.items():
                p = mass / total_mass
                smooth_weights[lang] = math.pow(p, 1.0 / temperature) / p if p > 0 else 1.0

            # Score each candidate: sum(freq_lang * smooth_weight_lang)
            def score_b(c: CandidateStats) -> float:
                return sum(f * smooth_weights.get(l, 1.0) for l, f in c.lang_frequencies.items())

            sorted_cands = sorted(all_cands, key=score_b, reverse=True)
            return sorted_cands[:available_slots]

        # -------------------------------------------------------------
        # Strategy C: Minimum guaranteed allocation per language/script
        # -------------------------------------------------------------
        elif strategy == AllocationStrategyType.C_MIN_GUARANTEE:
            # Group by script
            script_buckets: Dict[ScriptType, List[CandidateStats]] = {}
            for c in all_cands:
                script_buckets.setdefault(c.script, []).append(c)

            for s in script_buckets:
                script_buckets[s].sort(key=lambda c: c.total_corpus_freq, reverse=True)

            selected_set: Set[str] = set()
            selected_list: List[CandidateStats] = []

            # Step 1: Guarantee min tokens per script
            for script, cands in script_buckets.items():
                guarantee_count = min(len(cands), min_guarantee_per_script)
                for i in range(guarantee_count):
                    c = cands[i]
                    if c.token_str not in selected_set:
                        selected_set.add(c.token_str)
                        selected_list.append(c)

            # Step 2: Fill remainder by raw utility / frequency
            remaining_slots = available_slots - len(selected_list)
            if remaining_slots > 0:
                remaining_cands = [c for c in all_cands if c.token_str not in selected_set]
                remaining_cands.sort(key=lambda c: c.total_corpus_freq, reverse=True)
                selected_list.extend(remaining_cands[:remaining_slots])

            return selected_list[:available_slots]

        # -------------------------------------------------------------
        # Strategy D: Utility-weighted allocation (freq * positions_saved)
        # -------------------------------------------------------------
        elif strategy == AllocationStrategyType.D_UTILITY_WEIGHTED:
            def score_d(c: CandidateStats) -> float:
                # Positions saved * total frequency
                return float(c.positions_saved)

            sorted_cands = sorted(all_cands, key=score_d, reverse=True)
            return sorted_cands[:available_slots]

        # -------------------------------------------------------------
        # Strategy E: Fairness-aware utility: language_weight * freq * positions_saved
        # -------------------------------------------------------------
        elif strategy == AllocationStrategyType.E_FAIRNESS_AWARE_UTILITY:
            # Dynamic inverse-mass and script balancing
            # Computes language-fairness multiplier to prevent low-resource languages
            # and non-Latin scripts from falling back excessively to raw UTF-8 bytes.
            lang_totals: Dict[str, int] = {}
            for c in all_cands:
                for lang, freq in c.lang_frequencies.items():
                    lang_totals[lang] = lang_totals.get(lang, 0) + freq

            max_mass = max(lang_totals.values()) if lang_totals else 1
            fairness_weights: Dict[str, float] = {}
            for lang, mass in lang_totals.items():
                # Inverse root scaling + cross-language entropy bonus
                ratio = max_mass / max(1, mass)
                fairness_weights[lang] = math.sqrt(ratio)

            def score_e(c: CandidateStats) -> float:
                # Weighted utility across each occurring language
                lang_utility = sum(
                    (f * (len(c.token_bytes) - 1)) * fairness_weights.get(l, 1.0)
                    for l, f in c.lang_frequencies.items()
                )
                # Cross-language entropy bonus (promotes reusable international tokens)
                entropy_multiplier = 1.0 + (c.cross_language_entropy * 1.2)
                # Script complexity bonus for multi-byte Unicode scripts (Indic, CJK, Arabic, Dravidian)
                script_bonus = 1.3 if c.script not in (ScriptType.LATIN, ScriptType.PUNCTUATION_SPACE) else 1.0
                # Collision penalty
                collision_penalty = 1.0 - c.collision_risk

                return (lang_utility * entropy_multiplier * script_bonus * collision_penalty)

            sorted_cands = sorted(all_cands, key=score_e, reverse=True)
            return sorted_cands[:available_slots]

        else:
            raise ValueError(f"Unknown allocation strategy: {strategy}")
