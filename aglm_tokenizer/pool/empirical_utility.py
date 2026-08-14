"""
Empirical Multilingual Utility Scorer for Canonical Token Candidates.
Calculates language-balanced, temperature-smoothed utility strictly on TRAINING data.
Zero held-out data is used for scoring.
"""

from typing import Dict, List, Set, Tuple, Any
import math
import numpy as np

from aglm_tokenizer.corpus.language_registry import LANGUAGES
from aglm_tokenizer.corpus.multilingual_corpus import MultilingualCorpusManager
from aglm_tokenizer.core.script_handlers import ScriptSegmenter


class EmpiricalUtilityScorer:
    """Computes empirical multi-lingual utility for canonical token pool candidates."""

    def __init__(self, temperature: float = 2.5):
        self.temperature = temperature
        self.lang_training_corpora: Dict[str, str] = {}
        self.lang_masses: Dict[str, int] = {}
        self._load_training_corpora()

    def _load_training_corpora(self) -> None:
        """Loads and indexes training data across all 50+ languages."""
        for code in LANGUAGES.keys():
            train_text = MultilingualCorpusManager.get_training_corpus(code)
            self.lang_training_corpora[code] = train_text
            self.lang_masses[code] = len(train_text.encode("utf-8"))

    def score_canonical_pool(self, canonical_pool: Dict[bytes, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Scores all canonical candidates across languages.
        Returns sorted list of scored candidates.
        """
        print("[UTILITY SCORER] Computing empirical language-balanced utility for canonical pool...")

        total_corpus_mass = sum(self.lang_masses.values()) or 1
        lang_weights = {}
        for code, mass in self.lang_masses.items():
            p = mass / total_corpus_mass
            # Temperature smoothing: w_i = (p_i)^(1/T) / p_i
            lang_weights[code] = math.pow(p, 1.0 / self.temperature) / p if p > 0 else 1.0

        scored_candidates = []

        # Count occurrences and compute positions saved
        for raw_bytes, meta in canonical_pool.items():
            token_str = meta["text"]
            byte_len = meta["byte_length"]
            num_sources = len(meta["sources"])

            # Consensus bonus: tokens present in multiple production tokenizers get a confidence prior
            consensus_multiplier = 1.0 + (num_sources * 0.15)

            # Measure frequency across training sets
            total_freq = 0
            weighted_utility = 0.0
            lang_hits = {}

            if meta["is_valid_utf8"] and len(token_str) > 0:
                for lang_code, train_text in self.lang_training_corpora.items():
                    # Fast substring count
                    c = train_text.count(token_str)
                    if c > 0:
                        total_freq += c
                        lang_hits[lang_code] = c
                        positions_saved = max(1, (byte_len - 1)) * c
                        weighted_utility += positions_saved * lang_weights.get(lang_code, 1.0)

            # If token didn't directly occur as exact substring in seed passages, assign baseline prior from source tokenizers
            if total_freq == 0:
                total_freq = num_sources
                weighted_utility = float(num_sources * max(1, byte_len - 1)) * 0.5

            final_utility = weighted_utility * consensus_multiplier

            scored_item = dict(meta)
            scored_item["raw_bytes"] = raw_bytes
            scored_item["corpus_frequency"] = total_freq
            scored_item["language_hits"] = lang_hits
            scored_item["consensus_count"] = num_sources
            scored_item["empirical_utility"] = final_utility
            scored_candidates.append(scored_item)

        # Sort by empirical utility descending
        scored_candidates.sort(key=lambda x: x["empirical_utility"], reverse=True)
        return scored_candidates
