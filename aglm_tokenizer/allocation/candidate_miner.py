"""
Language-Conditional Token Candidate Miner and Utility Calculator.
Implements Section 5 of Mandatory Specifications:
Calculates multi-dimensional utility for every token candidate:
- languages where it occurs
- script
- corpus frequency
- held-out frequency
- positions saved: (original_subtoken_count - 1) * freq
- bytes saved: (byte_length - 1) * freq
- cross-language reuse score
- collision risk
- vocab cost
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import math
from aglm_tokenizer.core.token_types import TokenClassifier, TokenType
from aglm_tokenizer.core.script_handlers import ScriptDetector, ScriptType


@dataclass
class CandidateStats:
    token_str: str
    token_bytes: bytes
    script: ScriptType
    token_type: TokenType
    lang_frequencies: Dict[str, int] = field(default_factory=dict)
    held_out_frequencies: Dict[str, int] = field(default_factory=dict)
    total_corpus_freq: int = 0
    total_held_out_freq: int = 0
    positions_saved: int = 0
    bytes_saved: int = 0
    cross_language_entropy: float = 0.0
    collision_risk: float = 0.0
    vocab_cost: int = 1

    def compute_metrics(self, all_languages: List[str]) -> None:
        self.total_corpus_freq = sum(self.lang_frequencies.values())
        self.total_held_out_freq = sum(self.held_out_frequencies.values())

        # Bytes saved: when representing a token of length N bytes instead of individual bytes
        byte_len = len(self.token_bytes)
        self.bytes_saved = max(0, (byte_len - 1)) * self.total_corpus_freq

        # Positions saved: assuming character-level fallback or sub-morpheme fallback
        char_len = len(self.token_str)
        self.positions_saved = max(0, (char_len - 1)) * self.total_corpus_freq

        # Cross-language entropy and sharing score
        if self.total_corpus_freq > 0:
            probs = [c / self.total_corpus_freq for c in self.lang_frequencies.values() if c > 0]
            if len(probs) > 1:
                # Shannon entropy normalized by log(num_languages)
                raw_entropy = -sum(p * math.log2(p) for p in probs)
                max_entropy = math.log2(len(all_languages)) if len(all_languages) > 1 else 1.0
                self.cross_language_entropy = raw_entropy / max_entropy
            else:
                self.cross_language_entropy = 0.0

        # Collision risk: high if a short token occurs across wildly divergent semantic families
        # with low sub-morpheme coherence
        if len(self.token_str) <= 2 and len(self.lang_frequencies) > 4 and self.token_type == TokenType.WHOLE_WORD:
            self.collision_risk = 0.5
        else:
            self.collision_risk = 0.0


class CandidateMiner:
    """Extracts candidate n-grams, subwords, and morphemes across multilingual corpora."""

    def __init__(self, max_token_length: int = 16):
        self.max_token_length = max_token_length
        self.candidates: Dict[str, CandidateStats] = {}

    def add_occurrence(self, text: str, lang_code: str, count: int = 1, is_held_out: bool = False) -> None:
        """Records an occurrence of candidate token."""
        if not text:
            return

        token_bytes = text.encode("utf-8")
        if len(token_bytes) > self.max_token_length:
            return

        if text not in self.candidates:
            st = ScriptDetector.detect_text_script(text)
            tt = TokenClassifier.classify_token(token_bytes, text)
            self.candidates[text] = CandidateStats(
                token_str=text,
                token_bytes=token_bytes,
                script=st,
                token_type=tt
            )

        cand = self.candidates[text]
        if is_held_out:
            cand.held_out_frequencies[lang_code] = cand.held_out_frequencies.get(lang_code, 0) + count
        else:
            cand.lang_frequencies[lang_code] = cand.lang_frequencies.get(lang_code, 0) + count

    def finalize_all(self, all_languages: List[str]) -> Dict[str, CandidateStats]:
        """Computes all multi-dimensional statistics across all candidate tokens."""
        for cand in self.candidates.values():
            cand.compute_metrics(all_languages)
        return self.candidates
