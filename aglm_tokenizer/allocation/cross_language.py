"""
Cross-Language Sharing and Collision Analyzer.
Implements Section 11 of Mandatory Specifications:
Analyzes tokens/subwords reusable across languages:
- Latin roots shared across European languages (e.g., 'tion', 'inter', 'struct', 'trans')
- Named entities & international vocabulary ('Washington', 'Tokyo', 'Berlin', 'Gandhi', 'Python')
- Scientific / medical terminology ('tele', 'micro', 'macro', 'bio', 'geo', 'quantum', 'algorithm')
- Technical vocabulary & Code ('function', 'return', 'import', 'async', 'class', 'const')
- Numbers & dates ('2024', '100', '1999', '01')
- Loanwords across languages ('chai', 'karaoke', 'guru', 'bazaar', 'algebra', 'tsunami')
- Measures sharing benefit vs semantic collision risk.
"""

from typing import Dict, List, Set, Tuple
from aglm_tokenizer.allocation.candidate_miner import CandidateStats
from aglm_tokenizer.core.token_types import TokenType


class CrossLanguageAnalyzer:
    """Analyzes cross-lingual reuse benefits vs semantic collision risks."""

    SHARED_INTERNATIONAL_ROOTS = {
        "inter", "trans", "sub", "super", "micro", "macro", "tele", "auto", "bio", "geo",
        "hydro", "thermo", "chrono", "photo", "psych", "neuro", "socio", "econom", "politi",
        "struct", "port", "form", "script", "tract", "dict", "duc", "gress", "ject", "pel",
        "pend", "pos", "rupt", "sect", "spect", "vert", "vid", "voc", "gen", "log"
    }

    LOANWORDS_AND_ENTITIES = {
        "python", "linux", "google", "ai", "model", "data", "internet", "computer",
        "chai", "tea", "coffee", "sugar", "pizza", "pasta", "sushi", "bazaar", "guru",
        "yoga", "algebra", "algorithm", "zero", "shampoo", "avatar", "zen", "tsunami"
    }

    @classmethod
    def analyze_token_sharing(cls, cand: CandidateStats) -> Dict[str, any]:
        """
        Analyzes whether a candidate token offers clean cross-lingual sharing or creates collisions.
        """
        token_str = cand.token_str.strip().lower()
        num_langs = len(cand.lang_frequencies)
        entropy = cand.cross_language_entropy

        is_root = token_str in cls.SHARED_INTERNATIONAL_ROOTS
        is_entity_or_tech = (token_str in cls.LOANWORDS_AND_ENTITIES or
                             cand.token_type in (TokenType.CODE_SYNTAX, TokenType.NUMBER, TokenType.URL_EMAIL))

        # Sharing efficiency score
        # Benefit = total saved bytes * entropy factor
        sharing_efficiency = (cand.bytes_saved * (1.0 + entropy * 1.5))

        # Collision penalty: high if 1-2 char token occurs across divergent scripts without morphology justification
        collision_penalty = 0.0
        if len(token_str) <= 2 and num_langs > 5 and not is_root and not cand.token_type == TokenType.WHITESPACE_FACTOR:
            collision_penalty = 0.4

        net_utility = sharing_efficiency * (1.0 - collision_penalty)

        return {
            "token_str": cand.token_str,
            "num_languages": num_langs,
            "is_shared_root": is_root,
            "is_entity_or_tech": is_entity_or_tech,
            "cross_language_entropy": entropy,
            "collision_penalty": collision_penalty,
            "sharing_efficiency": sharing_efficiency,
            "net_utility": net_utility
        }
