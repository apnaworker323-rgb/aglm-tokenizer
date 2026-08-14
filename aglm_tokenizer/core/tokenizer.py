"""
AGLM Universal Multilingual Tokenizer Interface.
Wraps the BPE engine, token classifier, script detector, and serialization.
"""

from typing import List, Dict, Set, Tuple, Optional, Any
import json
import os
from aglm_tokenizer.core.bpe_engine import BPEEngine
from aglm_tokenizer.core.token_types import TokenClassifier, TokenType, TokenProfile
from aglm_tokenizer.core.script_handlers import ScriptDetector, ScriptType, ScriptSegmenter
from aglm_tokenizer.allocation.candidate_miner import CandidateStats


class AGLMUniversalTokenizer:
    """Universal Multilingual Tokenizer for AGLM."""

    def __init__(self, name: str = "AGLM-Universal", strategy: str = "Strategy_E_Fairness_Aware_Utility"):
        self.name = name
        self.strategy = strategy
        self.engine = BPEEngine()
        self.token_profiles: Dict[int, TokenProfile] = {}
        self._init_special_tokens()

    def _init_special_tokens(self) -> None:
        """Initializes standard control & special tokens."""
        special_tokens = [
            "<|pad|>",
            "<|bos|>",
            "<|eos|>",
            "<|unk|>",
            "<|mask|>",
            "<|lang_switch|>",
            "<|script_switch|>",
            "<|romanized|>",
            "<|code_switch|>"
        ]
        for st in special_tokens:
            self.engine.add_special_token(st)

    @property
    def vocab_size(self) -> int:
        return self.engine.vocab_size

    def add_token(self, token_bytes: bytes, profile: Optional[TokenProfile] = None) -> int:
        tid = self.engine.add_token(token_bytes)
        if profile:
            self.token_profiles[tid] = profile
        return tid

    def encode(self, text: str, allowed_special: Optional[Set[str]] = None) -> List[int]:
        """Encodes text to token IDs."""
        tokens, _ = self.engine.encode(text, allowed_special)
        return tokens

    def encode_with_stats(self, text: str) -> Dict[str, Any]:
        """Encodes text and returns detailed token breakdown and statistics."""
        raw_bytes = text.encode("utf-8")
        tokens, byte_fallbacks = self.engine.encode(text)
        num_tokens = len(tokens)
        num_bytes = len(raw_bytes)
        bytes_per_token = (num_bytes / num_tokens) if num_tokens > 0 else 0.0
        words = text.split()
        num_words = max(1, len(words))
        tokens_per_word = num_tokens / num_words
        fallback_ratio = (byte_fallbacks / num_tokens) if num_tokens > 0 else 0.0

        # Exact lossless check
        reconstructed = self.engine.decode(tokens)
        is_lossless = (reconstructed == text)

        return {
            "tokens": tokens,
            "num_tokens": num_tokens,
            "num_bytes": num_bytes,
            "num_words": num_words,
            "bytes_per_token": bytes_per_token,
            "tokens_per_word": tokens_per_word,
            "byte_fallback_count": byte_fallbacks,
            "byte_fallback_ratio": fallback_ratio,
            "is_lossless": is_lossless
        }

    def decode(self, tokens: List[int]) -> str:
        """Decodes token IDs back to text."""
        return self.engine.decode(tokens)

    def decode_to_bytes(self, tokens: List[int]) -> bytes:
        return self.engine.decode_to_bytes(tokens)

    def save(self, directory: str) -> None:
        """Saves tokenizer vocabulary and metadata to disk."""
        os.makedirs(directory, exist_ok=True)
        vocab_data = {
            "name": self.name,
            "strategy": self.strategy,
            "vocab_size": self.vocab_size,
            "tokens": [
                {
                    "id": tid,
                    "bytes_hex": self.engine.id_to_bytes[tid].hex(),
                    "str": self.engine.id_to_bytes[tid].decode("utf-8", errors="replace")
                }
                for tid in sorted(self.engine.id_to_bytes.keys())
            ],
            "special_tokens": self.engine.special_tokens
        vocab_path = os.path.join(directory, "aglm_vocab.json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)

        # Save compressed version for distribution (<30MB)
        import gzip
        with gzip.open(vocab_path + ".gz", "wt", encoding="utf-8") as f_gz:
            json.dump(vocab_data, f_gz, ensure_ascii=False)

    @classmethod
    def load(cls, directory: str) -> "AGLMUniversalTokenizer":
        """Loads tokenizer from disk, automatically resolving .json or .json.gz."""
        vocab_json = os.path.join(directory, "aglm_vocab.json")
        vocab_gz = os.path.join(directory, "aglm_vocab.json.gz")

        if os.path.exists(vocab_json):
            with open(vocab_json, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif os.path.exists(vocab_gz):
            import gzip
            with gzip.open(vocab_gz, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            raise FileNotFoundError(f"No aglm_vocab.json or aglm_vocab.json.gz found in {directory}")

        tok = cls(name=data.get("name", "AGLM-Loaded"), strategy=data.get("strategy", "unknown"))
        for item in data.get("tokens", []):
            tid = item["id"]
            if tid >= 256:
                token_bytes = bytes.fromhex(item["bytes_hex"])
                tok.engine.add_token(token_bytes, token_id=tid)

        for st, st_id in data.get("special_tokens", {}).items():
            tok.engine.add_special_token(st, token_id=st_id)

        return tok
