"""
Canonical Multi-Tokenizer Vocabulary Harvester and Classifier.
Extracts, canonicalizes to exact bytes, classifies, and computes overlap across:
- OpenAI o200k_base
- OpenAI cl100k_base
- XLM-V
- XLM-RoBERTa
- Gemma 2
- DeepSeek V3
- Qwen 2.5
- Llama 3
- Mistral v0.3
- AGLM local candidates
"""

from typing import Dict, List, Set, Tuple, Any, Optional
import os
import json
import csv
import regex as re
import unicodedata
import tiktoken
from transformers import AutoTokenizer

from aglm_tokenizer.core.token_types import TokenClassifier, TokenType
from aglm_tokenizer.core.script_handlers import ScriptDetector, ScriptType


def gpt2_bytes_to_unicode() -> Dict[str, int]:
    """Inverts GPT-2 / Llama / Qwen / DeepSeek byte-level BPE character mapping."""
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip([chr(c) for c in cs], bs))


BYTE_BPE_INVERSE = gpt2_bytes_to_unicode()


def decode_byte_bpe_string(s: str) -> bytes:
    """Converts byte-level BPE string (e.g. 'ĠThe') to exact raw bytes (b' The')."""
    return bytes([BYTE_BPE_INVERSE[c] for c in s if c in BYTE_BPE_INVERSE])


def decode_spm_piece(piece: str) -> bytes:
    """Converts SentencePiece piece (e.g. ' The' or '<0xXX>') to exact raw bytes."""
    if piece.startswith("<0x") and piece.endswith(">") and len(piece) == 6:
        try:
            return bytes([int(piece[3:5], 16)])
        except ValueError:
            pass
    # Replace U+2581 (' ') with standard space (0x20)
    cleaned = piece.replace("\u2581", " ")
    return cleaned.encode("utf-8")


class CanonicalTokenPool:
    """Stores exact-byte unified tokens with multi-tokenizer provenance and linguistic classification."""

    def __init__(self):
        # raw_bytes -> dict of metadata
        self.pool: Dict[bytes, Dict[str, Any]] = {}
        self.tokenizer_vocab_sizes: Dict[str, int] = {}

    def add_token(self, raw_bytes: bytes, source_tokenizer: str, source_id: int) -> None:
        if not raw_bytes:
            return

        if raw_bytes not in self.pool:
            try:
                printable_str = raw_bytes.decode("utf-8")
                is_valid_utf8 = True
            except UnicodeDecodeError:
                printable_str = raw_bytes.hex()
                is_valid_utf8 = False

            script = ScriptDetector.detect_text_script(printable_str) if is_valid_utf8 else ScriptType.UNKNOWN
            structural_type = self._classify_structural_type(raw_bytes, printable_str, is_valid_utf8)

            self.pool[raw_bytes] = {
                "bytes_hex": raw_bytes.hex(),
                "text": printable_str,
                "is_valid_utf8": is_valid_utf8,
                "byte_length": len(raw_bytes),
                "script": script.value,
                "structural_type": structural_type,
                "sources": {}
            }

        self.pool[raw_bytes]["sources"][source_tokenizer] = source_id

    @staticmethod
    def _classify_structural_type(raw_bytes: bytes, text: str, is_valid_utf8: bool) -> str:
        if len(raw_bytes) == 1:
            return "BYTE"
        if not is_valid_utf8:
            return "BYTE"

        if text.startswith("<|") and text.endswith("|>"):
            return "OTHER"

        # CJK
        if any('\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf' or '\u3040' <= c <= '\u30ff' or '\uac00' <= c <= '\ud7af' for c in text):
            return "CJK"

        # Indic
        if any('\u0900' <= c <= '\u0d7f' for c in text):
            return "INDIC"

        # Arabic
        if any('\u0600' <= c <= '\u08ff' for c in text):
            return "ARABIC"

        # Cyrillic
        if any('\u0400' <= c <= '\u052f' for c in text):
            return "CYRILLIC"

        # Number
        if re.fullmatch(r'\d+(\.\d+)?', text.strip()):
            return "NUMBER"

        # Punctuation
        if re.fullmatch(r'[^\w\s]+', text.strip()):
            return "PUNCTUATION"

        # Code keywords
        if text.strip() in {"def", "class", "function", "const", "let", "var", "import", "return", "if", "else", "while", "for", "public", "private", "int", "float", "bool", "string"}:
            return "CODE"

        # Romanized Indic / transliteration units heuristic
        if re.match(r'^(aa|ee|oo|kh|gh|ch|jh|th|dh|ph|bh|sh|zh|ng|ny|ts|shk)', text.strip(), re.IGNORECASE):
            return "ROMANIZED"

        # Space word (leading space attached to whole word)
        if text.startswith(" ") and len(text) > 1 and text[1:].isalpha():
            return "SPACE_WORD"

        # Multiword
        if " " in text.strip():
            return "MULTIWORD"

        # Character vs Subword vs Word
        if len(text) == 1:
            return "CHARACTER"
        if text.isalpha():
            if len(text) <= 3:
                return "SUBWORD"
            return "WORD"

        return "SUBWORD"

    def harvest_all_tokenizers(self) -> None:
        """Loads and harvests vocabularies from all 9 production tokenizers."""
        print("[HARVESTER] 1. Harvesting OpenAI o200k_base...")
        o200k = tiktoken.get_encoding("o200k_base")
        self.tokenizer_vocab_sizes["o200k_base"] = o200k.n_vocab
        for tid in range(o200k.n_vocab):
            try:
                b = o200k.decode_single_token_bytes(tid)
                self.add_token(b, "o200k_base", tid)
            except Exception:
                pass

        print("[HARVESTER] 2. Harvesting OpenAI cl100k_base...")
        cl100k = tiktoken.get_encoding("cl100k_base")
        self.tokenizer_vocab_sizes["cl100k_base"] = cl100k.n_vocab
        for tid in range(cl100k.n_vocab):
            try:
                b = cl100k.decode_single_token_bytes(tid)
                self.add_token(b, "cl100k_base", tid)
            except Exception:
                pass

        # HuggingFace tokenizers
        hf_models = [
            ("qwen2.5", "Qwen/Qwen2.5-7B", "byte_bpe"),
            ("gemma2", "unsloth/gemma-2-9b", "spm"),
            ("deepseek_v3", "deepseek-ai/DeepSeek-V3", "byte_bpe"),
            ("llama3", "NousResearch/Meta-Llama-3-8B", "byte_bpe"),
            ("mistral_v0.3", "mistralai/Mistral-7B-v0.3", "spm"),
            ("xlm_roberta", "xlm-roberta-base", "spm"),
            ("xlm_v", "facebook/xlm-v-base", "spm"),
        ]

        for tag, model_id, kind in hf_models:
            print(f"[HARVESTER] Harvesting {tag} ({model_id})...")
            try:
                tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                vocab = tok.get_vocab()
                self.tokenizer_vocab_sizes[tag] = len(vocab)
                for piece, tid in vocab.items():
                    if kind == "byte_bpe":
                        b = decode_byte_bpe_string(piece)
                    else:
                        b = decode_spm_piece(piece)
                    if b:
                        self.add_token(b, tag, tid)
            except Exception as e:
                print(f"[ERROR] Failed to harvest {tag}: {e}")

        print(f"[HARVESTER] Total unique canonical tokens extracted: {len(self.pool):,}")

    def compute_overlap_matrix(self) -> Tuple[List[str], List[List[int]], List[List[float]]]:
        """Computes pairwise raw token overlap and Jaccard similarity between all tokenizers."""
        tok_names = sorted(self.tokenizer_vocab_sizes.keys())
        n = len(tok_names)

        # Build set of token bytes per tokenizer
        tok_token_sets = {name: set() for name in tok_names}
        for raw_bytes, meta in self.pool.items():
            for src in meta["sources"]:
                if src in tok_token_sets:
                    tok_token_sets[src].add(raw_bytes)

        raw_overlap = [[0] * n for _ in range(n)]
        jaccard_overlap = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                s1 = tok_token_sets[tok_names[i]]
                s2 = tok_token_sets[tok_names[j]]
                intersection = len(s1 & s2)
                union = len(s1 | s2)
                raw_overlap[i][j] = intersection
                jaccard_overlap[i][j] = (intersection / union) if union > 0 else 0.0

        return tok_names, raw_overlap, jaccard_overlap

    def export_canonical_jsonl(self, filepath: str) -> None:
        """Exports full canonical token pool to JSONL."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for raw_bytes, meta in self.pool.items():
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    def export_overlap_csv(self, filepath: str) -> None:
        """Exports pairwise overlap matrix to CSV."""
        names, raw_mat, jaccard_mat = self.compute_overlap_matrix()
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Tokenizer"] + names)
            for i, name in enumerate(names):
                writer.writerow([name] + [f"{raw_mat[i][j]:,} ({jaccard_mat[i][j]*100:.1f}%)" for j in range(len(names))])
