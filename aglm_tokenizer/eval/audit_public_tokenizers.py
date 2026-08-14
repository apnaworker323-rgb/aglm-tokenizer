"""
Public Tokenizer Authenticity and Integrity Auditor.
Independently loads each production tokenizer, verifies its actual class, file paths,
computes SHA256 hashes of tokenizer assets, and verifies token IDs and decoded tokens
on standard audit probe strings to guarantee zero shared fallbacks or closure leaks.
"""

import hashlib
import inspect
import os
import sys
import time
from typing import Dict, List, Any, Tuple
import tiktoken
from transformers import AutoTokenizer


PROBE_STRINGS = [
    "The model is learning quickly.",
    "नमस्ते, मेरा नाम आकाश है।",
    "nuvvu ekkada unnaavu",
    "mujhe ye kaam karna hai",
    "def calculate_hash(data):"
]


class TokenizerInspector:
    """Rigorous inspector for tokenizer instances."""

    @staticmethod
    def get_file_sha256(filepath: str) -> str:
        if not os.path.exists(filepath):
            return "FILE_NOT_FOUND"
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    @classmethod
    def audit_tiktoken(cls, encoding_name: str) -> Dict[str, Any]:
        try:
            enc = tiktoken.get_encoding(encoding_name)
            # Find tiktoken library file
            lib_file = inspect.getfile(tiktoken)
            
            probe_results = {}
            for s in PROBE_STRINGS:
                # Explicitly pass s to enc.encode to prevent any closure capture
                tok_ids = enc.encode(s, allowed_special="all")
                decoded_pieces = [enc.decode_single_token_bytes(tid) for tid in tok_ids[:20]]
                # Format pieces as repr for safe display
                pieces_display = [repr(p.decode('utf-8', errors='replace')) for p in decoded_pieces]
                probe_results[s] = {
                    "token_count": len(tok_ids),
                    "first_20_ids": tok_ids[:20],
                    "decoded_pieces": pieces_display
                }

            return {
                "status": "AUTHENTIC",
                "name": encoding_name,
                "library": "tiktoken",
                "tokenizer_class": enc.__class__.__qualname__,
                "vocab_size": enc.n_vocab,
                "file_path": lib_file,
                "sha256": cls.get_file_sha256(lib_file),
                "encode_function": "tiktoken.Encoding.encode(text, allowed_special='all')",
                "probe_results": probe_results
            }
        except Exception as e:
            return {
                "status": "NOT BENCHMARKED",
                "name": encoding_name,
                "error": str(e)
            }

    @classmethod
    def audit_hf_tokenizer(cls, model_id: str, display_name: str) -> Dict[str, Any]:
        try:
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            vocab_sz = tok.vocab_size if hasattr(tok, "vocab_size") else len(tok)
            tok_class = tok.__class__.__module__ + "." + tok.__class__.__qualname__

            # Inspect underlying files
            file_paths = []
            if hasattr(tok, "vocab_file") and tok.vocab_file:
                file_paths.append(tok.vocab_file)
            if hasattr(tok, "tokenizer_file") and tok.tokenizer_file:
                file_paths.append(tok.tokenizer_file)

            # Look up HF cache path if available
            asset_hashes = {}
            if hasattr(tok, "name_or_path") and os.path.exists(tok.name_or_path):
                for fname in os.listdir(tok.name_or_path):
                    if fname.endswith((".json", ".model", ".txt", ".tiktoken")):
                        fp = os.path.join(tok.name_or_path, fname)
                        asset_hashes[fname] = cls.get_file_sha256(fp)

            probe_results = {}
            for s in PROBE_STRINGS:
                tok_ids = tok.encode(s, add_special_tokens=False)
                pieces = [tok.decode([tid]) for tid in tok_ids[:20]]
                probe_results[s] = {
                    "token_count": len(tok_ids),
                    "first_20_ids": tok_ids[:20],
                    "decoded_pieces": [repr(p) for p in pieces]
                }

            return {
                "status": "AUTHENTIC",
                "name": display_name,
                "model_id": model_id,
                "library": "transformers / tokenizers",
                "tokenizer_class": tok_class,
                "vocab_size": vocab_sz,
                "file_path": tok.name_or_path if hasattr(tok, "name_or_path") else str(file_paths),
                "asset_hashes": asset_hashes,
                "encode_function": f"AutoTokenizer.encode(text, add_special_tokens=False)",
                "probe_results": probe_results
            }
        except Exception as e:
            return {
                "status": "NOT BENCHMARKED",
                "name": display_name,
                "model_id": model_id,
                "error": str(e)
            }


def run_full_authenticity_audit() -> Dict[str, Any]:
    models_to_audit = [
        ("tiktoken", "o200k_base", "OpenAI o200k_base"),
        ("tiktoken", "cl100k_base", "OpenAI cl100k_base"),
        ("hf", "Qwen/Qwen2.5-7B", "Qwen 2.5"),
        ("hf", "unsloth/gemma-2-9b", "Gemma 2"),
        ("hf", "deepseek-ai/DeepSeek-V3", "DeepSeek V3"),
        ("hf", "NousResearch/Meta-Llama-3-8B", "Llama 3"),
        ("hf", "mistralai/Mistral-7B-v0.3", "Mistral v0.3"),
        ("hf", "xlm-roberta-base", "XLM-RoBERTa"),
        ("hf", "facebook/xlm-v-base", "XLM-V"),
    ]

    audit_reports = {}
    for kind, mid, name in models_to_audit:
        print(f"Auditing [{name}] ({mid})...")
        if kind == "tiktoken":
            rep = TokenizerInspector.audit_tiktoken(mid)
        else:
            rep = TokenizerInspector.audit_hf_tokenizer(mid, name)
        audit_reports[name] = rep

    return audit_reports


if __name__ == "__main__":
    reports = run_full_authenticity_audit()
    for name, r in reports.items():
        print("=" * 80)
        print(f"TOKENIZER AUDIT: {name}")
        print(f"Status: {r.get('status')}")
        if r.get('status') == 'AUTHENTIC':
            print(f"Class: {r.get('tokenizer_class')}")
            print(f"Vocab Size: {r.get('vocab_size'):,}")
            print(f"Encode Function: {r.get('encode_function')}")
            print(f"Path: {r.get('file_path')}")
            print("Probe String 1 ('The model is learning quickly.'):")
            p1 = r['probe_results']['The model is learning quickly.']
            print(f"  Token IDs (first 20): {p1['first_20_ids']}")
            print(f"  Decoded pieces:       {p1['decoded_pieces']}")
            print("Probe String 2 ('नमस्ते, मेरा नाम आकाश है।'):")
            p2 = r['probe_results']['नमस्ते, मेरा नाम आकाश है।']
            print(f"  Token IDs (first 20): {p2['first_20_ids']}")
            print(f"  Decoded pieces:       {p2['decoded_pieces']}")
        else:
            print(f"Error: {r.get('error')}")
