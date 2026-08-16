#!/usr/bin/env python3
"""
Indic-First Multilingual Vocabulary Builder for AGLM.
Rules:
1. 100% of ALL Indic Tokens (Hindi, Devanagari, Sanskrit, Marathi, Bengali, Tamil,
   Telugu, Gujarati, Kannada, Malayalam, Punjabi, Odia) are PRESERVED 100% INTACT (ZERO TOUCH).
2. 256 UTF-8 Byte Fallback Tokens (0x00-0xFF) & 9 Special Tokens are 100% PRESERVED.
3. Code Syntax, Python Indentation & Math Operators are 100% PRESERVED.
4. Active English Words & Multilingual Core Subwords (Arabic, European, Cyrillic, CJK) are included.
5. All Dead Scraper Bloat & Zero-Frequency Multi-Words in Foreign/English are pruned.
"""

import os
import sys
import json
import gzip
import numpy as np

sys.path.insert(0, "/run/media/akash/18FAA791FAA76A28/aglm_project/tokenizer")
from aglm_tokenizer.core.tokenizer import AGLMUniversalTokenizer

ROOT = "/run/media/akash/18FAA791FAA76A28/aglm_project"
SOURCE_VOCAB_GZ = os.path.join(ROOT, "tokenizer", "exported_tokenizers", "aglm_core_clean", "aglm_vocab.json.gz")
FREQ_FILE = os.path.join(ROOT, "aglm_tokenized_dataset", "metadata", "token_frequency.npy")
TARGET_DIR = os.path.join(ROOT, "tokenizer", "exported_tokenizers", "aglm_universal_max")
TARGET_DIR_500K = os.path.join(ROOT, "tokenizer", "exported_tokenizers", "aglm_universal_500k")
TARGET_DIR_1M = os.path.join(ROOT, "tokenizer", "exported_tokenizers", "aglm_universal_1m")

def is_indic(s):
    return any(
        ('\u0900' <= c <= '\u097f') or # Devanagari (Hindi, Marathi, Sanskrit, Nepali)
        ('\u0980' <= c <= '\u09ff') or # Bengali / Assamese
        ('\u0a00' <= c <= '\u0a7f') or # Gurmukhi (Punjabi)
        ('\u0a80' <= c <= '\u0aff') or # Gujarati
        ('\u0b00' <= c <= '\u0b7f') or # Oriya (Odia)
        ('\u0b80' <= c <= '\u0bff') or # Tamil
        ('\u0c00' <= c <= '\u0c7f') or # Telugu
        ('\u0c80' <= c <= '\u0cff') or # Kannada
        ('\u0d00' <= c <= '\u0d7f') or # Malayalam
        ('\u0d80' <= c <= '\u0dff') or # Sinhala
        ('\ua8e0' <= c <= '\ua8ff') or # Devanagari Extended
        ('\u1cd0' <= c <= '\u1cff')    # Vedic Extensions
        for c in s
    )

def is_cjk(s):
    return any(('\u4e00' <= c <= '\u9fff') or ('\u3400' <= c <= '\u4dbf') or ('\u3040' <= c <= '\u30ff') for c in s)

def is_cyrillic(s):
    return any(('\u0400' <= c <= '\u04ff') or ('\u0500' <= c <= '\u052f') for c in s)

def is_arabic(s):
    return any(('\u0600' <= c <= '\u06ff') or ('\u0750' <= c <= '\u077f') or ('\u08a0' <= c <= '\u08ff') or ('\ufb50' <= c <= '\ufdff') or ('\ufe70' <= c <= '\ufeff') for c in s)

def is_european_diacritic(s):
    return any(
        ('\u00c0' <= c <= '\u00ff') or ('\u0100' <= c <= '\u017f') or
        ('\u0180' <= c <= '\u024f') or ('\u1e00' <= c <= '\u1eff')
        for c in s
    )

def is_code_syntax(s):
    return all(c in ' \t\n\r{}[]()<>=+-*/%&|^~!?:;.,\'"`@#$\\_' or c.isdigit() for c in s)

def main():
    print("=" * 75)
    print("BUILDING INDIC-FIRST MULTILINGUAL OPTIMIZED VOCABULARY")
    print("=" * 75)

    print(f"Loading master vocabulary: {SOURCE_VOCAB_GZ}...")
    with gzip.open(SOURCE_VOCAB_GZ, "rt", encoding="utf-8") as f:
        data = json.load(f)

    tokens = data["tokens"]
    print(f"Loaded {len(tokens):,} raw tokens.")

    if os.path.exists(FREQ_FILE):
        freqs = np.load(FREQ_FILE)
        print(f"Loaded corpus frequencies ({len(freqs):,} entries).")
    else:
        freqs = np.zeros(len(tokens), dtype=np.int64)

    seen_bytes = set()
    for i in range(256):
        seen_bytes.add(bytes([i]))

    indic_tokens = []
    code_syntax_tokens = []
    arabic_tokens = []
    european_tokens = []
    english_active_tokens = []
    other_tokens = []

    for idx, t in enumerate(tokens):
        tid = t["id"]
        if tid < 256:
            continue
        raw_hex = t["bytes_hex"]
        token_bytes = bytes.fromhex(raw_hex)
        s = t.get("str", "")
        f = int(freqs[idx]) if idx < len(freqs) else 0
        item = (token_bytes, s, raw_hex, f, tid)

        # 1. INDIC: Keep EVERY SINGLE ONE (Zero Pruning!)
        if is_indic(s):
            indic_tokens.append(item)
        elif is_code_syntax(s):
            code_syntax_tokens.append(item)
        elif is_arabic(s):
            if f > 0 or len(token_bytes) <= 4:
                arabic_tokens.append(item)
        elif is_european_diacritic(s):
            if f > 0 or len(token_bytes) <= 4:
                european_tokens.append(item)
        elif f > 0:
            english_active_tokens.append(item)
        else:
            if len(token_bytes) <= 4: # Keep short primitives
                other_tokens.append(item)

    print("\nCategorized Tokens:")
    print(f"• Indic (Hindi, Sanskrit, Tamil, Bengali, Telugu, etc.): {len(indic_tokens):>8,} (100% PRESERVED)")
    print(f"• Code Syntax & Math Operators:                          {len(code_syntax_tokens):>8,} (100% PRESERVED)")
    print(f"• Active English & Active Corpus Tokens:                 {len(english_active_tokens):>8,}")
    print(f"• Arabic / Persian Core Tokens:                          {len(arabic_tokens):>8,}")
    print(f"• European Diacritics Core Tokens:                       {len(european_tokens):>8,}")
    print(f"• Short Multilingual Character Primitives:               {len(other_tokens):>8,}")

    final_normal_tokens = []

    # Add 100% of Indic tokens first
    for item in indic_tokens:
        if item[0] not in seen_bytes:
            seen_bytes.add(item[0])
            final_normal_tokens.append(item)

    # Add 100% of Code Syntax tokens
    for item in code_syntax_tokens:
        if item[0] not in seen_bytes:
            seen_bytes.add(item[0])
            final_normal_tokens.append(item)

    # Add Active English and Corpus tokens
    for item in english_active_tokens:
        if item[0] not in seen_bytes:
            seen_bytes.add(item[0])
            final_normal_tokens.append(item)

    # Add Arabic, European, and short multilingual primitives
    for bucket in [arabic_tokens, european_tokens, other_tokens]:
        for item in bucket:
            if item[0] not in seen_bytes:
                seen_bytes.add(item[0])
                final_normal_tokens.append(item)

    print(f"\nTotal Selected Normal Tokens: {len(final_normal_tokens):,}")

    # Build AGLMUniversalTokenizer
    tok = AGLMUniversalTokenizer(
        name="AGLM-Universal-IndicFirst-Core",
        strategy="Indic_100Percent_Preserved_Clean"
    )

    next_id = 265
    for item in final_normal_tokens:
        tok.engine.add_token(item[0], token_id=next_id)
        next_id += 1

    print(f"Final Tokenizer Vocab Size: {tok.vocab_size:,}")
    print(f"Special Tokens Count:       {len(tok.engine.special_tokens)}")
    print(f"Max Token ID:               {max(set(tok.engine.id_to_bytes.keys()) | set(tok.engine.special_tokens.values())):,}")

    # Multilingual Lossless Verification
    test_samples = {
        "Hindi": "नमस्ते दुनिया! भारत का अपना सबसे शक्तिशाली और तेज़ AGLM भाषा मॉडल।",
        "Sanskrit": "विद्या ददाति विनयं विनयाद्याति पात्रताम्।",
        "Tamil": "வணக்கம் உலகம்! இது ஒரு சக்திவாய்ந்த மொழி மாதிரி.",
        "Telugu": "నమస్కారం ప్రపంచం! ఇది అత్యంత శక్తివంతమైన భాషా మోడల్.",
        "Bengali": "নমস্কার বিশ্ব! এটি একটি শক্তিশালী ভাষা মডেল।",
        "English": "The quick brown fox jumps over the lazy dog. Building clean neural architectures.",
        "Code": "import torch\nclass Attention(nn.Module):\n    def __init__(self, dim=384): pass",
        "Arabic": "مرحبا بالعالم! هذا هو نموذج الذكاء الاصطناعي متعدد اللغات.",
        "French": "Bonjour le monde! Modèle d'intelligence artificielle performant."
    }

    print("\n=== MULTILINGUAL ENCODE/DECODE VERIFICATION ===")
    for lang, text in test_samples.items():
        encoded = tok.encode(text)
        decoded = tok.decode(encoded)
        match = (text == decoded)
        print(f"• {lang:<10}: {len(encoded):>3} tokens | Decoded Match: {'PASS (100% Exact)' if match else 'FAIL'}")
        if not match:
            raise ValueError(f"Roundtrip failed for {lang}!")

    # Save to export directories
    manifest = {
        "model_name": "AGLM-Universal-IndicFirst-Core",
        "vocab_size": tok.vocab_size,
        "algorithm": "Byte-Level BPE with Code-Aware Longest-Prefix Match Trie",
        "byte_fallback": "Guaranteed 256 byte tokens (0x00-0xFF)",
        "indic_preservation": "100% of Devanagari, Sanskrit, Tamil, Telugu, Bengali, Gujarati, Kannada, Malayalam, Punjabi, Odia preserved",
        "export_timestamp": "2026-08-16T21:46:00Z"
    }

    for d in [TARGET_DIR, TARGET_DIR_500K, TARGET_DIR_1M]:
        os.makedirs(d, exist_ok=True)
        print(f"Saving to {d}...")
        tok.save(d)
        with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    print("\n[SUCCESS] Indic-First Vocab successfully exported to all targets!")

if __name__ == "__main__":
    main()
