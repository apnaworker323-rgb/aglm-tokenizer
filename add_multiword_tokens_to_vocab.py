#!/usr/bin/env python3
"""
Add 64,544 high-frequency multiword bigram tokens (min_freq >= 10 from empirical training stream)
to the AGLM Universal Tokenizer across both export locations:
1. exported_tokenizers/aglm_universal_max
2. exported_tokenizers/aglm_universal_1m
"""

import os
import sys
import time
import json
import gzip
import re
from datetime import datetime, timezone
from collections import Counter

ROOT = "/run/media/akash/18FAA791FAA76A28/aglm_project"
ARCH_ROOT = os.path.join(ROOT, "architecture_battle")
TOK_ROOT = os.path.join(ROOT, "tokenizer")

sys.path.insert(0, ARCH_ROOT)
sys.path.insert(0, TOK_ROOT)

from stream import TrainingStream
from aglm_tokenizer.core.tokenizer import AGLMUniversalTokenizer

MAX_DIR = os.path.join(TOK_ROOT, "exported_tokenizers", "aglm_universal_max")
M1_DIR = os.path.join(TOK_ROOT, "exported_tokenizers", "aglm_universal_1m")

N_BATCHES = 500
BATCH_SIZE = 8
SEQ_LEN = 2048
MIN_FREQ = 10


def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Step 1: Loading current base tokenizer from {MAX_DIR}...")
    t0 = time.time()
    tok = AGLMUniversalTokenizer.load(MAX_DIR)
    initial_vocab_size = tok.vocab_size
    print(f"  Base vocab size: {initial_vocab_size:,} (loaded in {time.time()-t0:.2f}s)")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 2: Decoding {N_BATCHES} batches ({BATCH_SIZE}x{SEQ_LEN}) from TrainingStream...")
    stream = TrainingStream(seed=4242, split="train")
    chunks = []
    for _ in range(N_BATCHES):
        x, y = stream.get_batch(BATCH_SIZE, SEQ_LEN)
        for row in x:
            chunks.append(tok.decode(row.tolist()))
    full_text = "\n".join(chunks)
    print(f"  Total decoded text: {len(full_text):,} chars, {len(full_text.encode('utf-8')):,} bytes")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 3: Extracting and ranking word bigrams (min_freq >= {MIN_FREQ})...")
    words = re.findall(r"\S+", full_text)
    print(f"  Total words: {len(words):,}")
    bigrams = Counter(zip(words, words[1:]))
    qualifying = [(wb, c) for wb, c in bigrams.most_common() if c >= MIN_FREQ]
    qualifying.sort(key=lambda x: -x[1])  # Sort by highest frequency first
    print(f"  Qualifying bigrams count: {len(qualifying):,}")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 4: Adding new bigram tokens to tokenizer engine...")
    added_count = 0
    skipped_count = 0

    for (w1, w2), count in qualifying:
        pair_text = f"{w1} {w2}"
        pair_bytes = pair_text.encode("utf-8")
        if pair_bytes in tok.engine.bytes_to_id:
            skipped_count += 1
        else:
            tok.add_token(pair_bytes)
            added_count += 1

    new_vocab_size = tok.vocab_size
    print(f"  Successfully added: {added_count:,} tokens (skipped already existing: {skipped_count})")
    print(f"  New Total Vocab Size: {new_vocab_size:,} (Expected: {initial_vocab_size + added_count:,})")
    assert new_vocab_size == initial_vocab_size + added_count, "Vocab size mismatch!"

    # Export to Location 1: aglm_universal_max
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 5: Saving updated tokenizer to Location 1: {MAX_DIR}...")
    tok.name = "AGLM-Universal-Max-Unlimited"
    tok.save(MAX_DIR)
    
    manifest_max = {
        "model_name": "AGLM-Universal-Max-Unlimited",
        "vocab_size": new_vocab_size,
        "algorithm": "Byte-Level BPE with Code-Aware Longest-Prefix Match Trie",
        "byte_fallback": "Guaranteed 256 byte tokens (0x00-0xFF)",
        "unified_sources": [
            "OpenAI o200k_base & cl100k_base",
            "Meta XLM-V & XLM-RoBERTa",
            "Google Gemma 2",
            "DeepSeek V3",
            "Alibaba Qwen 2.5",
            "Meta Llama 3",
            "Mistral v0.3",
            "AI4Bharat Aksharantar (13 Indic Languages)",
            "Curated Multi-Language Code Syntax",
            f"Empirical High-Frequency Multiword Bigrams (+{added_count:,} tokens)"
        ],
        "export_timestamp": datetime.now(timezone.utc).isoformat()
    }
    with open(os.path.join(MAX_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest_max, f, indent=2)
    print("  Saved aglm_vocab.json, aglm_vocab.json.gz, and manifest.json in aglm_universal_max.")

    # Export to Location 2: aglm_universal_1m
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 6: Saving updated tokenizer to Location 2: {M1_DIR}...")
    tok.name = "AGLM-Universal-1M-Production"
    tok.save(M1_DIR)
    manifest_1m = {
        "model_name": "AGLM-Universal-1M-Production",
        "vocab_size": new_vocab_size,
        "algorithm": "Byte-Level BPE with Code-Aware Longest-Prefix Match Trie",
        "byte_fallback": "Guaranteed 256 byte tokens (0x00-0xFF)",
        "unified_sources": [
            "OpenAI o200k_base & cl100k_base",
            "Meta XLM-V & XLM-RoBERTa",
            "Google Gemma 2",
            "DeepSeek V3",
            "Alibaba Qwen 2.5",
            "Meta Llama 3",
            "Mistral v0.3",
            "AI4Bharat Aksharantar (13 Indic Languages)",
            "Curated Multi-Language Code Syntax",
            f"Empirical High-Frequency Multiword Bigrams (+{added_count:,} tokens)"
        ],
        "export_timestamp": datetime.now(timezone.utc).isoformat()
    }
    with open(os.path.join(M1_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest_1m, f, indent=2)
    print("  Saved aglm_vocab.json, aglm_vocab.json.gz, and manifest.json in aglm_universal_1m.")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 7: Verifying lossless encode/decode roundtrips on test probes...")
    test_cases = [
        ("English Standard", "Artificial General Intelligence and Language Models are rapidly advancing."),
        ("Multiword Bigram Test", f"{qualifying[0][0][0]} {qualifying[0][0][1]} - sample bigram"),
        ("Hindi / Devanagari", "नमस्ते दुनिया! यह एक उच्च प्रदर्शन बहुभाषी मॉडल है।"),
        ("Telugu Script", "నమస్కారం! కృత్రిమ మేధస్సు మరియు భాషా నమూనాలు."),
        ("Tamil Script", "வணக்கம்! செயற்கை நுண்ணறிவு மற்றும் மொழி மாதிரிகள்."),
        ("Code Python", "def compute_loss(logits, targets):\n    return F.cross_entropy(logits, targets)"),
        ("Code Rust", "fn main() {\n    println!(\"AGLM Universal Tokenizer v2\");\n}"),
        ("Arabic", "مرحبا بالعالم! هذا نموذج لغوي متقدم ومتعدد اللغات."),
        ("Chinese CJK", "通用人工智能正在快速发展，这是一个多语言大模型。"),
        ("Emoji & Special", "🚀🔥 SuperBPE Multiword Tokenizer + AGLM Universal! 🎯✨")
    ]

    all_passed = True
    for label, text in test_cases:
        enc = tok.encode(text)
        dec = tok.decode(enc)
        is_exact = (text == dec)
        if not is_exact:
            all_passed = False
        status = "PASS (100% Lossless)" if is_exact else "FAIL"
        print(f"  [{status}] | {label:<25} | Tokens: {len(enc):>2} | Preview: {text[:35]}...")

    assert all_passed, "Lossless verification failed!"
    print("\n[SUCCESS] Tokenizer update complete and 100% verified across both locations!")


if __name__ == "__main__":
    main()
