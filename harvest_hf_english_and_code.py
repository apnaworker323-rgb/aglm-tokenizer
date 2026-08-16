#!/usr/bin/env python3
"""
Hugging Face English & GitHub Code Superwords Harvester & Integrator.
Streams authentic GitHub code and English datasets from Hugging Face Hub:
1. sahil2801/CodeAlpaca-20k (Multi-language coding instructions & solutions)
2. flytech/python-codes-25k (Real Python source code, algorithms, & frameworks)
3. roneneldan/TinyStories (Pure high-quality English discourse & narrative text)

Extracts high-utility 2-gram to 5-gram superwords and integrates them into AGLM tokenizers.
Exports to:
- exported_tokenizers/aglm_universal_max
- exported_tokenizers/aglm_universal_1m
"""

import os
import sys
import time
import json
import gzip
import re
from datetime import datetime, timezone
from collections import Counter
from typing import List, Set, Tuple

ROOT = "/run/media/akash/18FAA791FAA76A28/aglm_project"
ARCH_ROOT = os.path.join(ROOT, "architecture_battle")
TOK_ROOT = os.path.join(ROOT, "tokenizer")

sys.path.insert(0, ARCH_ROOT)
sys.path.insert(0, TOK_ROOT)

from datasets import load_dataset
from aglm_tokenizer.core.tokenizer import AGLMUniversalTokenizer

MAX_DIR = os.path.join(TOK_ROOT, "exported_tokenizers", "aglm_universal_max")
M1_DIR = os.path.join(TOK_ROOT, "exported_tokenizers", "aglm_universal_1m")

MAX_N = 5
MIN_FREQ = 3


def stream_hf_code_and_english(max_samples_per_dataset: int = 5000) -> str:
    """Streams GitHub coding data and English text directly from Hugging Face."""
    all_text = []
    
    # 1. CodeAlpaca (Coding tasks, Python, C++, JS, Algorithms)
    try:
        print("  [HF Stream 1/3] Streaming CodeAlpaca (sahil2801/CodeAlpaca-20k)...")
        ds_code = load_dataset("sahil2801/CodeAlpaca-20k", split="train", streaming=True)
        count = 0
        for item in ds_code:
            text = (item.get("instruction", "") + "\n" + item.get("input", "") + "\n" + item.get("output", "")).strip()
            if text:
                all_text.append(text)
                count += 1
            if count >= max_samples_per_dataset:
                break
        print(f"    Streamed {count:,} GitHub code samples.")
    except Exception as e:
        print(f"    Warning: CodeAlpaca streaming error: {e}")

    # 2. Python Codes (flytech/python-codes-25k)
    try:
        print("  [HF Stream 2/3] Streaming Python Codes (flytech/python-codes-25k)...")
        ds_py = load_dataset("flytech/python-codes-25k", split="train", streaming=True)
        count = 0
        for item in ds_py:
            text = (item.get("instruction", "") + "\n" + item.get("input", "") + "\n" + item.get("output", "")).strip()
            if text:
                all_text.append(text)
                count += 1
            if count >= max_samples_per_dataset:
                break
        print(f"    Streamed {count:,} Python source samples.")
    except Exception as e:
        print(f"    Warning: Python codes streaming error: {e}")

    # 3. English Stories & Knowledge (roneneldan/TinyStories)
    try:
        print("  [HF Stream 3/3] Streaming English Knowledge (roneneldan/TinyStories)...")
        ds_eng = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        count = 0
        for item in ds_eng:
            text = item.get("text", "").strip()
            if text:
                all_text.append(text)
                count += 1
            if count >= max_samples_per_dataset:
                break
        print(f"    Streamed {count:,} English narrative samples.")
    except Exception as e:
        print(f"    Warning: TinyStories streaming error: {e}")

    combined = "\n".join(all_text)
    return combined


def main():
    print("=" * 85)
    print("🚀 HUGGING FACE GITHUB CODE & ENGLISH SUPERWORD HARVESTER")
    print("=" * 85)
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Step 1: Loading current tokenizer from {MAX_DIR}...")
    t0 = time.time()
    tok = AGLMUniversalTokenizer.load(MAX_DIR)
    initial_vocab_size = tok.vocab_size
    print(f"  Base Vocab Size: {initial_vocab_size:,} (loaded in {time.time()-t0:.2f}s)\n")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Step 2: Streaming Code & English datasets from Hugging Face...")
    t0 = time.time()
    streamed_text = stream_hf_code_and_english(max_samples_per_dataset=5000)
    print(f"  Total Streamed Text: {len(streamed_text):,} chars, {len(streamed_text.encode('utf-8')):,} bytes ({time.time()-t0:.2f}s)\n")

    words = re.findall(r"\S+", streamed_text)
    n_words = len(words)
    print(f"  Total Words in Streamed Dataset: {n_words:,}\n")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Step 3: Extracting 2-gram to 5-gram Coding & English N-grams (min_freq >= {MIN_FREQ})...")
    t0 = time.time()
    candidates = []
    for n in range(2, MAX_N + 1):
        counter = Counter()
        for i in range(n_words - n + 1):
            ngram = tuple(words[i:i + n])
            counter[ngram] += 1
        
        n_qualifying = 0
        for ngram, count in counter.items():
            if count >= MIN_FREQ:
                phrase = " ".join(ngram)
                if 4 <= len(phrase) <= 120:
                    candidates.append((phrase, count, n))
                    n_qualifying += 1
        print(f"  {n}-grams qualifying (>= {MIN_FREQ}): {n_qualifying:,}")

    # Priority sort: highest frequency first, longer n-gram breaks ties
    candidates.sort(key=lambda x: (-x[1], -x[2]))
    print(f"  Total candidates to evaluate: {len(candidates):,} (harvested in {time.time()-t0:.2f}s)\n")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Step 4: Adding deduplicated GitHub Code & English superwords to tokenizer...")
    added_count = 0
    skipped_count = 0

    for phrase, count, n in candidates:
        phrase_bytes = phrase.encode("utf-8")
        if phrase_bytes in tok.engine.bytes_to_id:
            skipped_count += 1
        else:
            tok.add_token(phrase_bytes)
            added_count += 1

    new_vocab_size = tok.vocab_size
    print(f"  Successfully Added: {added_count:,} new code/English superword tokens")
    print(f"  Skipped (Already Present): {skipped_count:,}")
    print(f"  New Total Vocab Size: {new_vocab_size:,} (Expected: {initial_vocab_size + added_count:,})")
    assert new_vocab_size == initial_vocab_size + added_count, "Vocab size mismatch!"

    # Export to Location 1: aglm_universal_max
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 5: Exporting to Location 1: {MAX_DIR}...")
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
            "Empirical High-Frequency Word Bigrams (+64,544 tokens)",
            "Universal 2-to-5 Gram Multiword Superwords (+24,341 tokens)",
            "High-Leverage Reasoning, Code, LaTeX & Indic Superwords (+48,789 tokens)",
            f"Hugging Face GitHub Code & English Knowledge Superwords (+{added_count:,} tokens)"
        ],
        "export_timestamp": datetime.now(timezone.utc).isoformat()
    }
    with open(os.path.join(MAX_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest_max, f, indent=2)
    print("  Saved aglm_vocab.json, aglm_vocab.json.gz, and manifest.json in aglm_universal_max.")

    # Export to Location 2: aglm_universal_1m
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 6: Exporting to Location 2: {M1_DIR}...")
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
            "Empirical High-Frequency Word Bigrams (+64,544 tokens)",
            "Universal 2-to-5 Gram Multiword Superwords (+24,341 tokens)",
            "High-Leverage Reasoning, Code, LaTeX & Indic Superwords (+48,789 tokens)",
            f"Hugging Face GitHub Code & English Knowledge Superwords (+{added_count:,} tokens)"
        ],
        "export_timestamp": datetime.now(timezone.utc).isoformat()
    }
    with open(os.path.join(M1_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest_1m, f, indent=2)
    print("  Saved aglm_vocab.json, aglm_vocab.json.gz, and manifest.json in aglm_universal_1m.")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 7: Verifying Lossless Roundtrip on GitHub Code & English Probes...")
    test_cases = [
        ("GitHub Python Alg", "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1"),
        ("GitHub PyTorch", "class NeuralNetwork(nn.Module):\n    def __init__(self):"),
        ("GitHub JS React", "const handleSubmit = (event) => {\n    event.preventDefault();"),
        ("English Narrative", "Once upon a time, there was a little girl who loved to read books."),
        ("Reasoning Prompt", "Write a Python function to find the longest palindromic substring."),
        ("Hindi Devanagari", "भारत सरकार द्वारा शुरू की गई इस नई योजना का मुख्य उद्देश्य है।"),
        ("Hinglish Chat", "bhai is code ko optimize karne ke baad speed kitni badhegi?"),
        ("Emoji & Special", "🚀🔥 GitHub Code + English Superwords + AGLM Universal! 🎯✨")
    ]

    all_passed = True
    for label, text in test_cases:
        enc = tok.encode(text)
        dec = tok.decode(enc)
        is_exact = (text == dec)
        if not is_exact:
            all_passed = False
        status = "PASS (100% Lossless)" if is_exact else "FAIL"
        print(f"  [{status}] | {label:<22} | Tokens: {len(enc):>2} | Preview: {text[:38]}...")

    assert all_passed, "Lossless verification failed!"
    print(f"\n[SUCCESS] Hugging Face Code & English Superwords integrated! Added {added_count:,} tokens. Final Vocab: {new_vocab_size:,}")


if __name__ == "__main__":
    main()
