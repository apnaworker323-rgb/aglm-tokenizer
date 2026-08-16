#!/usr/bin/env python3
"""
Grammar Books & Multilingual Syntax Superwords Harvester & Integrator.
Harvests high-frequency 2-gram to 5-gram grammatical structures, syntax rules,
and syntactic collocations from:
1. data/english_grammar_books.txt (Baskervill, Kirkham, Stewart, Armstrong)
2. data/multilingual_grammar_rules.txt (Multilingual GEC & grammar pairs)
3. data/master_grammar_corpus.txt

Integrates unique superwords into AGLM Universal Tokenizer and exports to both:
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
DATA_ROOT = os.path.join(ROOT, "data")

sys.path.insert(0, ARCH_ROOT)
sys.path.insert(0, TOK_ROOT)

from aglm_tokenizer.core.tokenizer import AGLMUniversalTokenizer

MAX_DIR = os.path.join(TOK_ROOT, "exported_tokenizers", "aglm_universal_max")
M1_DIR = os.path.join(TOK_ROOT, "exported_tokenizers", "aglm_universal_1m")

MAX_N = 5
MIN_FREQ = 3


def load_grammar_corpus() -> str:
    path = os.path.join(DATA_ROOT, "master_grammar_corpus.txt")
    if not os.path.exists(path):
        # Fallback to individual files
        files = [
            os.path.join(DATA_ROOT, "english_grammar_books.txt"),
            os.path.join(DATA_ROOT, "multilingual_grammar_rules.txt")
        ]
        text_chunks = []
        for f in files:
            if os.path.exists(f):
                with open(f, "r", encoding="utf-8", errors="ignore") as fp:
                    text_chunks.append(fp.read())
        return "\n".join(text_chunks)
        
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def main():
    print("=" * 85)
    print("📖 GRAMMAR BOOKS & MULTILINGUAL SYNTAX SUPERWORD HARVESTER")
    print("=" * 85)
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Step 1: Loading current tokenizer from {MAX_DIR}...")
    t0 = time.time()
    tok = AGLMUniversalTokenizer.load(MAX_DIR)
    initial_vocab_size = tok.vocab_size
    print(f"  Base Vocab Size: {initial_vocab_size:,} (loaded in {time.time()-t0:.2f}s)\n")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Step 2: Loading Grammar Corpus...")
    t0 = time.time()
    grammar_text = load_grammar_corpus()
    print(f"  Loaded Grammar Corpus: {len(grammar_text):,} chars, {len(grammar_text.encode('utf-8')):,} bytes ({time.time()-t0:.2f}s)\n")

    words = re.findall(r"\S+", grammar_text)
    n_words = len(words)
    print(f"  Total Words in Grammar Corpus: {n_words:,}\n")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Step 3: Extracting 2-gram to 5-gram Grammatical N-grams (min_freq >= {MIN_FREQ})...")
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
    print(f"  Total grammatical candidates to evaluate: {len(candidates):,} (harvested in {time.time()-t0:.2f}s)\n")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Step 4: Adding deduplicated Grammar superwords to tokenizer...")
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
    print(f"  Successfully Added: {added_count:,} new grammatical superword tokens")
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
            "Hugging Face GitHub Code & English Knowledge Superwords (+159,404 tokens)",
            f"English Grammar Books & Multilingual Syntax Superwords (+{added_count:,} tokens)"
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
            "Hugging Face GitHub Code & English Knowledge Superwords (+159,404 tokens)",
            f"English Grammar Books & Multilingual Syntax Superwords (+{added_count:,} tokens)"
        ],
        "export_timestamp": datetime.now(timezone.utc).isoformat()
    }
    with open(os.path.join(M1_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest_1m, f, indent=2)
    print("  Saved aglm_vocab.json, aglm_vocab.json.gz, and manifest.json in aglm_universal_1m.")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 7: Verifying Lossless Roundtrip across Grammar Probes...")
    test_cases = [
        ("Grammar Rule 1", "A noun is the name of any person, place, or thing."),
        ("Grammar Rule 2", "The subject of a finite verb is in the nominative case."),
        ("Syntax Agreement", "Transitive verbs govern the objective case in a sentence."),
        ("Multilingual GEC", "Language: Spanish\nIncorrect: Yo saber la verdad\nCorrect Grammar: Yo sé la verdad"),
        ("Hindi Devanagari", "संज्ञा किसी व्यक्ति, वस्तु, स्थान या भाव के नाम को कहते हैं।"),
        ("Hinglish Chat", "bhai grammar rules aur syntax structure dono exact lossless hain."),
        ("GitHub PyTorch", "class TransformerBlock(nn.Module):\n    def __init__(self, d_model):"),
        ("Emoji & Special", "📚✨ Grammar Books + Multilingual Syntax + AGLM Universal! 🚀🔥")
    ]

    all_passed = True
    for label, text in test_cases:
        enc = tok.encode(text)
        dec = tok.decode(enc)
        is_exact = (text == dec)
        if not is_exact:
            all_passed = False
        status = "PASS (100% Lossless)" if is_exact else "FAIL"
        print(f"  [{status}] | {label:<20} | Tokens: {len(enc):>2} | Preview: {text[:38]}...")

    assert all_passed, "Lossless verification failed!"
    print(f"\n[SUCCESS] Grammar Books & Syntax Superwords integrated! Added {added_count:,} tokens. Final Vocab: {new_vocab_size:,}")


if __name__ == "__main__":
    main()
