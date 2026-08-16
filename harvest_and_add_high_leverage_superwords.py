#!/usr/bin/env python3
"""
High-Leverage Multi-Domain Superwords Harvester & Integrator.
Targets:
1. LLM Reasoning, Chain-of-Thought, & Prompt Template Collocations
2. Code Idioms (Python, PyTorch, React, JS/TS, SQL, Rust, C++, JSON schemas)
3. LaTeX Mathematical & Scientific Formulae
4. Conversational Hindi, Hinglish, & Dravidian Multiword Discourse Markers
5. High-Frequency Web & Dialogue Corpus N-grams

Exports updated models to:
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
from typing import List, Set, Dict

ROOT = "/run/media/akash/18FAA791FAA76A28/aglm_project"
ARCH_ROOT = os.path.join(ROOT, "architecture_battle")
TOK_ROOT = os.path.join(ROOT, "tokenizer")
DATA_ROOT = os.path.join(ROOT, "data")

sys.path.insert(0, ARCH_ROOT)
sys.path.insert(0, TOK_ROOT)

from aglm_tokenizer.core.tokenizer import AGLMUniversalTokenizer

MAX_DIR = os.path.join(TOK_ROOT, "exported_tokenizers", "aglm_universal_max")
M1_DIR = os.path.join(TOK_ROOT, "exported_tokenizers", "aglm_universal_1m")


def generate_domain_targeted_superwords() -> List[str]:
    """Generates a curated list of high-leverage multiword patterns across the 4 key domains."""
    targeted_phrases = set()

    # =========================================================================
    # Domain 1: Reasoning, Chain-of-Thought & Prompt Patterns
    # =========================================================================
    reasoning_templates = [
        "Let's think step by step",
        "Here is the step-by-step solution:",
        "To solve this problem, we need to",
        "Based on the provided information,",
        "It is important to note that",
        "As shown in the following example:",
        "In conclusion, we can observe that",
        "According to the context provided,",
        "First, let's analyze the problem:",
        "Next, we can compute the value as follows:",
        "Therefore, the final answer is:",
        "Let's break down the solution into steps:",
        "In order to achieve optimal performance,",
        "For example, consider the case where",
        "On the other hand, if we consider",
        "As a result of this operation,",
        "From the given equation, we obtain:",
        "This means that for every element,",
        "Let us verify the correctness of the result:",
        "By substituting the values into the formula,",
        "The key intuition behind this approach is",
        "Specifically, in this scenario,",
        "Without loss of generality, we can assume",
        "It follows immediately from the definition that",
        "In this section, we provide a detailed overview of"
    ]
    targeted_phrases.update(reasoning_templates)

    # =========================================================================
    # Domain 2: Code, Web & Data Formats
    # =========================================================================
    code_templates = [
        # Python / PyTorch / Data Science
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "import torch.nn as nn",
        "import torch.nn.functional as F",
        "from typing import List, Dict, Tuple, Optional, Any, Union",
        "from collections import defaultdict, Counter",
        "from dataclasses import dataclass",
        "if __name__ == '__main__':",
        "def __init__(self, *args, **kwargs):",
        "def forward(self, x: torch.Tensor) -> torch.Tensor:",
        "super().__init__()",
        "raise NotImplementedError('Subclasses must implement this method')",
        "with torch.no_grad():",
        "loss.backward()",
        "optimizer.step()",
        "optimizer.zero_grad()",
        "model.eval()",
        "model.train()",
        "@torch.no_grad()",
        "@classmethod",
        "@staticmethod",
        "@property",
        
        # JS / TS / React / Web
        "import React, { useState, useEffect, useMemo, useCallback } from 'react';",
        "export default function App() {",
        "const [loading, setLoading] = useState(false);",
        "const [error, setError] = useState<string | null>(null);",
        "const [data, setData] = useState(null);",
        "useEffect(() => {\n    const fetchData = async () => {\n",
        "document.addEventListener('DOMContentLoaded', () => {",
        "document.getElementById(",
        "document.querySelector(",
        "export const ",
        "export interface ",
        "export type ",
        "console.log('Debug:', ",
        "console.error('Error occurred:', ",
        "async (req: Request, res: Response) => {",
        "return res.status(200).json({",
        "return res.status(500).json({ error:",
        
        # SQL / Databases
        "SELECT * FROM users WHERE",
        "SELECT COUNT(*) FROM",
        "CREATE TABLE IF NOT EXISTS",
        "PRIMARY KEY AUTOINCREMENT",
        "GROUP BY id ORDER BY created_at DESC",
        "INSERT INTO table_name (column1, column2) VALUES",
        "ON CONFLICT (id) DO UPDATE SET",
        "INNER JOIN ON",
        "LEFT OUTER JOIN ON",
        
        # Rust / C++ / Java
        "fn main() -> Result<(), Box<dyn std::error::Error>> {",
        "pub fn new() -> Self {",
        "impl<T> Default for",
        "#include <iostream>",
        "#include <vector>",
        "#include <string>",
        "#include <memory>",
        "using namespace std;",
        "public static void main(String[] args) {",
        "public class Main {",
        "System.out.println(",
        
        # JSON / REST / Schema
        "{\n  \"status\": \"success\",\n  \"data\":",
        "{\n  \"type\": \"object\",\n  \"properties\":",
        "\"Content-Type\": \"application/json\"",
        "\"Authorization\": \"Bearer \"",
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>",
        "<meta charset=\"UTF-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">"
    ]
    targeted_phrases.update(code_templates)

    # =========================================================================
    # Domain 3: LaTeX, Mathematics & Science
    # =========================================================================
    latex_templates = [
        "\\begin{equation}\n",
        "\\end{equation}\n",
        "\\begin{align*}\n",
        "\\end{align*}\n",
        "\\begin{pmatrix}\n",
        "\\end{pmatrix}\n",
        "\\frac{\\partial f}{\\partial x}",
        "\\frac{d}{dx}",
        "\\sum_{i=1}^{n}",
        "\\prod_{i=1}^{n}",
        "\\int_{-\\infty}^{\\infty}",
        "\\mathbb{R}^{n \\times d}",
        "\\mathbf{W}^T \\mathbf{x} + \\mathbf{b}",
        "\\mathcal{L}(\\theta) =",
        "\\arg\\max_{\\theta}",
        "\\arg\\min_{\\theta}",
        "\\nabla_{\\theta} J(\\theta)",
        "\\mathcal{O}(n \\log n)",
        "\\mathcal{O}(1)",
        "\\mathcal{O}(n^2)",
        "\\text{softmax}\\left(",
        "\\text{ReLU}\\left(",
        "\\text{CrossEntropyLoss}("
    ]
    targeted_phrases.update(latex_templates)

    # =========================================================================
    # Domain 4: Conversational Hindi, Hinglish & Dravidian Multiword Collocations
    # =========================================================================
    indic_conversational = [
        # Hindi Devanagari Collocations
        "भारत सरकार द्वारा",
        "के माध्यम से",
        "हो सकता है कि",
        "किया जा सकता है",
        "करने के लिए",
        "के रूप में",
        "कहा जा सकता है कि",
        "इस बात का ध्यान रखें",
        "प्राप्त करने के लिए",
        "के संदर्भ में",
        "के अनुसार",
        "होने के बाद",
        "इस प्रकार के",
        "महत्वपूर्ण भूमिका निभाता है",
        "विकास के लिए",
        "सहायता प्रदान करने",
        "जानकारी के अनुसार",
        "विशेष रूप से",
        "समय के साथ",
        "सुविधाएं उपलब्ध कराई",
        
        # Hinglish High-Frequency Dialogues
        "kya aap mujhe bata sakte ho",
        "mujhe lagta hai ki yeh",
        "aapka bahut bahut dhanyawad",
        "is baare mein kya kehna hai",
        "kripya karke mujhe bataiye",
        "agar aapko koi problem ho toh",
        "theek hai main check karta hoon",
        "kuch samajh nahi aa raha hai",
        "chalo shuru karte hain",
        "batao yaar kya scene hai",
        "kaise ho bhai sab theek hai na",
        "isse speed kaafi badh jayegi",
        "aaj ka plan kya hai",
        "apna khayal rakhna",
        "jaldi se batao kya hua",
        "koi baat nahi tension mat lo",
        "yeh cheez bohot sahi hai",
        "thoda time lag sakta hai",
        "ek baar dekh ke batana",
        "poora code share kar do please",
        
        # Dravidian Romanized Collocations
        "nenu meeku cheppali anukuntunnanu",
        "chala bagundi andi",
        "ee vishayam gurinchi",
        "indha vishayam romba mukkiyam",
        "nalla irukku bro",
        "seekiram pannanum appo dhaan",
        "oru puthiya project",
        "ellam shariyavum tension venda",
        "nanage thumba ishta aayithu"
    ]
    targeted_phrases.update(indic_conversational)

    return list(targeted_phrases)


def harvest_corpus_deep_ngrams(min_freq: int = 4, max_lines: int = 50000) -> List[Tuple[str, int]]:
    """Deep-mines high-frequency 2-to-5 word combinations from FineWeb and LMSYS text datasets."""
    print("  [Deep Mining] Scanning FineWeb & LMSYS datasets for additional recurring multiwords...")
    counter = Counter()
    
    files_to_scan = [
        os.path.join(DATA_ROOT, "fineweb_combined_train_95.txt"),
        os.path.join(DATA_ROOT, "lmsys_train_95.txt")
    ]
    
    for fpath in files_to_scan:
        if not os.path.exists(fpath):
            continue
        print(f"    Scanning {os.path.basename(fpath)} (up to {max_lines:,} lines)...")
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                line_str = line.strip()
                if not line_str or len(line_str) < 15:
                    continue
                words = line_str.split()
                n_w = len(words)
                # Harvest 2..5 grams
                for n in range(2, 6):
                    for j in range(n_w - n + 1):
                        ngram_str = " ".join(words[j:j + n])
                        counter[ngram_str] += 1
                if i >= max_lines:
                    break

    qualifying = [(phrase, count) for phrase, count in counter.items() if count >= min_freq and 4 <= len(phrase) <= 120]
    qualifying.sort(key=lambda x: -x[1])
    print(f"    Deep-mined {len(qualifying):,} qualifying corpus multiwords (freq >= {min_freq})")
    return qualifying


def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Step 1: Loading current tokenizer from {MAX_DIR}...")
    t0 = time.time()
    tok = AGLMUniversalTokenizer.load(MAX_DIR)
    initial_vocab_size = tok.vocab_size
    print(f"  Base Vocab Size: {initial_vocab_size:,} (loaded in {time.time()-t0:.2f}s)")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 2: Generating Domain-Targeted Superwords (Reasoning + Code + LaTeX + Indic)...")
    targeted_list = generate_domain_targeted_superwords()
    print(f"  Curated targeted domain superwords: {len(targeted_list):,}")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 3: Deep-Mining Multi-Domain Corpus N-grams...")
    corpus_ngrams = harvest_corpus_deep_ngrams(min_freq=4, max_lines=40000)

    # Combine all candidates
    all_candidates = set(targeted_list)
    for phrase, count in corpus_ngrams:
        all_candidates.add(phrase)

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 4: Adding deduplicated high-leverage superwords to tokenizer...")
    added_count = 0
    skipped_count = 0

    for phrase in all_candidates:
        phrase_bytes = phrase.encode("utf-8")
        if phrase_bytes in tok.engine.bytes_to_id:
            skipped_count += 1
        else:
            tok.add_token(phrase_bytes)
            added_count += 1

    new_vocab_size = tok.vocab_size
    print(f"  Successfully Added: {added_count:,} new high-leverage superwords")
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
            f"High-Leverage Reasoning, Code, LaTeX & Indic Superwords (+{added_count:,} tokens)"
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
            f"High-Leverage Reasoning, Code, LaTeX & Indic Superwords (+{added_count:,} tokens)"
        ],
        "export_timestamp": datetime.now(timezone.utc).isoformat()
    }
    with open(os.path.join(M1_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest_1m, f, indent=2)
    print("  Saved aglm_vocab.json, aglm_vocab.json.gz, and manifest.json in aglm_universal_1m.")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 7: Testing high-leverage probes for single-token compression...")
    test_cases = [
        ("Reasoning CoT", "Let's think step by step"),
        ("Problem Solving", "To solve this problem, we need to"),
        ("Code Python", "import matplotlib.pyplot as plt"),
        ("Code Typing", "from typing import List, Dict, Tuple, Optional, Any, Union"),
        ("Code React", "const [loading, setLoading] = useState(false);"),
        ("LaTeX Formula", "\\frac{\\partial f}{\\partial x}"),
        ("LaTeX Sum", "\\sum_{i=1}^{n}"),
        ("Hinglish Chat", "kya aap mujhe bata sakte ho"),
        ("Hindi Devanagari", "भारत सरकार द्वारा शुरू की गई"),
        ("Telugu Script", "నమస్కారం! కృత్రిమ మేధస్సు మరియు భాషా నమూనాలు."),
        ("Tamil Script", "வணக்கம்! செயற்கை நுண்ணறிவு மற்றும் மொழி மாதிரிகள்."),
        ("Emoji & Special", "🚀🔥 SuperBPE Multiword Superwords + AGLM Universal! 🎯✨")
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
    print(f"\n[SUCCESS] High-leverage superwords integrated! Added {added_count:,} tokens. Final Vocab: {new_vocab_size:,}")


if __name__ == "__main__":
    main()
