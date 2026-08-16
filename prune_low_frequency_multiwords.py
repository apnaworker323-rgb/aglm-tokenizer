#!/usr/bin/env python3
"""
Prune ONLY low-frequency Multi-Word (Superword/Compound) tokens (frequency < 4).
Retains 100% of single tokens, subwords, characters, byte tokens, and special tokens.
Also removes Chinese & Russian tokens as requested.
"""

import os
import sys
import gzip
import json
import time
import hashlib
import numpy as np
from pathlib import Path

ROOT = Path("/run/media/akash/18FAA791FAA76A28/aglm_project")
SOURCE_DIR = ROOT / "tokenizer" / "exported_tokenizers" / "aglm_universal_max"
TARGET_DIR = ROOT / "tokenizer" / "exported_tokenizers" / "aglm_pruned_multiword_clean"
FREQ_FILE = ROOT / "aglm_tokenized_dataset" / "metadata" / "token_frequency.npy"

def is_cjk(s: str) -> bool:
    return any(
        '\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf' or
        '\u20000' <= c <= '\u2a6df' or '\u3040' <= c <= '\u30ff' or
        '\uac00' <= c <= '\ud7af' for c in s
    )

def is_cyrillic(s: str) -> bool:
    return any('\u0400' <= c <= '\u04ff' or '\u0500' <= c <= '\u052f' for c in s)

def is_multiword(s: str) -> bool:
    """
    Identifies if a token is a compound / multi-word token (created by joining 2+ words/tokens).
    Examples: ' of the', 'unique_words = set(', 'for word in', 'def calculate_total('
    Single words (e.g. ' the', 'apple', 'function', '123', 'ing', 'करण') return False.
    """
    stripped = s.strip()
    words = stripped.split()
    # 2 or more words separated by space, newline, or tab
    if len(words) >= 2:
        return True
    if '\n' in stripped or '\t' in stripped:
        return True
    # Check if contains multiple words connected by spaces
    if '  ' in s or (s.startswith(' ') and ' ' in s[1:]):
        return True
    return False

def prune_low_freq_multiwords():
    print("=" * 80)
    print("AGLM TARGETED PRUNING: REMOVING LOW-FREQ MULTI-WORD TOKENS (FREQ < 4)")
    print("=" * 80)
    
    start_t = time.perf_counter()
    vocab_gz = SOURCE_DIR / "aglm_vocab.json.gz"
    
    print(f"Loading base vocabulary from: {vocab_gz}")
    with gzip.open(vocab_gz, "rt", encoding="utf-8") as f:
        data = json.load(f)
        
    orig_tokens = data["tokens"]
    orig_special = data.get("special_tokens", {})
    orig_vocab_size = data.get("vocab_size", len(orig_tokens))
    
    print(f"Loading exact token frequencies from: {FREQ_FILE}")
    freqs = np.load(FREQ_FILE)
    
    print(f"\nOriginal Vocab Size: {orig_vocab_size:,} tokens")
    
    # Pruning counters
    retained_tokens = []
    removed_cjk = 0
    removed_cyrillic = 0
    removed_low_freq_multiword = 0
    kept_single_tokens = 0
    kept_high_freq_multiword = 0
    
    special_values = set(orig_special.values())
    
    for t in orig_tokens:
        old_id = t["id"]
        s = t.get("str", "")
        f_val = freqs[old_id] if old_id < len(freqs) else 0
        
        # 1. ALWAYS KEEP Byte Tokens (0..255) and Special Control Tokens
        if old_id < 256 or s in special_values:
            retained_tokens.append(t)
            kept_single_tokens += 1
            continue
            
        # 2. Filter Chinese / CJK
        if is_cjk(s):
            removed_cjk += 1
            continue
            
        # 3. Filter Cyrillic / Russian
        if is_cyrillic(s):
            removed_cyrillic += 1
            continue
            
        # 4. Check Multi-Word vs Single Token
        if is_multiword(s):
            if f_val < 4:
                # Remove ONLY if frequency is less than 4 (0, 1, 2, 3)
                removed_low_freq_multiword += 1
                continue
            else:
                # Keep active / high frequency multiword token
                kept_high_freq_multiword += 1
                retained_tokens.append(t)
        else:
            # SINGLE TOKEN: ALWAYS KEEP!
            kept_single_tokens += 1
            retained_tokens.append(t)
            
    print("\n--- PRUNING SUMMARY ---")
    print(f"  • Single Tokens Kept (100% Intact):       {kept_single_tokens:>10,}")
    print(f"  • High-Freq Multi-Words Kept (Freq >= 4):  {kept_high_freq_multiword:>10,}")
    print(f"  • Chinese / CJK Tokens Cut:               {removed_cjk:>10,}")
    print(f"  • Russian / Cyrillic Tokens Cut:          {removed_cyrillic:>10,}")
    print(f"  • Low-Freq Multi-Words Cut (Freq < 4):    {removed_low_freq_multiword:>10,}")
    print(f"  -------------------------------------------------------------")
    print(f"  • Total Tokens Cut:                       {removed_cjk + removed_cyrillic + removed_low_freq_multiword:>10,}")
    print(f"  • NEW CLEAN VOCAB SIZE:                   {len(retained_tokens):>10,} tokens")
    
    # Re-index token IDs continuously 0 .. N-1
    print("\nRe-indexing token IDs continuously (0 .. N-1)...")
    reindexed_tokens = []
    for new_id, t in enumerate(retained_tokens):
        reindexed_tokens.append({
            "id": new_id,
            "bytes_hex": t["bytes_hex"],
            "str": t.get("str", "")
        })
        
    reindexed_special = {}
    for st_name, old_st_id in orig_special.items():
        found_new_id = None
        for t in reindexed_tokens:
            if t.get("str") == st_name or (old_st_id < 256 and t["id"] == old_st_id):
                found_new_id = t["id"]
                break
        if found_new_id is not None:
            reindexed_special[st_name] = found_new_id
            
    clean_vocab_payload = {
        "name": "AGLM-Universal-Core-Clean",
        "strategy": "Pruned_LowFreq_Multiwords_NoCJK_NoCyrillic",
        "vocab_size": len(reindexed_tokens),
        "tokens": reindexed_tokens,
        "special_tokens": reindexed_special
    }
    
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    json_path = TARGET_DIR / "aglm_vocab.json"
    json_gz_path = TARGET_DIR / "aglm_vocab.json.gz"
    manifest_path = TARGET_DIR / "manifest.json"
    
    print(f"Writing clean vocabulary to {TARGET_DIR}...")
    json_str = json.dumps(clean_vocab_payload, ensure_ascii=False, indent=2)
    json_bytes = json_str.encode("utf-8")
    
    json_path.write_bytes(json_bytes)
    with gzip.open(json_gz_path, "wb", compresslevel=6) as f_gz:
        f_gz.write(json_bytes)
        
    gz_sha256 = hashlib.sha256(json_gz_path.read_bytes()).hexdigest()
    
    manifest_payload = {
        "model_name": "AGLM-Universal-Core-Clean",
        "vocab_size": len(reindexed_tokens),
        "algorithm": "Byte-Level BPE with Code-Aware Longest-Prefix Match Trie",
        "byte_fallback": "Guaranteed 256 byte tokens (0x00-0xFF)",
        "pruning": "Removed multiword tokens with freq < 4, removed Chinese/CJK and Cyrillic/Russian tokens; retained all single words, code primitives, Indic, and byte tokens",
        "sha256": gz_sha256,
        "export_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    
    dt = time.perf_counter() - start_t
    print(f"\n[SUCCESS] Targeted Pruned Vocabulary created in {dt:.2f}s!")
    print(f"  • JSON size:    {len(json_bytes)/(1024*1024):.1f} MB")
    print(f"  • JSON.GZ size: {json_gz_path.stat().st_size/(1024*1024):.1f} MB")
    print("=" * 80)

if __name__ == "__main__":
    prune_low_freq_multiwords()
