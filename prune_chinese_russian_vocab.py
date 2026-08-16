#!/usr/bin/env python3
"""
Prune Chinese (CJK) and Russian (Cyrillic) tokens from the AGLM Vocabulary.
Produces a clean, lightweight, ultra-fast core vocabulary with 100% UTF-8 Byte-Fallback.
"""

import os
import sys
import gzip
import json
import time
import hashlib
from pathlib import Path

ROOT = Path("/run/media/akash/18FAA791FAA76A28/aglm_project")
SOURCE_DIR = ROOT / "tokenizer" / "exported_tokenizers" / "aglm_universal_max"
TARGET_DIR = ROOT / "tokenizer" / "exported_tokenizers" / "aglm_core_clean"

def is_cjk(s: str) -> bool:
    """Check if token string contains Chinese / Japanese Kanji / Korean Hangul characters."""
    return any(
        '\u4e00' <= c <= '\u9fff' or   # CJK Unified Ideographs
        '\u3400' <= c <= '\u4dbf' or   # CJK Unified Ideographs Extension A
        '\u20000' <= c <= '\u2a6df' or # CJK Extension B
        '\u3040' <= c <= '\u30ff' or   # Hiragana & Katakana
        '\uac00' <= c <= '\ud7af'      # Korean Hangul Syllables
        for c in s
    )

def is_cyrillic(s: str) -> bool:
    """Check if token string contains Cyrillic (Russian/Slavic) characters."""
    return any('\u0400' <= c <= '\u04ff' or '\u0500' <= c <= '\u052f' for c in s)

def prune_vocabulary():
    print("=" * 75)
    print("AGLM VOCABULARY PRUNING: REMOVING CHINESE & RUSSIAN TOKENS")
    print("=" * 75)
    
    start_t = time.perf_counter()
    vocab_gz = SOURCE_DIR / "aglm_vocab.json.gz"
    
    print(f"Loading base vocabulary from: {vocab_gz}")
    with gzip.open(vocab_gz, "rt", encoding="utf-8") as f:
        data = json.load(f)
        
    orig_tokens = data["tokens"]
    orig_special = data.get("special_tokens", {})
    orig_vocab_size = data.get("vocab_size", len(orig_tokens))
    
    print(f"Original Vocab Size: {orig_vocab_size:,} tokens")
    print(f"Special Tokens: {orig_special}")
    
    # 1. Filter tokens
    new_tokens = []
    removed_cjk = 0
    removed_cyrillic = 0
    special_values = set(orig_special.values())
    
    for t in orig_tokens:
        old_id = t["id"]
        s = t.get("str", "")
        
        # Always preserve byte tokens (0..255) and control special tokens
        if old_id < 256 or s in special_values:
            new_tokens.append(t)
            continue
            
        if is_cjk(s):
            removed_cjk += 1
            continue
            
        if is_cyrillic(s):
            removed_cyrillic += 1
            continue
            
        new_tokens.append(t)
        
    print(f"\nPruning Results:")
    print(f"  • Removed Chinese / CJK Tokens:       {removed_cjk:>9,}")
    print(f"  • Removed Cyrillic / Russian Tokens:   {removed_cyrillic:>9,}")
    print(f"  • Total Tokens Cut:                   {removed_cjk + removed_cyrillic:>9,}")
    print(f"  • New Clean Vocab Size:               {len(new_tokens):>9,} tokens")
    
    # 2. Re-index token IDs continuously 0 .. N-1
    print("\nRe-indexing token IDs continuously (0 .. N-1)...")
    reindexed_tokens = []
    for new_id, t in enumerate(new_tokens):
        reindexed_tokens.append({
            "id": new_id,
            "bytes_hex": t["bytes_hex"],
            "str": t.get("str", "")
        })
        
    # Re-map special tokens
    reindexed_special = {}
    for st_name, old_st_id in orig_special.items():
        # Find new id
        found_new_id = None
        for t in reindexed_tokens:
            if t.get("str") == st_name or (old_st_id < 256 and t["id"] == old_st_id):
                found_new_id = t["id"]
                break
        if found_new_id is not None:
            reindexed_special[st_name] = found_new_id
            
    clean_vocab_payload = {
        "name": "AGLM-Universal-Core-Clean",
        "strategy": "Pruned_No_CJK_No_Cyrillic_Byte_Fallback",
        "vocab_size": len(reindexed_tokens),
        "tokens": reindexed_tokens,
        "special_tokens": reindexed_special
    }
    
    # 3. Save new exported tokenizer
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    json_path = TARGET_DIR / "aglm_vocab.json"
    json_gz_path = TARGET_DIR / "aglm_vocab.json.gz"
    manifest_path = TARGET_DIR / "manifest.json"
    
    print(f"\nWriting clean vocabulary to {TARGET_DIR}...")
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
        "pruning": "Removed Chinese/CJK and Cyrillic/Russian tokens; retained English, Code, Indic, and Byte Fallback",
        "sha256": gz_sha256,
        "export_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    
    dt = time.perf_counter() - start_t
    print(f"[SUCCESS] Cleaned vocabulary generated in {dt:.2f}s!")
    print(f"  • JSON size:    {len(json_bytes)/(1024*1024):.1f} MB")
    print(f"  • JSON.GZ size: {json_gz_path.stat().st_size/(1024*1024):.1f} MB")
    print(f"  • Manifest:     {manifest_path}")
    print("=" * 75)

if __name__ == "__main__":
    prune_vocabulary()
