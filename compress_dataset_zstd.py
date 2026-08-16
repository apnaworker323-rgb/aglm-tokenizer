#!/usr/bin/env python3
"""
Compress AGLM Tokenized Dataset shards (.bin -> .bin.zst) losslessly using Zstandard.
Reduces dataset footprint from ~5.7 GB down to ~2.53 GB on disk.
Includes 100% SHA256 integrity verification before removing original .bin files.
"""

import os
import sys
import glob
import time
import hashlib
import zstandard as zstd
from pathlib import Path

ROOT = Path("/run/media/akash/18FAA791FAA76A28/aglm_project")
DATASET_DIR = ROOT / "aglm_tokenized_dataset"

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def compress_shards():
    print("=" * 70)
    print("COMPRESSING AGLM DATASET SHARDS (Zstandard Level 1 Lossless)")
    print("=" * 70)
    
    splits = ["train", "val"]
    cctx = zstd.ZstdCompressor(level=1)
    dctx = zstd.ZstdDecompressor()
    
    total_orig = 0
    total_comp = 0
    start_t = time.perf_counter()
    
    for split in splits:
        split_dir = DATASET_DIR / split
        if not split_dir.exists():
            continue
            
        bin_files = sorted(split_dir.glob("shard_*.bin"))
        print(f"\nProcessing split '{split}': {len(bin_files)} shards found.")
        
        for fpath in bin_files:
            zst_path = fpath.with_name(fpath.name + ".zst")
            raw_bytes = fpath.read_bytes()
            orig_len = len(raw_bytes)
            orig_hash = sha256_bytes(raw_bytes)
            
            # Compress
            compressed = cctx.compress(raw_bytes)
            comp_len = len(compressed)
            
            # Verification: Decompress and compare SHA256
            decompressed = dctx.decompress(compressed)
            if sha256_bytes(decompressed) != orig_hash:
                raise ValueError(f"FATAL: Decompression hash mismatch on {fpath}!")
                
            # Write .bin.zst
            zst_path.write_bytes(compressed)
            
            # Remove uncompressed .bin
            fpath.unlink()
            
            total_orig += orig_len
            total_comp += comp_len
            print(f"  [OK] {fpath.name} -> {zst_path.name}: {orig_len/(1024*1024):.1f} MB -> {comp_len/(1024*1024):.1f} MB (Ratio: {orig_len/comp_len:.2f}x)")
            
    dt = time.perf_counter() - start_t
    print("\n" + "=" * 70)
    print("DATASET COMPRESSION COMPLETE & VERIFIED 100% BIT-IDENTICAL")
    print("=" * 70)
    print(f"  Original Size:   {total_orig / (1024**3):.2f} GB ({total_orig:,} bytes)")
    print(f"  Compressed Size: {total_comp / (1024**3):.2f} GB ({total_comp:,} bytes)")
    print(f"  Space Saved:     {(total_orig - total_comp) / (1024**3):.2f} GB ({(1 - total_comp/total_orig)*100:.1f}%)")
    print(f"  Time Taken:      {dt:.2f}s")
    print("=" * 70)

if __name__ == "__main__":
    compress_shards()
