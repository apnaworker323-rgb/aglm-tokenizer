"""
Production Corpus Preparation & PDF Extraction Pipeline for AGLM Training.
1. Extracts clean text from all PDF files in /aglm_project/data using PyMuPDF (fitz).
2. Integrates with text datasets (FineWeb, LMSYS).
3. Pre-tokenizes and compiles streaming binary dataset chunks (.bin) with byte-metadata for fast training.
"""

from typing import List, Dict, Any, Generator
import os
import sys
import glob
import json
import pymupdf  # PyMuPDF
import numpy as np
import torch

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from aglm_tokenizer.core.tokenizer import AGLMUniversalTokenizer


DATA_DIR = "/run/media/akash/18FAA791FAA76A28/aglm_project/data"
OUTPUT_DIR = "/run/media/akash/18FAA791FAA76A28/aglm_project/data/processed_training_data"


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts and normalizes clean text from a PDF file."""
    doc = pymupdf.open(pdf_path)
    text_chunks = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text and len(text.strip()) > 0:
            text_chunks.append(text.strip())
    doc.close()
    return "\n\n".join(text_chunks)


def process_all_pdfs(data_dir: str = DATA_DIR) -> Dict[str, Any]:
    """Extracts all PDF books in data_dir."""
    pdf_files = sorted(glob.glob(os.path.join(data_dir, "*.pdf")))
    print(f"[INFO] Found {len(pdf_files)} PDF files in {data_dir}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdf_manifest = {}
    total_extracted_chars = 0
    total_extracted_bytes = 0

    out_pdf_text_file = os.path.join(OUTPUT_DIR, "extracted_pdf_books.txt")
    with open(out_pdf_text_file, "w", encoding="utf-8") as out_f:
        for idx, pdf_file in enumerate(pdf_files, 1):
            fname = os.path.basename(pdf_file)
            size_mb = os.path.getsize(pdf_file) / (1024 * 1024)
            print(f"  [{idx}/{len(pdf_files)}] Extracting: {fname} ({size_mb:.2f} MB)...", end=" ", flush=True)
            try:
                text = extract_text_from_pdf(pdf_file)
                num_chars = len(text)
                num_bytes = len(text.encode("utf-8"))
                total_extracted_chars += num_chars
                total_extracted_bytes += num_bytes

                out_f.write(f"\n\n=== FILE: {fname} ===\n\n")
                out_f.write(text)

                pdf_manifest[fname] = {
                    "size_mb": round(size_mb, 2),
                    "chars": num_chars,
                    "bytes": num_bytes,
                    "status": "SUCCESS"
                }
                print(f"DONE ({num_chars:,} chars, {num_bytes / (1024*1024):.2f} MB text)")
            except Exception as e:
                print(f"FAILED: {e}")
                pdf_manifest[fname] = {"error": str(e), "status": "FAILED"}

    print(f"\n[INFO] Total PDF Text Extracted: {total_extracted_chars:,} characters ({total_extracted_bytes / (1024*1024):.2f} MB)")
    return pdf_manifest


def build_binary_training_dataset(
    processed_dir: str = OUTPUT_DIR,
    max_train_samples: int = 50000,
    seq_len: int = 256
):
    """
    Compiles text files into ready-to-train numpy memory-mapped binary token files.
    """
    print("\n[INFO] Compiling Binary Training Data for Model Trainer...")
    
    # 1. Gather all available text files
    text_files = [
        os.path.join(processed_dir, "extracted_pdf_books.txt"),
        os.path.join(DATA_DIR, "lmsys_val_5.txt"),
        os.path.join(DATA_DIR, "fineweb_combined_val_5.txt"),
    ]

    all_lines = []
    for tf in text_files:
        if os.path.exists(tf):
            print(f"  • Reading source: {os.path.basename(tf)}...", end=" ", flush=True)
            with open(tf, "r", encoding="utf-8", errors="replace") as f:
                # Sample lines
                count = 0
                for line in f:
                    line_str = line.strip()
                    if len(line_str) > 20:
                        all_lines.append(line_str)
                        count += 1
                        if len(all_lines) >= max_train_samples:
                            break
            print(f"Loaded {count:,} documents.")
        if len(all_lines) >= max_train_samples:
            break

    print(f"[INFO] Total Clean Documents Gathered: {len(all_lines):,}")

    # 2. Tokenize into int32 / uint16 binary tokens
    # Using byte-level fallback for universal coverage
    token_list = []
    byte_list = []

    print("[INFO] Tokenizing documents into training tokens...")
    for idx, doc in enumerate(all_lines):
        raw_b = doc.encode("utf-8")
        toks = list(raw_b)  # Universal byte token representation
        token_list.extend(toks)
        byte_list.extend([1] * len(toks))
        if idx % 10000 == 0 and idx > 0:
            print(f"  Processed {idx:,}/{len(all_lines):,} docs ({len(token_list):,} tokens)...")

    # 3. Split into Train / Validation (95% / 5%)
    n_tokens = len(token_list)
    split_idx = int(n_tokens * 0.95)

    train_tokens = np.array(token_list[:split_idx], dtype=np.uint16)
    val_tokens = np.array(token_list[split_idx:], dtype=np.uint16)

    train_bin_path = os.path.join(processed_dir, "train_tokens.bin")
    val_bin_path = os.path.join(processed_dir, "val_tokens.bin")

    train_tokens.tofile(train_bin_path)
    val_tokens.tofile(val_bin_path)

    manifest = {
        "total_documents": len(all_lines),
        "total_tokens": n_tokens,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "train_bin_path": train_bin_path,
        "val_bin_path": val_bin_path,
        "seq_len": seq_len,
        "train_sequences": len(train_tokens) // seq_len,
        "val_sequences": len(val_tokens) // seq_len,
    }

    manifest_path = os.path.join(processed_dir, "dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING DATASET PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"  • Train Binary Tokens: {len(train_tokens):,} tokens ({os.path.getsize(train_bin_path)/(1024*1024):.2f} MB)")
    print(f"  • Val Binary Tokens:   {len(val_tokens):,} tokens ({os.path.getsize(val_bin_path)/(1024*1024):.2f} MB)")
    print(f"  • Total Training Sequences (seq_len={seq_len}): {manifest['train_sequences']:,}")
    print(f"  • Manifest File: {manifest_path}")
    print("=" * 80)
    return manifest


if __name__ == "__main__":
    process_all_pdfs()
    build_binary_training_dataset()
