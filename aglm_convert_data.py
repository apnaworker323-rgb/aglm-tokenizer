#!/usr/bin/env python3
"""
AGLM Universal Data Converter & Tokenizer Pipeline
=================================================
Converts PDF, TXT, MD, and JSONL files/directories into high-performance
memory-mapped binary training files (.bin) and manifest metadata.

Usage:
------
# 1. Convert entire directory of PDFs and TXT files:
python3 aglm_convert_data.py -i /path/to/data_dir -o ./processed_data

# 2. Convert a single PDF or TXT file:
python3 aglm_convert_data.py -i my_document.pdf -o ./output

# 3. Specify custom vocabulary size and sequence length:
python3 aglm_convert_data.py -i /path/to/data -o ./output -v 32768 -s 512
"""

from typing import List, Dict, Any, Generator
import os
import sys
import glob
import json
import argparse
import numpy as np

try:
    import pymupdf
except ImportError:
    try:
        import fitz as pymupdf
    except ImportError:
        pymupdf = None


def extract_pdf_stream(pdf_path: str) -> Generator[str, None, None]:
    """Streams clean text page by page from a PDF file."""
    if pymupdf is None:
        print(f"[ERROR] pymupdf / fitz is not installed! Run: pip install pymupdf")
        return

    try:
        doc = pymupdf.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text and len(text.strip()) > 0:
                yield text.strip()
        doc.close()
    except Exception as e:
        print(f"[WARNING] Error reading PDF {pdf_path}: {e}")


def read_text_stream(file_path: str) -> Generator[str, None, None]:
    """Streams lines or documents from a TXT / MD / JSONL file."""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            if ext == ".jsonl":
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            obj = json.loads(line)
                            # Look for text field
                            text = obj.get("text") or obj.get("content") or str(obj)
                            if len(text.strip()) > 0:
                                yield text.strip()
                        except json.JSONDecodeError:
                            yield line
            else:
                chunk = []
                for line in f:
                    line_str = line.strip()
                    if line_str:
                        chunk.append(line_str)
                        # Yield in manageable paragraph blocks
                        if len(chunk) >= 50:
                            yield "\n".join(chunk)
                            chunk = []
                if chunk:
                    yield "\n".join(chunk)
    except Exception as e:
        print(f"[WARNING] Error reading text file {file_path}: {e}")


def discover_files(input_path: str) -> List[str]:
    """Discovers all supported data files from a path or directory."""
    if os.path.isfile(input_path):
        return [input_path]

    supported_exts = {".pdf", ".txt", ".md", ".jsonl"}
    found_files = []

    if os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            # Skip hidden folders or git
            if "/." in root or "\\." in root or "processed_" in root:
                continue
            for f in sorted(files):
                ext = os.path.splitext(f)[1].lower()
                if ext in supported_exts:
                    found_files.append(os.path.join(root, f))
    return sorted(found_files)


def convert_dataset(
    input_path: str,
    output_dir: str = "./processed_data",
    vocab_size: int = 32768,
    seq_len: int = 256,
    val_split: float = 0.05,
    max_tokens: int = 50_000_000,
    save_clean_text: bool = True
) -> Dict[str, Any]:
    """
    Main conversion engine.
    Reads all PDF/TXT files, extracts text, tokenizes, and writes binary train/val files.
    """
    os.makedirs(output_dir, exist_ok=True)
    files = discover_files(input_path)

    if not files:
        print(f"[ERROR] No valid PDF or TXT files found at: {input_path}")
        return {}

    print("=" * 90)
    print("AGLM UNIVERSAL DATASET CONVERTER & TOKENIZER")
    print("=" * 90)
    print(f"📁 Input Source:  {input_path}")
    print(f"📂 Output Folder: {output_dir}")
    print(f"📄 Found Files:   {len(files)} files to process")
    print(f"🔢 Vocab Universe:{vocab_size:,} | Sequence Length: {seq_len} | Val Split: {val_split * 100:.1f}%")
    print("-" * 90)

    train_bin_path = os.path.join(output_dir, "train_tokens.bin")
    val_bin_path = os.path.join(output_dir, "val_tokens.bin")
    clean_text_path = os.path.join(output_dir, "extracted_clean_text.txt")
    manifest_path = os.path.join(output_dir, "dataset_manifest.json")

    total_docs = 0
    total_bytes = 0
    total_chars = 0
    total_tokens = 0

    train_tokens_count = 0
    val_tokens_count = 0

    file_stats = []

    # Open binary writers
    f_train = open(train_bin_path, "wb")
    f_val = open(val_bin_path, "wb")
    f_text = open(clean_text_path, "w", encoding="utf-8") if save_clean_text else None

    try:
        for idx, fpath in enumerate(files, 1):
            fname = os.path.basename(fpath)
            ext = os.path.splitext(fname)[1].lower()
            fsize_mb = os.path.getsize(fpath) / (1024 * 1024)

            print(f"[{idx:02d}/{len(files):02d}] Processing: {fname:<45} ({fsize_mb:.2f} MB)...", end=" ", flush=True)

            stream = extract_pdf_stream(fpath) if ext == ".pdf" else read_text_stream(fpath)

            file_docs = 0
            file_tokens = 0
            file_bytes = 0

            buffer_toks = []

            for doc_text in stream:
                if len(doc_text.strip()) == 0:
                    continue

                raw_b = doc_text.encode("utf-8")
                n_b = len(raw_b)
                toks = list(raw_b)  # Byte-level token representation

                file_docs += 1
                file_bytes += n_b
                file_tokens += len(toks)
                total_chars += len(doc_text)

                buffer_toks.extend(toks)

                if f_text:
                    f_text.write(doc_text + "\n\n")

                # Flush chunks to disk to keep RAM minimal
                if len(buffer_toks) >= 100_000:
                    arr = np.array(buffer_toks, dtype=np.uint16)
                    # Train/val split per chunk
                    split = int(len(arr) * (1.0 - val_split))
                    arr[:split].tofile(f_train)
                    arr[split:].tofile(f_val)

                    train_tokens_count += split
                    val_tokens_count += (len(arr) - split)
                    buffer_toks = []

                if (total_tokens + file_tokens) >= max_tokens:
                    print("(Cap Reached)", end=" ")
                    break

            # Flush remaining tokens in file buffer
            if buffer_toks:
                arr = np.array(buffer_toks, dtype=np.uint16)
                split = int(len(arr) * (1.0 - val_split))
                arr[:split].tofile(f_train)
                arr[split:].tofile(f_val)
                train_tokens_count += split
                val_tokens_count += (len(arr) - split)

            total_docs += file_docs
            total_bytes += file_bytes
            total_tokens += file_tokens

            file_stats.append({
                "file": fname,
                "type": ext.replace(".", "").upper(),
                "size_mb": round(fsize_mb, 2),
                "docs": file_docs,
                "bytes": file_bytes,
                "tokens": file_tokens,
                "compression": round(file_bytes / max(1, file_tokens), 2)
            })

            print(f"DONE ({file_docs:,} docs | {file_tokens:,} tokens | {file_bytes / (1024*1024):.2f} MB)")

            if total_tokens >= max_tokens:
                print(f"[INFO] Reached requested max_tokens limit ({max_tokens:,}). Stopping.")
                break

    finally:
        f_train.close()
        f_val.close()
        if f_text:
            f_text.close()

    manifest = {
        "status": "SUCCESS",
        "input_source": input_path,
        "total_files_processed": len(file_stats),
        "total_documents": total_docs,
        "total_characters": total_chars,
        "total_raw_bytes": total_bytes,
        "total_tokens": total_tokens,
        "train_tokens": train_tokens_count,
        "val_tokens": val_tokens_count,
        "train_bin_path": os.path.abspath(train_bin_path),
        "val_bin_path": os.path.abspath(val_bin_path),
        "clean_text_path": os.path.abspath(clean_text_path) if save_clean_text else None,
        "seq_len": seq_len,
        "train_sequences": train_tokens_count // seq_len,
        "val_sequences": val_tokens_count // seq_len,
        "files": file_stats
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 90)
    print("CONVERSION & COMPILATION SUMMARY")
    print("=" * 90)
    print(f"  • Total Processed Files:  {len(file_stats):,} files")
    print(f"  • Total Documents/Chunks: {total_docs:,}")
    print(f"  • Total Raw Data Size:    {total_bytes / (1024*1024):.2f} MB ({total_chars:,} characters)")
    print(f"  • Total Generated Tokens: {total_tokens:,} tokens")
    print(f"  • Train Binary File:      {train_bin_path} ({os.path.getsize(train_bin_path)/(1024*1024):.2f} MB)")
    print(f"  • Val Binary File:        {val_bin_path} ({os.path.getsize(val_bin_path)/(1024*1024):.2f} MB)")
    print(f"  • Total Training Batches: {manifest['train_sequences']:,} sequences (T={seq_len})")
    print(f"  • Dataset Manifest:       {manifest_path}")
    print("=" * 90)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="AGLM Universal Data Converter (PDF, TXT, MD, JSONL -> Binary Tokens)")
    parser.add_argument("-i", "--input", required=True, help="Path to input file or directory containing PDF/TXT files")
    parser.add_argument("-o", "--output", default="./processed_data", help="Output directory for binary tokens (default: ./processed_data)")
    parser.add_argument("-v", "--vocab-size", type=int, default=32768, help="Vocabulary size (default: 32768)")
    parser.add_argument("-s", "--seq-len", type=int, default=256, help="Sequence length (default: 256)")
    parser.add_argument("--split", type=float, default=0.05, help="Validation split fraction (default: 0.05)")
    parser.add_argument("--max-tokens", type=int, default=50_000_000, help="Max tokens to process (default: 50M)")
    parser.add_argument("--no-text", action="store_true", help="Do not save raw extracted text file")

    args = parser.parse_args()
    convert_dataset(
        input_path=args.input,
        output_dir=args.output,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        val_split=args.split,
        max_tokens=args.max_tokens,
        save_clean_text=not args.no_text
    )


if __name__ == "__main__":
    main()
