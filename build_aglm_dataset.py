#!/usr/bin/env python3
"""
AGLM Production Sharded Dataset Builder (20 TB Architecture Ready)
=================================================================
Recursively ingests multi-format corpora (TXT, MD, JSONL, JSON, CSV, TSV, GZ, PDF, Parquet),
tokenizes using the AGLM ~1.55M Universal Tokenizer, and compiles deterministic,
lossless, sharded numpy.uint32 memory-mapped training files with comprehensive metadata.

Usage:
------
# 1. Dry Run / 100 MB Sample Test:
python3 build_aglm_dataset.py --input-dir /path/to/raw --output-dir ./aglm_dataset --sample-mb 100 --dry-run

# 2. Production Sharded Conversion:
python3 build_aglm_dataset.py --input-dir /path/to/raw --output-dir ./aglm_dataset --shard-tokens 50000000 --workers 4 --dedupe-exact --resume
"""

from typing import List, Dict, Tuple, Any, Optional, Generator, Set
import os
import sys
import time
import math
import glob
import json
import gzip
import hashlib
import argparse
import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

try:
    import yaml
except ImportError:
    yaml = None

try:
    import pymupdf
except ImportError:
    try:
        import fitz as pymupdf
    except ImportError:
        pymupdf = None

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aglm_tokenizer.core.tokenizer import AGLMUniversalTokenizer


SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".json", ".jsonl", ".csv", ".tsv",
    ".gz", ".pdf", ".parquet"
}

IGNORED_EXTENSIONS = {
    ".bin", ".npy", ".npz", ".pt", ".pth", ".safetensors",
    ".jpg", ".jpeg", ".png", ".gif", ".mp4", ".wav", ".mp3",
    ".zip", ".tar", ".pyc", ".lock", ".git"
}


def pack_21bit_vectorized(ids: np.ndarray) -> bytes:
    """Packs uint32 token IDs into exact 21-bit bitstream (8 tokens -> 21 bytes)."""
    pad = (8 - len(ids) % 8) % 8
    if pad:
        ids = np.pad(ids, (0, pad), constant_values=0)

    c = ids.reshape(-1, 8).astype(np.uint64)
    b = np.zeros((len(c), 21), dtype=np.uint8)

    t0, t1, t2, t3, t4, t5, t6, t7 = [c[:, i] for i in range(8)]

    b[:, 0] = t0 & 0xFF
    b[:, 1] = (t0 >> 8) & 0xFF
    b[:, 2] = ((t0 >> 16) & 0x1F) | ((t1 & 0x07) << 5)
    b[:, 3] = (t1 >> 3) & 0xFF
    b[:, 4] = (t1 >> 11) & 0xFF
    b[:, 5] = ((t1 >> 19) & 0x03) | ((t2 & 0x3F) << 2)
    b[:, 6] = (t2 >> 6) & 0xFF
    b[:, 7] = ((t2 >> 14) & 0x7F) | ((t3 & 0x01) << 7)
    b[:, 8] = (t3 >> 1) & 0xFF
    b[:, 9] = (t3 >> 9) & 0xFF
    b[:, 10] = ((t3 >> 17) & 0x0F) | ((t4 & 0x0F) << 4)
    b[:, 11] = (t4 >> 4) & 0xFF
    b[:, 12] = (t4 >> 12) & 0xFF
    b[:, 13] = ((t4 >> 20) & 0x01) | ((t5 & 0x7F) << 1)
    b[:, 14] = (t5 >> 7) & 0xFF
    b[:, 15] = ((t5 >> 15) & 0x3F) | ((t6 & 0x03) << 6)
    b[:, 16] = (t6 >> 2) & 0xFF
    b[:, 17] = (t6 >> 10) & 0xFF
    b[:, 18] = ((t6 >> 18) & 0x07) | ((t7 & 0x1F) << 3)
    b[:, 19] = (t7 >> 5) & 0xFF
    b[:, 20] = (t7 >> 13) & 0xFF

    return b.tobytes()


def unpack_21bit_vectorized(data: bytes, num_tokens: int) -> np.ndarray:
    """Unpacks 21-bit bitstream into numpy uint32 array."""
    num_chunks = (num_tokens + 7) // 8
    b = np.frombuffer(data, dtype=np.uint8).reshape(num_chunks, 21).astype(np.uint32)

    t0 = b[:, 0] | (b[:, 1] << 8) | ((b[:, 2] & 0x1F) << 16)
    t1 = ((b[:, 2] >> 5) & 0x07) | (b[:, 3] << 3) | (b[:, 4] << 11) | ((b[:, 5] & 0x03) << 19)
    t2 = ((b[:, 5] >> 2) & 0x3F) | (b[:, 6] << 6) | ((b[:, 7] & 0x7F) << 14)
    t3 = ((b[:, 7] >> 7) & 0x01) | (b[:, 8] << 1) | (b[:, 9] << 9) | ((b[:, 10] & 0x0F) << 17)
    t4 = ((b[:, 10] >> 4) & 0x0F) | (b[:, 11] << 4) | (b[:, 12] << 12) | ((b[:, 13] & 0x01) << 20)
    t5 = ((b[:, 13] >> 1) & 0x7F) | (b[:, 14] << 7) | ((b[:, 15] & 0x3F) << 15)
    t6 = ((b[:, 15] >> 6) & 0x03) | (b[:, 16] << 2) | (b[:, 17] << 10) | ((b[:, 18] & 0x07) << 18)
    t7 = ((b[:, 18] >> 3) & 0x1F) | (b[:, 19] << 5) | (b[:, 20] << 13)

    res = np.column_stack([t0, t1, t2, t3, t4, t5, t6, t7]).flatten()[:num_tokens]
    return res


class TokenizerCensus:
    """Verifies and holds metadata for the active AGLM 1.55M tokenizer."""

    def __init__(self, tokenizer_dir: str):
        self.tokenizer_dir = tokenizer_dir
        self.vocab_gz_path = os.path.join(tokenizer_dir, "aglm_vocab.json.gz")
        self.vocab_json_path = os.path.join(tokenizer_dir, "aglm_vocab.json")
        self.manifest_path = os.path.join(tokenizer_dir, "manifest.json")

        if not (os.path.exists(self.vocab_gz_path) or os.path.exists(self.vocab_json_path)):
            raise FileNotFoundError(f"No aglm_vocab found in {tokenizer_dir}")

        # Compute SHA256
        target_vocab_file = self.vocab_gz_path if os.path.exists(self.vocab_gz_path) else self.vocab_json_path
        with open(target_vocab_file, "rb") as f:
            self.model_sha256 = hashlib.sha256(f.read()).hexdigest()

        print(f"[INFO] Loading Active AGLM Tokenizer from: {tokenizer_dir}...")
        t0 = time.time()
        self.tokenizer = AGLMUniversalTokenizer.load(tokenizer_dir)
        self.load_duration = time.time() - t0

        self.vocab_size = self.tokenizer.vocab_size
        self.valid_ids = sorted(list(self.tokenizer.engine.id_to_bytes.keys()))
        self.min_id = min(self.valid_ids)
        self.max_id = max(self.valid_ids)
        self.special_tokens = self.tokenizer.engine.special_tokens
        self.eos_token_id = self.special_tokens.get("<|eos|>", 258)
        self.bos_token_id = self.special_tokens.get("<|bos|>", 257)

        # Strict Assertions
        assert self.vocab_size == 1551017, f"Vocab size mismatch! Expected 1,551,017, got {self.vocab_size:,}"
        assert self.max_id == 1551016, f"Max token ID mismatch! Expected 1,551,016, got {self.max_id:,}"
        assert self.min_id == 0, f"Min token ID {self.min_id} != 0"

        print(f"  • Tokenizer Name:        {self.tokenizer.name}")
        print(f"  • Verified Vocab Size:   {self.vocab_size:,}")
        print(f"  • Minimum Token ID:      {self.min_id}")
        print(f"  • Maximum Token ID:      {self.max_id:,}")
        print(f"  • EOS Token ID:          {self.eos_token_id}")
        print(f"  • Vocab SHA256:          {self.model_sha256}")
        print(f"  • Load Time:             {self.load_duration:.2f} s")
        print("  ✅ [ASSERTION PASSED]: Tokenizer verified 100% genuine.")


def discover_input_inventory(input_dir: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Recursively inventories all files in input_dir and categorizes them.
    """
    print(f"\n[STEP 1] INVENTORING INPUT DIRECTORY: {input_dir}...")
    supported_files = []
    ignored_files = []
    total_bytes = 0
    type_counts = {}

    for root, _, files in os.walk(input_dir):
        if "/." in root or "\\." in root or "aglm_tokenized" in root or "processed_" in root:
            continue
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()
            if ext == ".gz":
                sub_ext = os.path.splitext(os.path.splitext(fname)[0])[1].lower()
                ext = sub_ext + ".gz" if sub_ext else ".gz"

            fsize = os.path.getsize(fpath)

            if ext in SUPPORTED_EXTENSIONS or any(fname.endswith(se) for se in [".txt.gz", ".jsonl.gz"]):
                supported_files.append({
                    "path": fpath,
                    "rel_path": os.path.relpath(fpath, input_dir),
                    "filename": fname,
                    "ext": ext,
                    "size_bytes": fsize,
                    "size_mb": round(fsize / (1024 * 1024), 2)
                })
                total_bytes += fsize
                type_counts[ext] = type_counts.get(ext, 0) + 1
            else:
                ignored_files.append({"path": fpath, "ext": ext, "size_bytes": fsize})

    summary = {
        "input_dir": os.path.abspath(input_dir),
        "total_supported_files": len(supported_files),
        "total_ignored_files": len(ignored_files),
        "total_raw_bytes": total_bytes,
        "total_raw_mb": round(total_bytes / (1024 * 1024), 2),
        "total_raw_gb": round(total_bytes / (1024 * 1024 * 1024), 3),
        "format_breakdown": type_counts
    }

    print(f"  • Total Supported Files: {summary['total_supported_files']:,}")
    print(f"  • Total Raw Data Volume: {summary['total_raw_gb']} GB ({summary['total_raw_mb']} MB)")
    print(f"  • Formats Discovered:    {type_counts}")
    return supported_files, summary


def stream_documents_from_file(
    file_info: Dict[str, Any],
    text_fields: Set[str]
) -> Generator[Dict[str, Any], None, None]:
    """
    Streams clean document strings and metadata with zero RAM overhead.
    """
    fpath = file_info["path"]
    ext = file_info["ext"]

    # 1. Plain Text / Markdown / Code
    if ext in {".txt", ".md"}:
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            chunk = []
            for line in f:
                line_str = line.strip()
                if line_str:
                    chunk.append(line_str)
                    if len(chunk) >= 50:
                        yield {"text": "\n".join(chunk), "source": file_info["rel_path"]}
                        chunk = []
            if chunk:
                yield {"text": "\n".join(chunk), "source": file_info["rel_path"]}

    # 2. Gzipped Plain Text (.txt.gz)
    elif ext in {".txt.gz", ".gz"}:
        with gzip.open(fpath, "rt", encoding="utf-8", errors="replace") as f:
            chunk = []
            for line in f:
                line_str = line.strip()
                if line_str:
                    chunk.append(line_str)
                    if len(chunk) >= 50:
                        yield {"text": "\n".join(chunk), "source": file_info["rel_path"]}
                        chunk = []
            if chunk:
                yield {"text": "\n".join(chunk), "source": file_info["rel_path"]}

    # 3. JSONL / JSONL.GZ
    elif ext in {".jsonl", ".jsonl.gz"}:
        open_fn = gzip.open if ext.endswith(".gz") else open
        with open_fn(fpath, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    obj = json.loads(line_str)
                    extracted_text = ""
                    if isinstance(obj, dict):
                        # Priority text fields
                        for tf in text_fields:
                            if tf in obj and isinstance(obj[tf], str) and len(obj[tf].strip()) > 0:
                                extracted_text = obj[tf].strip()
                                break
                        # Chat message structure
                        if not extracted_text and "messages" in obj and isinstance(obj["messages"], list):
                            msg_strs = []
                            for m in obj["messages"]:
                                if isinstance(m, dict):
                                    role = m.get("role", "user")
                                    content = m.get("content", "")
                                    msg_strs.append(f"<|{role}|>\n{content}")
                            extracted_text = "\n\n".join(msg_strs)
                    elif isinstance(obj, str):
                        extracted_text = obj.strip()

                    if extracted_text and len(extracted_text) >= 10:
                        yield {"text": extracted_text, "source": file_info["rel_path"]}
                except json.JSONDecodeError:
                    continue

    # 4. JSON
    elif ext == ".json":
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        text = item.get("text", "") if isinstance(item, dict) else str(item)
                        if len(text.strip()) >= 10:
                            yield {"text": text.strip(), "source": file_info["rel_path"]}
                elif isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, str) and len(v.strip()) >= 10:
                            yield {"text": v.strip(), "source": file_info["rel_path"]}
            except Exception:
                pass

    # 5. PDF Documents
    elif ext == ".pdf" and pymupdf is not None:
        try:
            doc = pymupdf.open(fpath)
            for page_idx in range(len(doc)):
                page_text = doc[page_idx].get_text("text").strip()
                if page_text and len(page_text) >= 10:
                    yield {"text": page_text, "source": f"{file_info['rel_path']}#p{page_idx+1}"}
            doc.close()
        except Exception:
            pass

    # 6. CSV / TSV
    elif ext in {".csv", ".tsv"}:
        sep = "\t" if ext == ".tsv" else ","
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.strip().split(sep)
                for part in parts:
                    clean_p = part.strip().strip('"')
                    if len(clean_p) >= 20:
                        yield {"text": clean_p, "source": file_info["rel_path"]}

    # 7. Parquet
    elif ext == ".parquet" and pq is not None:
        try:
            table = pq.read_table(fpath)
            df = table.to_pandas()
            for col in df.columns:
                if col in text_fields or df[col].dtype == object:
                    for val in df[col].dropna():
                        s = str(val).strip()
                        if len(s) >= 10:
                            yield {"text": s, "source": file_info["rel_path"]}
        except Exception:
            pass


class ProductionDatasetBuilder:
    """
    Production-grade, sharded, streaming dataset builder.
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        tokenizer_census: TokenizerCensus,
        shard_tokens: int = 50_000_000,
        val_ratio: float = 0.005,
        dedupe_exact: bool = True,
        enable_packed21: bool = False,
        text_fields: Optional[List[str]] = None,
        sample_mb: Optional[int] = None,
        dry_run: bool = False,
        resume: bool = True
    ):
        self.input_dir = os.path.abspath(input_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.tc = tokenizer_census
        self.shard_tokens = shard_tokens
        self.val_ratio = val_ratio
        self.dedupe_exact = dedupe_exact
        self.enable_packed21 = enable_packed21
        self.sample_mb = sample_mb
        self.dry_run = dry_run
        self.resume = resume

        self.text_fields = set(text_fields or [
            "text", "content", "body", "document", "response",
            "prompt", "question", "answer", "messages"
        ])

        # Directory structure
        self.train_dir = os.path.join(self.output_dir, "train")
        self.val_dir = os.path.join(self.output_dir, "val")
        self.meta_dir = os.path.join(self.output_dir, "metadata")
        self.logs_dir = os.path.join(self.output_dir, "logs")
        self.scripts_dir = os.path.join(self.output_dir, "scripts")
        self.packed_dir = os.path.join(self.output_dir, "packed21") if enable_packed21 else None

        for d in [self.train_dir, self.val_dir, self.meta_dir, self.logs_dir, self.scripts_dir]:
            os.makedirs(d, exist_ok=True)
        if self.packed_dir:
            os.makedirs(os.path.join(self.packed_dir, "train"), exist_ok=True)
            os.makedirs(os.path.join(self.packed_dir, "val"), exist_ok=True)

        # Buffers & Indexing
        self.train_buffer: List[int] = []
        self.val_buffer: List[int] = []
        self.train_shard_idx = 0
        self.val_shard_idx = 0

        self.shards_manifest = {"train": [], "val": []}
        self.processed_sources: Set[str] = set()

        # Telemetry
        self.total_docs_scanned = 0
        self.total_docs_written = 0
        self.total_exact_duplicates = 0
        self.total_raw_bytes = 0
        self.total_train_tokens = 0
        self.total_val_tokens = 0

        # Frequency array for 1.55M vocab
        self.token_freq = np.zeros(self.tc.vocab_size, dtype=np.uint64)
        self.seen_hashes: Set[int] = set()

        # Checkpoint restoration
        if self.resume:
            self._restore_checkpoint()

    def _restore_checkpoint(self):
        """Restores progress from previous shard_index and source_files."""
        shard_idx_path = os.path.join(self.meta_dir, "shard_index.json")
        src_files_path = os.path.join(self.meta_dir, "source_files.jsonl")

        if os.path.exists(shard_idx_path):
            try:
                with open(shard_idx_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.shards_manifest = data.get("shards", self.shards_manifest)
                    self.train_shard_idx = len(self.shards_manifest.get("train", []))
                    self.val_shard_idx = len(self.shards_manifest.get("val", []))
                print(f"[RESUME] Loaded existing shard index: {self.train_shard_idx} train shards, {self.val_shard_idx} val shards.")
            except Exception as e:
                print(f"[WARNING] Could not restore shard index: {e}")

        if os.path.exists(src_files_path):
            try:
                with open(src_files_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            obj = json.loads(line)
                            self.processed_sources.add(obj.get("path", ""))
                print(f"[RESUME] Loaded {len(self.processed_sources):,} previously completed source files.")
            except Exception as e:
                print(f"[WARNING] Could not restore source files checkpoint: {e}")

    def _flush_shard(self, split: str, is_final: bool = False):
        """Flushes in-memory buffer to an atomic little-endian uint32 shard."""
        buffer = self.train_buffer if split == "train" else self.val_buffer
        if not buffer:
            return

        if not is_final and len(buffer) < self.shard_tokens:
            return

        shard_num = self.train_shard_idx if split == "train" else self.val_shard_idx
        shard_fname = f"shard_{shard_num:05d}.bin"
        target_dir = self.train_dir if split == "train" else self.val_dir
        final_path = os.path.join(target_dir, shard_fname)
        tmp_path = os.path.join(target_dir, f"{shard_fname}.tmp")

        arr = np.array(buffer, dtype=np.uint32)
        n_toks = len(arr)

        # Assert token range bounds
        assert arr.min() >= 0, f"Negative token ID detected in {split} buffer!"
        assert arr.max() <= self.tc.max_id, f"Token ID {arr.max()} exceeds max vocab ID {self.tc.max_id}!"

        # Write to temporary file
        with open(tmp_path, "wb") as f:
            arr.tofile(f)
            f.flush()
            os.fsync(f.fileno())

        # Verify size & SHA256
        actual_size = os.path.getsize(tmp_path)
        expected_size = n_toks * 4
        assert actual_size == expected_size, f"Size mismatch! Actual {actual_size} != Expected {expected_size}"

        with open(tmp_path, "rb") as f:
            shard_sha256 = hashlib.sha256(f.read()).hexdigest()

        # Atomic rename
        os.replace(tmp_path, final_path)

        # Optional 21-bit packed generation
        packed_info = None
        if self.enable_packed21 and self.packed_dir:
            packed_target_dir = os.path.join(self.packed_dir, split)
            packed_final_path = os.path.join(packed_target_dir, shard_fname)
            packed_bytes = pack_21bit_vectorized(arr)
            with open(packed_final_path, "wb") as f_p:
                f_p.write(packed_bytes)
            packed_info = {
                "packed_bytes": len(packed_bytes),
                "packed_path": os.path.relpath(packed_final_path, self.output_dir)
            }

        shard_record = {
            "shard_id": shard_num,
            "split": split,
            "filename": shard_fname,
            "path": os.path.relpath(final_path, self.output_dir),
            "token_count": n_toks,
            "byte_size": actual_size,
            "min_token_id": int(arr.min()),
            "max_token_id": int(arr.max()),
            "sha256": shard_sha256,
            "dtype": "uint32",
            "endian": "little",
            "packed21": packed_info
        }

        self.shards_manifest[split].append(shard_record)

        if split == "train":
            self.train_shard_idx += 1
            self.total_train_tokens += n_toks
            self.train_buffer = []
        else:
            self.val_shard_idx += 1
            self.total_val_tokens += n_toks
            self.val_buffer = []

        self._persist_manifest()

    def _persist_manifest(self):
        """Atomically persists dataset_manifest.json and shard_index.json."""
        shard_idx_path = os.path.join(self.meta_dir, "shard_index.json")
        with open(shard_idx_path + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"shards": self.shards_manifest}, f, indent=2)
        os.replace(shard_idx_path + ".tmp", shard_idx_path)

    def build_dataset(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes streaming ingestion across all files.
        """
        print(f"\n[STEP 2] STARTING PRODUCTION STREAMING DATASET BUILDER...")
        print(f"  • Target Shard Capacity: {self.shard_tokens:,} tokens (~{self.shard_tokens * 4 / (1024*1024):.1f} MB uint32)")
        print(f"  • Validation Ratio:      {self.val_ratio * 100:.2f}% (Document-hash split)")
        print(f"  • Deduplication Exact:   {self.dedupe_exact}")
        print(f"  • 21-Bit Packing:        {self.enable_packed21}")
        print("-" * 90)

        t_start = time.time()
        sample_byte_cap = (self.sample_mb * 1024 * 1024) if self.sample_mb else None
        accumulated_raw_bytes = 0
        src_log_file = open(os.path.join(self.meta_dir, "source_files.jsonl"), "a", encoding="utf-8")

        try:
            for f_idx, finfo in enumerate(files, 1):
                fpath = finfo["path"]
                fname = finfo["filename"]
                fsize_mb = finfo["size_mb"]

                if fpath in self.processed_sources:
                    print(f"[{f_idx:02d}/{len(files):02d}] SKIPPING (Already Processed): {fname}")
                    continue

                print(f"[{f_idx:02d}/{len(files):02d}] Ingesting: {fname:<40} ({fsize_mb:.2f} MB)...", end=" ", flush=True)

                file_doc_count = 0
                file_token_count = 0
                file_raw_bytes = 0

                for doc in stream_documents_from_file(finfo, self.text_fields):
                    doc_text = doc["text"]
                    if len(doc_text) < 10:
                        continue

                    # Exact deduplication check
                    if self.dedupe_exact:
                        doc_hash = hash(doc_text)
                        if doc_hash in self.seen_hashes:
                            self.total_exact_duplicates += 1
                            continue
                        self.seen_hashes.add(doc_hash)

                    self.total_docs_scanned += 1
                    raw_b = doc_text.encode("utf-8")
                    n_b = len(raw_b)
                    file_raw_bytes += n_b
                    self.total_raw_bytes += n_b
                    accumulated_raw_bytes += n_b

                    # Tokenize
                    toks, _ = self.tc.tokenizer.engine.encode(doc_text)
                    # Append EOS document separator
                    toks.append(self.tc.eos_token_id)
                    n_toks = len(toks)
                    file_token_count += n_toks

                    # Update frequency stats
                    for tid in toks:
                        if tid < len(self.token_freq):
                            self.token_freq[tid] += 1

                    # Deterministic document-level split
                    # MD5 hash of first 128 chars + length
                    h_val = int(hashlib.md5(raw_b[:128]).hexdigest()[:4], 16) % 10000
                    if h_val < int(self.val_ratio * 10000):
                        self.val_buffer.extend(toks)
                        if len(self.val_buffer) >= self.shard_tokens:
                            self._flush_shard("val")
                    else:
                        self.train_buffer.extend(toks)
                        if len(self.train_buffer) >= self.shard_tokens:
                            self._flush_shard("train")

                    file_doc_count += 1
                    self.total_docs_written += 1

                    if sample_byte_cap and accumulated_raw_bytes >= sample_byte_cap:
                        print(f"[SAMPLE CAP REACHED: {self.sample_mb} MB]", end=" ")
                        break

                # Record source log
                src_log_file.write(json.dumps({
                    "path": fpath,
                    "filename": fname,
                    "docs": file_doc_count,
                    "tokens": file_token_count,
                    "bytes": file_raw_bytes,
                    "timestamp": time.time()
                }) + "\n")
                src_log_file.flush()
                self.processed_sources.add(fpath)

                rss_mb = psutil.Process().memory_info().rss / (1024 * 1024) if psutil else 0
                elapsed = time.time() - t_start
                overall_mb_s = (self.total_raw_bytes / (1024 * 1024)) / max(1e-6, elapsed)
                print(f"DONE ({file_doc_count:,} docs | {file_token_count:,} toks | RSS: {rss_mb:.1f} MB | {overall_mb_s:.2f} MB/s)")

                if sample_byte_cap and accumulated_raw_bytes >= sample_byte_cap:
                    print(f"\n[INFO] Sample limit of {self.sample_mb} MB reached. Stopping stream.")
                    break

            # Flush final leftover buffers
            self._flush_shard("train", is_final=True)
            self._flush_shard("val", is_final=True)

        finally:
            src_log_file.close()

        total_time = time.time() - t_start
        total_tokens_all = self.total_train_tokens + self.total_val_tokens
        bytes_per_tok = self.total_raw_bytes / max(1, total_tokens_all)

        # Token frequency statistics
        used_mask = (self.token_freq > 0)
        num_unique_used = int(np.sum(used_mask))
        vocab_pct_used = (num_unique_used / self.tc.vocab_size) * 100

        counts_used = self.token_freq[used_mask]
        p50 = float(np.percentile(counts_used, 50)) if len(counts_used) > 0 else 0
        p90 = float(np.percentile(counts_used, 90)) if len(counts_used) > 0 else 0
        p99 = float(np.percentile(counts_used, 99)) if len(counts_used) > 0 else 0

        # Top 10 tokens
        top_indices = np.argsort(-self.token_freq)[:10]
        top_tokens = []
        for tid in top_indices:
            cnt = int(self.token_freq[tid])
            if cnt > 0:
                tb = self.tc.tokenizer.engine.id_to_bytes.get(int(tid), b"")
                tstr = tb.decode("utf-8", errors="replace").replace("\n", "\\n").replace("\t", "\\t")
                top_tokens.append({"id": int(tid), "str": tstr, "count": cnt, "pct": round(cnt / max(1, total_tokens_all) * 100, 2)})

        # Save Final Manifest
        manifest = {
            "dataset_name": "AGLM-Universal-Production-Dataset",
            "tokenizer": {
                "name": self.tc.tokenizer.name,
                "vocab_size": self.tc.vocab_size,
                "max_token_id": self.tc.max_id,
                "sha256": self.tc.model_sha256,
                "eos_token_id": self.tc.eos_token_id,
                "bos_token_id": self.tc.bos_token_id
            },
            "statistics": {
                "total_documents_scanned": self.total_docs_scanned,
                "total_documents_written": self.total_docs_written,
                "exact_duplicates_skipped": self.total_exact_duplicates,
                "total_raw_bytes": self.total_raw_bytes,
                "total_raw_mb": round(self.total_raw_bytes / (1024 * 1024), 2),
                "total_raw_gb": round(self.total_raw_bytes / (1024 * 1024 * 1024), 3),
                "total_tokens": total_tokens_all,
                "train_tokens": self.total_train_tokens,
                "val_tokens": self.total_val_tokens,
                "bytes_per_token": round(bytes_per_tok, 4),
                "sequence_compression_factor": round(bytes_per_tok, 2),
                "unique_tokens_used": num_unique_used,
                "vocab_utilization_pct": round(vocab_pct_used, 2),
                "p50_frequency": p50,
                "p90_frequency": p90,
                "p99_frequency": p99,
                "top_tokens": top_tokens
            },
            "shards": self.shards_manifest,
            "configuration": {
                "dtype": "uint32",
                "endian": "little",
                "shard_tokens_target": self.shard_tokens,
                "val_ratio": self.val_ratio,
                "dedupe_exact": self.dedupe_exact,
                "enable_packed21": self.enable_packed21,
                "elapsed_seconds": round(total_time, 2),
                "throughput_mb_s": round((self.total_raw_bytes / (1024 * 1024)) / max(1e-6, total_time), 2),
                "throughput_tok_s": int(total_tokens_all / max(1e-6, total_time))
            }
        }

        manifest_path = os.path.join(self.meta_dir, "dataset_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print("\n" + "=" * 90)
        print("PRODUCTION DATASET BUILD COMPLETE")
        print("=" * 90)
        print(f"  • Total Raw Data Ingested: {manifest['statistics']['total_raw_mb']} MB ({manifest['statistics']['total_raw_gb']} GB)")
        print(f"  • Total Tokens Compiled:   {total_tokens_all:,} tokens")
        print(f"  • Train Shards Created:    {len(self.shards_manifest['train'])} shards ({self.total_train_tokens:,} tokens)")
        print(f"  • Val Shards Created:      {len(self.shards_manifest['val'])} shards ({self.total_val_tokens:,} tokens)")
        print(f"  • Information Density:     {manifest['statistics']['bytes_per_token']} raw bytes/token position")
        print(f"  • Unique Vocab Used:       {num_unique_used:,} ({vocab_pct_used:.2f}%)")
        print(f"  • Manifest File:           {manifest_path}")
        print("=" * 90)

        return manifest


def main():
    parser = argparse.ArgumentParser(description="AGLM Production Sharded Dataset Builder (20 TB Ready)")
    parser.add_argument("-i", "--input-dir", required=True, help="Input directory with raw files")
    parser.add_argument("-o", "--output-dir", required=True, help="Output directory for sharded dataset")
    parser.add_argument("-t", "--tokenizer-path", default="exported_tokenizers/aglm_universal_max", help="Path to tokenizer directory")
    parser.add_argument("--shard-tokens", type=int, default=50_000_000, help="Tokens per shard file (default: 50M)")
    parser.add_argument("--val-ratio", type=float, default=0.005, help="Validation ratio (default: 0.005 = 0.5 percent)")
    parser.add_argument("--workers", type=int, default=4, help="Worker count (default: 4)")
    parser.add_argument("--dedupe-exact", action="store_true", default=True, help="Enable exact-document hash deduplication")
    parser.add_argument("--packed21", action="store_true", help="Generate optional 21-bit packed shards alongside uint32")
    parser.add_argument("--sample-mb", type=int, default=None, help="Process only first N megabytes for dry-run/audit")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run verification mode")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from previous checkpoint")

    args = parser.parse_args()

    # 1. Verify Tokenizer
    tc = TokenizerCensus(args.tokenizer_path)

    # 2. Inventory Files
    files, inv_summary = discover_input_inventory(args.input_dir)

    # Write inventory report
    inv_md = f"""# Dataset Input Inventory Report

**Date**: August 15, 2026  
**Input Directory**: `{inv_summary['input_dir']}`  
**Total Supported Files**: {inv_summary['total_supported_files']:,}  
**Total Raw Volume**: {inv_summary['total_raw_gb']} GB ({inv_summary['total_raw_mb']} MB)  

## Format Breakdown
{json.dumps(inv_summary['format_breakdown'], indent=2)}

## Discovered Files
| Index | File Name | Extension | Size (MB) | Full Path |
|:---|:---|:---:|:---:|:---|
"""
    for idx, f in enumerate(files, 1):
        inv_md += f"| {idx} | `{f['filename']}` | `{f['ext']}` | {f['size_mb']} MB | `{f['path']}` |\n"

    with open("DATASET_INPUT_INVENTORY.md", "w", encoding="utf-8") as f:
        f.write(inv_md)
    print(f"[INFO] Written Inventory to: DATASET_INPUT_INVENTORY.md")

    # 3. Build Dataset
    builder = ProductionDatasetBuilder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tokenizer_census=tc,
        shard_tokens=args.shard_tokens,
        val_ratio=args.val_ratio,
        dedupe_exact=args.dedupe_exact,
        enable_packed21=args.packed21,
        sample_mb=args.sample_mb,
        dry_run=args.dry_run,
        resume=args.resume
    )

    manifest = builder.build_dataset(files)
    return manifest


if __name__ == "__main__":
    main()
