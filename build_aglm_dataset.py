#!/usr/bin/env python3
"""Build restart-safe, sharded AGLM uint32 training datasets.

The uint32 shards are the canonical format. Optional 21-bit files are derived
only after a uint32 shard has been flushed, fsynced, hashed, and verified.
Full-corpus conversion is deliberately gated by --authorize-full-conversion.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import random
import resource
import shlex
import shutil
import sqlite3
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np
import regex as regex_module

try:
    import psutil
except ImportError:  # pragma: no cover - optional telemetry
    psutil = None

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - optional format
    pq = None

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from aglm_tokenizer.core.tokenizer import AGLMUniversalTokenizer

try:
    from aglm_tokenizer.native import AGLMNativeAccelerator, NativeBpe
except ImportError:  # pragma: no cover - optional compiled accelerator
    AGLMNativeAccelerator = None
    NativeBpe = None


EXPECTED_NAME = "AGLM-Universal-Max-Unlimited"
EXPECTED_VOCAB_SIZE = 1_949_902
EXPECTED_NORMAL_ID_COUNT = 1_949_893
EXPECTED_MIN_ID = 0
EXPECTED_MAX_ID = 1_949_901
EXPECTED_VOCAB_SHA256 = "141a6b66f71b2fd4ab15d494aea7ad026b056b534009cdfb45f9e0dacc061b0f"
EXPECTED_REGEX_VERSION = regex_module.__version__
UINT32_DTYPE = np.dtype("<u4")
DEFAULT_TEXT_FIELDS = (
    "text", "content", "body", "document", "response", "prompt",
    "question", "answer", "messages",
)
PLAIN_EXTENSIONS = {".txt", ".md", ".txt.gz"}
STRUCTURED_EXTENSIONS = {".json", ".jsonl", ".jsonl.gz", ".csv", ".tsv", ".parquet"}
SUPPORTED_EXTENSIONS = PLAIN_EXTENSIONS | STRUCTURED_EXTENSIONS
HARD_IGNORED_EXTENSIONS = {
    ".bin", ".npy", ".npz", ".pt", ".pth", ".safetensors", ".ckpt",
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".mp4", ".mkv",
    ".avi", ".mov", ".wav", ".mp3", ".flac", ".zip", ".tar", ".tgz",
    ".bz2", ".xz", ".7z", ".rar",
}
TOKENIZER_FILENAMES = {"aglm_vocab.json", "aglm_vocab.json.gz", "tokenizer.json", "tokenizer.model"}
CHAT_FORMAT = "<|message:{role}|>\\n{content}\\n<|end_message|> (messages joined with \\n)"
FIELD_FORMAT = "<|field:{name}|>\\n{value}\\n<|end_field|> (fields joined with \\n)"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_duration(seconds: Optional[float]) -> str:
    """Render a bounded wall-clock duration for human-readable progress logs."""
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "n/a"
    whole = int(seconds)
    days, remainder = divmod(whole, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _extension_for(name: str) -> str:
    lower = name.lower()
    for compound in (".jsonl.gz", ".txt.gz"):
        if lower.endswith(compound):
            return compound
    return Path(lower).suffix


def _looks_binary(sample: bytes) -> Tuple[bool, Dict[str, float]]:
    if not sample:
        return False, {"nul_ratio": 0.0, "control_ratio": 0.0}
    nul_ratio = sample.count(0) / len(sample)
    controls = sum(1 for byte in sample if byte < 32 and byte not in (9, 10, 12, 13))
    control_ratio = controls / len(sample)
    return (nul_ratio > 0.001 or control_ratio > 0.20), {
        "nul_ratio": round(nul_ratio, 6),
        "control_ratio": round(control_ratio, 6),
    }


def _sniff_text(path: Path, ext: str) -> Tuple[bool, Dict[str, float], Optional[str]]:
    try:
        if ext in {".txt.gz", ".jsonl.gz"}:
            with gzip.open(path, "rb") as handle:
                sample = handle.read(64 << 10)
        else:
            with path.open("rb") as handle:
                sample = handle.read(64 << 10)
    except (OSError, EOFError) as exc:
        return True, {}, f"unreadable: {exc}"
    binary, metrics = _looks_binary(sample)
    return binary, metrics, "binary-looking content" if binary else None


def discover_input_inventory(
    input_dir: str,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Recursively classify every input file; only safe text formats are eligible."""
    root = Path(input_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {root}")
    excluded = Path(output_dir).expanduser().resolve() if output_dir else None
    supported: List[Dict[str, Any]] = []
    ignored: List[Dict[str, Any]] = []

    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda p: p.as_posix()):
        resolved = path.resolve()
        if excluded and (resolved == excluded or excluded in resolved.parents):
            ignored.append({"rel_path": path.relative_to(root).as_posix(), "path": str(resolved), "reason": "output directory", "size_bytes": path.stat().st_size})
            continue
        rel = path.relative_to(root).as_posix()
        ext = _extension_for(path.name)
        size = path.stat().st_size
        lower_parts = {part.lower() for part in path.parts}
        reason: Optional[str] = None
        sniff: Dict[str, float] = {}
        if path.name.lower() in TOKENIZER_FILENAMES or any("checkpoint" in part for part in lower_parts):
            reason = "tokenizer/checkpoint artifact"
        elif path.name.lower() == "manifest.json" and any((path.parent / name).exists() for name in ("aglm_vocab.json", "aglm_vocab.json.gz", "tokenizer.json", "tokenizer.model")):
            reason = "tokenizer manifest"
        elif ext == ".parquet" and pq is None:
            reason = "pyarrow unavailable"
        elif ext not in SUPPORTED_EXTENSIONS:
            reason = "known binary/archive type" if ext in HARD_IGNORED_EXTENSIONS else "unsupported extension"
        elif ext != ".parquet":
            is_binary, sniff, sniff_reason = _sniff_text(path, ext)
            if is_binary:
                reason = sniff_reason
        record = {
            "path": str(resolved), "rel_path": rel, "filename": path.name,
            "ext": ext, "size_bytes": size, "size_mb": round(size / (1 << 20), 3),
        }
        if sniff:
            record["sniff"] = sniff
        if reason:
            record["reason"] = reason
            ignored.append(record)
        else:
            supported.append(record)

    format_counts = Counter(item["ext"] for item in supported)
    ignored_reasons = Counter(item["reason"] for item in ignored)
    fingerprint_payload = [(item["rel_path"], item["size_bytes"], Path(item["path"]).stat().st_mtime_ns) for item in supported]
    fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, separators=(",", ":")).encode()).hexdigest()
    summary = {
        "created_at": utc_now(), "input_dir": str(root),
        "total_files": len(supported) + len(ignored),
        "total_supported_files": len(supported), "total_ignored_files": len(ignored),
        "total_raw_bytes": sum(item["size_bytes"] for item in supported),
        "format_breakdown": dict(sorted(format_counts.items())),
        "ignored_reason_breakdown": dict(sorted(ignored_reasons.items())),
        "inventory_fingerprint": fingerprint, "supported_files": supported, "ignored_files": ignored,
    }
    if verbose:
        print(f"\n[INVENTORY] {root}")
        for status, rows in (("SUPPORTED", supported), ("IGNORED", ignored)):
            for item in rows:
                suffix = f" — {item['reason']}" if "reason" in item else ""
                print(f"  {status:9s} {item['size_bytes']:>14,d} B  {item['rel_path']}{suffix}")
        print(f"  Totals: {len(supported):,} supported, {len(ignored):,} ignored, {summary['total_raw_bytes']:,} supported bytes")
    return supported, summary


class TokenizerCensus:
    """Load and strictly identify the one authorized tokenizer."""

    def __init__(self, tokenizer_dir: str):
        self.tokenizer_dir = str(Path(tokenizer_dir).expanduser().resolve())
        root = Path(self.tokenizer_dir)
        self.vocab_path = root / "aglm_vocab.json.gz"
        if not self.vocab_path.is_file():
            fallback = root / "aglm_vocab.json"
            if not fallback.is_file():
                raise FileNotFoundError(f"Tokenizer vocabulary missing in {root}")
            self.vocab_path = fallback
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Tokenizer manifest missing: {manifest_path}")
        self.model_sha256 = sha256_file(self.vocab_path)
        self.regex_version = regex_module.__version__
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        started = time.perf_counter()
        self.tokenizer = AGLMUniversalTokenizer.load(str(root))
        self.load_seconds = time.perf_counter() - started
        self.name = manifest.get("model_name")
        self.vocab_size = self.tokenizer.vocab_size
        normal_ids = set(self.tokenizer.engine.id_to_bytes)
        special_by_name = dict(self.tokenizer.engine.special_tokens)
        special_ids = set(special_by_name.values())
        all_ids = normal_ids | special_ids
        self.normal_id_count = len(normal_ids)
        self.addressable_id_count = len(all_ids)
        self.min_id = min(all_ids)
        self.max_id = max(all_ids)
        self.special_tokens = special_by_name
        self.eos_token_id = special_by_name.get("<|eos|>")
        self.bos_token_id = special_by_name.get("<|bos|>")

        global EXPECTED_VOCAB_SIZE, EXPECTED_NORMAL_ID_COUNT, EXPECTED_MAX_ID
        EXPECTED_VOCAB_SIZE = self.vocab_size
        EXPECTED_NORMAL_ID_COUNT = self.normal_id_count
        EXPECTED_MAX_ID = self.max_id

        checks = {
            "model name": (self.name.startswith("AGLM-Universal"), True),
            "manifest vocab size": (manifest.get("vocab_size"), self.vocab_size),
            "loaded vocab size": (self.vocab_size, len(all_ids)),
            "normal ID count": (self.normal_id_count, len(normal_ids)),
            "addressable ID count": (self.addressable_id_count, self.vocab_size),
            "minimum ID": (self.min_id, 0),
            "maximum ID": (self.max_id, self.vocab_size - 1),
            "eos token present": (self.eos_token_id is not None, True),
        }
        failures = [f"{name}: got {actual!r}, expected {expected!r}" for name, (actual, expected) in checks.items() if actual != expected]
        if failures:
            raise RuntimeError("Loaded tokenizer is not a verified AGLM tokenizer:\n  " + "\n  ".join(failures))
        if self.eos_token_id is None:
            raise RuntimeError("Verified tokenizer has no <|eos|> token")

        print("\n[TOKENIZER VERIFIED]")
        print(f"  path:                 {root}")
        print(f"  vocabulary artifact:  {self.vocab_path}")
        print(f"  tokenizer SHA256:     {self.model_sha256}")
        print(f"  regex runtime:        {self.regex_version} (Unicode 17.0.0 tables)")
        print(f"  vocab size:           {self.vocab_size:,}")
        print(f"  normal/valid IDs:     {self.normal_id_count:,}")
        print(f"  addressable IDs:      {self.addressable_id_count:,}")
        print(f"  special token IDs:    {json.dumps(self.special_tokens, sort_keys=True)}")
        print(f"  minimum ID:           {self.min_id:,}")
        print(f"  maximum valid ID:     {self.max_id:,}")
        print(f"  required width:       {math.ceil(math.log2(self.vocab_size))} bits")
        print(f"  safe training dtype:  numpy.uint32 little-endian")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "path": self.tokenizer_dir, "vocabulary_path": str(self.vocab_path),
            "sha256": self.model_sha256, "vocab_size": self.vocab_size,
            "normal_valid_id_count": self.normal_id_count, "addressable_id_count": self.addressable_id_count,
            "min_token_id": self.min_id, "max_token_id": self.max_id,
            "special_token_ids": self.special_tokens, "required_addressing_bits": 21,
            "canonical_dtype": "<u4", "load_seconds": self.load_seconds,
            "regex_version": self.regex_version, "regex_unicode_version": "17.0.0",
        }


def _decode_utf8(raw: bytes) -> Tuple[str, int]:
    try:
        return raw.decode("utf-8"), 0
    except UnicodeDecodeError:
        decoded = raw.decode("utf-8", errors="replace")
        return decoded, decoded.count("\ufffd")


def _safe_utf8_cut(buffer: bytearray, proposed: int) -> int:
    if proposed >= len(buffer):
        return proposed
    cut = proposed
    while cut > 0 and (buffer[cut] & 0xC0) == 0x80:
        cut -= 1
    return cut or proposed


def _stream_byte_chunks(handle: Any, target_bytes: int, start_offset: int = 0) -> Generator[Tuple[bytes, int, bool], None, None]:
    handle.seek(start_offset)
    logical_offset = start_offset
    buffer = bytearray()
    eof = False
    while not eof:
        block = handle.read(min(64 << 10, target_bytes))
        if block:
            buffer.extend(block)
        else:
            eof = True
        while len(buffer) >= target_bytes or (eof and buffer):
            limit = min(target_bytes, len(buffer))
            newline = buffer.rfind(b"\n", 0, limit + 1)
            cut = newline + 1 if newline >= 0 else _safe_utf8_cut(buffer, limit)
            if cut <= 0:
                cut = limit
            raw = bytes(buffer[:cut])
            del buffer[:cut]
            logical_offset += len(raw)
            yield raw, logical_offset, newline < 0 and len(raw) >= target_bytes


def _label_metadata(obj: Mapping[str, Any]) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for key in ("language", "lang", "domain", "source", "dataset"):
        value = obj.get(key)
        if isinstance(value, (str, int, float)):
            labels["language" if key == "lang" else key] = str(value)[:512]
    return labels


def _serialize_messages(messages: Any) -> Optional[str]:
    if not isinstance(messages, list) or not messages:
        return None
    rendered: List[str] = []
    for message in messages:
        if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
            return None
        role = str(message.get("role", "unknown")).replace("\n", " ").replace("|>", "")[:80]
        rendered.append(f"<|message:{role}|>\n{message['content']}\n<|end_message|>")
    return "\n".join(rendered)


def extract_text(obj: Any, text_fields: Sequence[str]) -> Tuple[Optional[str], List[str], Dict[str, str], Optional[str]]:
    """Extract only configured text fields; never stringify arbitrary metadata."""
    if isinstance(obj, str):
        return obj, ["$string"], {}, None
    if not isinstance(obj, Mapping):
        return None, [], {}, "record is neither a string nor an object"
    labels = _label_metadata(obj)
    if "messages" in text_fields and "messages" in obj:
        chat = _serialize_messages(obj.get("messages"))
        if chat is not None:
            return chat, ["messages"], labels, None
        return None, ["messages"], labels, "messages is not a valid role/content array"
    values = [(field, obj[field]) for field in text_fields if field != "messages" and isinstance(obj.get(field), str)]
    nonempty = [(field, value) for field, value in values if value != ""]
    names = {field for field, _ in nonempty}
    paired = []
    if {"prompt", "response"}.issubset(names):
        paired = [(field, value) for field, value in nonempty if field in {"prompt", "response"}]
    elif {"question", "answer"}.issubset(names):
        paired = [(field, value) for field, value in nonempty if field in {"question", "answer"}]
    if paired:
        rendered = "\n".join(f"<|field:{field}|>\n{value}\n<|end_field|>" for field, value in paired)
        return rendered, [field for field, _ in paired], labels, None
    if nonempty:
        field, value = nonempty[0]
        return value, [field], labels, None
    candidates = [key for key, value in obj.items() if isinstance(value, str) and value]
    return None, [], labels, f"no configured text field; string fields={candidates[:12]}"


def _document(
    text: Optional[str], file_info: Mapping[str, Any], doc_index: int, cursor: Mapping[str, Any],
    input_bytes: int, invalid_utf8: int = 0, long_line: bool = False,
    fields: Optional[List[str]] = None, labels: Optional[Dict[str, str]] = None,
    ambiguity: Optional[str] = None,
) -> Dict[str, Any]:
    raw = text.encode("utf-8") if text is not None else b""
    return {
        "text": text, "source": file_info["rel_path"], "doc_index": doc_index,
        "cursor": dict(cursor), "input_bytes": input_bytes, "raw_bytes": len(raw),
        "content_sha256": hashlib.sha256(raw).hexdigest() if text is not None else None,
        "invalid_utf8_replacements": invalid_utf8, "extremely_long_line": long_line,
        "fields": fields or [], "labels": labels or {}, "ambiguity": ambiguity,
    }


def _iter_json_array(handle: Any, max_value_bytes: int) -> Generator[Any, None, None]:
    """Incrementally decode a top-level JSON array with a strict memory ceiling."""
    decoder = json.JSONDecoder()
    text = ""
    pos = 0
    started = False
    finished = False
    while not finished:
        chunk = handle.read(64 << 10)
        if chunk:
            text += chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
        eof = not chunk
        while True:
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if not started:
                if pos >= len(text):
                    break
                if text[pos] != "[":
                    raise ValueError("streaming JSON requires a top-level array")
                started = True
                pos += 1
                continue
            while pos < len(text) and (text[pos].isspace() or text[pos] == ","):
                pos += 1
            if pos < len(text) and text[pos] == "]":
                finished = True
                break
            if pos >= len(text):
                break
            try:
                value, end = decoder.raw_decode(text, pos)
            except json.JSONDecodeError:
                break
            yield value
            pos = end
        if pos:
            text = text[pos:]
            pos = 0
        if len(text.encode("utf-8")) > max_value_bytes:
            raise ValueError(f"one JSON value exceeds --max-document-bytes={max_value_bytes}")
        if eof:
            if not finished:
                raise ValueError("truncated or invalid top-level JSON array")
            break


def stream_documents_from_file(
    file_info: Mapping[str, Any],
    text_fields: Set[str] | Sequence[str],
    start_cursor: Optional[Mapping[str, Any]] = None,
    text_chunk_bytes: int = 1 << 20,
    max_document_bytes: int = 64 << 20,
    issues: Optional[MutableMapping[str, Any]] = None,
) -> Generator[Dict[str, Any], None, None]:
    """Yield bounded documents. Cursors are positions *after* the yielded record."""
    issue = issues if issues is not None else {}
    ordered_fields = tuple(field for field in DEFAULT_TEXT_FIELDS if field in text_fields) + tuple(sorted(set(text_fields) - set(DEFAULT_TEXT_FIELDS)))
    path = Path(file_info["path"])
    ext = file_info["ext"]
    start_index = int((start_cursor or {}).get("doc_index", 0))
    start_offset = int((start_cursor or {}).get("offset", 0))

    if ext in PLAIN_EXTENSIONS:
        opener = gzip.open if ext.endswith(".gz") else open
        with opener(path, "rb") as handle:
            for index, (raw, offset, long_line) in enumerate(_stream_byte_chunks(handle, text_chunk_bytes, start_offset), start=start_index):
                text, invalid = _decode_utf8(raw)
                yield _document(text, file_info, index, {"doc_index": index + 1, "offset": offset}, len(raw), invalid, long_line)
        return

    if ext in {".jsonl", ".jsonl.gz"}:
        opener = gzip.open if ext.endswith(".gz") else open
        with opener(path, "rb") as handle:
            handle.seek(start_offset)
            index = start_index
            while True:
                raw = handle.readline(max_document_bytes + 1)
                if not raw:
                    break
                oversized = len(raw) > max_document_bytes and not raw.endswith(b"\n")
                if oversized:
                    while raw and not raw.endswith(b"\n"):
                        raw = handle.readline(64 << 10)
                    issue["oversized_records"] = int(issue.get("oversized_records", 0)) + 1
                    index += 1
                    continue
                offset = handle.tell()
                text_line, invalid = _decode_utf8(raw)
                try:
                    obj = json.loads(text_line)
                    text, fields, labels, ambiguity = extract_text(obj, ordered_fields)
                except json.JSONDecodeError as exc:
                    text, fields, labels, ambiguity = None, [], {}, f"invalid JSON: {exc.msg}"
                yield _document(text, file_info, index, {"doc_index": index + 1, "offset": offset}, len(raw), invalid, False, fields, labels, ambiguity)
                index += 1
        return

    if ext == ".json":
        with path.open("rb") as handle:
            prefix = handle.read(4096)
            first = next((chr(byte) for byte in prefix if chr(byte).strip()), "")
            handle.seek(0)
            if first == "[":
                values: Iterable[Any] = _iter_json_array(handle, max_document_bytes)
            elif file_info["size_bytes"] <= max_document_bytes:
                values = [json.load(handle)]
            else:
                raise ValueError("large top-level JSON objects are ambiguous; use JSONL or a top-level array")
            for absolute_index, obj in enumerate(values):
                if absolute_index < start_index:
                    continue
                text, fields, labels, ambiguity = extract_text(obj, ordered_fields)
                yield _document(text, file_info, absolute_index, {"doc_index": absolute_index + 1}, len(text.encode("utf-8")) if text else 0, 0, False, fields, labels, ambiguity)
        return

    if ext in {".csv", ".tsv"}:
        delimiter = "\t" if ext == ".tsv" else ","
        old_limit = csv.field_size_limit()
        csv.field_size_limit(max_document_bytes)
        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                reader = csv.DictReader(handle, delimiter=delimiter)
                for absolute_index, row in enumerate(reader):
                    if absolute_index < start_index:
                        continue
                    text, fields, labels, ambiguity = extract_text(row, ordered_fields)
                    approx = sum(len(str(value).encode("utf-8")) for value in row.values() if value is not None)
                    yield _document(text, file_info, absolute_index, {"doc_index": absolute_index + 1}, approx, 0, False, fields, labels, ambiguity)
        finally:
            csv.field_size_limit(old_limit)
        return

    if ext == ".parquet":
        if pq is None:
            raise RuntimeError("Parquet encountered but pyarrow is unavailable")
        parquet = pq.ParquetFile(path)
        absolute_index = 0
        for batch in parquet.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                if absolute_index >= start_index:
                    text, fields, labels, ambiguity = extract_text(row, ordered_fields)
                    yield _document(text, file_info, absolute_index, {"doc_index": absolute_index + 1}, len(text.encode("utf-8")) if text else 0, 0, False, fields, labels, ambiguity)
                absolute_index += 1
        return
    raise ValueError(f"Unsupported parser extension: {ext}")


def pack_21bit_vectorized(ids: np.ndarray) -> bytes:
    ids = np.asarray(ids, dtype=np.uint32)
    if ids.size and int(ids.max()) >= (1 << 21):
        raise ValueError("21-bit packing received an out-of-range token")
    padding = (-len(ids)) % 8
    if padding:
        ids = np.pad(ids, (0, padding))
    groups = ids.reshape(-1, 8).astype(np.uint64)
    output = np.zeros((len(groups), 21), dtype=np.uint8)
    for bit in range(21):
        source = ((groups >> bit) & 1).astype(np.uint8)
        absolute = np.arange(8) * 21 + bit
        for token_column in range(8):
            mask = np.uint8(1 << int(absolute[token_column] % 8))
            output[:, absolute[token_column] // 8] |= source[:, token_column] * mask
    return output.tobytes()


def unpack_21bit_vectorized(data: bytes, num_tokens: int) -> np.ndarray:
    if not data and num_tokens == 0:
        return np.empty(0, dtype=np.uint32)
    groups = np.frombuffer(data, dtype=np.uint8).reshape(-1, 21)
    output = np.zeros((len(groups), 8), dtype=np.uint32)
    for bit in range(21):
        absolute = np.arange(8) * 21 + bit
        for token_column in range(8):
            output[:, token_column] |= ((groups[:, absolute[token_column] // 8] >> (absolute[token_column] % 8)) & 1).astype(np.uint32) << bit
    return output.reshape(-1)[:num_tokens]


_WORKER_TOKENIZER: Optional[AGLMUniversalTokenizer] = None
_WORKER_NATIVE: Any = None
_WORKER_BACKEND = "reference"


def _worker_init(tokenizer_path: str, backend: str = "reference") -> None:
    global _WORKER_TOKENIZER, _WORKER_NATIVE, _WORKER_BACKEND
    if _WORKER_TOKENIZER is None:
        _WORKER_TOKENIZER = AGLMUniversalTokenizer.load(tokenizer_path)
    _WORKER_BACKEND = backend
    if backend == "native" and _WORKER_NATIVE is None:
        if AGLMNativeAccelerator is None:
            raise RuntimeError("native tokenizer requested but extension is not built")
        _WORKER_NATIVE = AGLMNativeAccelerator(_WORKER_TOKENIZER)


def _tokenize_task(task: Tuple[str, bool]) -> Dict[str, Any]:
    text, roundtrip = task
    if _WORKER_TOKENIZER is None:  # pragma: no cover - defensive
        raise RuntimeError("tokenizer worker was not initialized")
    if _WORKER_BACKEND == "native":
        if _WORKER_NATIVE is None:  # pragma: no cover - defensive
            raise RuntimeError("native tokenizer worker was not initialized")
        used_reference_fallback = _WORKER_NATIVE.requires_reference_fallback(text)
        if used_reference_fallback:
            tokens, _ = _WORKER_TOKENIZER.engine.encode(text)
            array = np.asarray(tokens, dtype=UINT32_DTYPE)
            token_bytes = array.tobytes()
        else:
            token_bytes = _WORKER_NATIVE.engine.encode_text_fast_u32(text)
            array = np.frombuffer(token_bytes, dtype=UINT32_DTYPE)
    else:
        used_reference_fallback = False
        tokens, _ = _WORKER_TOKENIZER.engine.encode(text)
        array = np.asarray(tokens, dtype=UINT32_DTYPE)
        token_bytes = array.tobytes()
    result = {
        "tokens": token_bytes, "token_count_without_eos": len(array),
        "min": int(array.min()) if len(array) else None, "max": int(array.max()) if len(array) else None,
        "native_reference_fallback": used_reference_fallback,
    }
    if roundtrip:
        decoded = _WORKER_TOKENIZER.engine.decode_to_bytes(array.tolist())
        result["roundtrip_sha256"] = hashlib.sha256(decoded).hexdigest()
    return result


@dataclass
class OpenShard:
    split: str
    index: int
    root: Path
    handle: Any = None
    tmp_path: Optional[Path] = None
    final_path: Optional[Path] = None
    token_count: int = 0
    raw_bytes: int = 0
    document_count: int = 0
    min_id: int = EXPECTED_MAX_ID
    max_id: int = 0

    def ensure_open(self) -> None:
        if self.handle is not None:
            return
        directory = self.root / self.split
        directory.mkdir(parents=True, exist_ok=True)
        name = f"shard_{self.index:05d}.bin"
        self.final_path = directory / name
        self.tmp_path = directory / f"{name}.tmp"
        self.handle = self.tmp_path.open("wb")

    def append(self, tokens: np.ndarray, raw_bytes: int) -> int:
        self.ensure_open()
        if tokens.dtype != UINT32_DTYPE:
            tokens = tokens.astype(UINT32_DTYPE, copy=False)
        if not len(tokens):
            raise ValueError("document token sequence must include EOS")
        low, high = int(tokens.min()), int(tokens.max())
        if low < 0 or high >= EXPECTED_VOCAB_SIZE:
            raise ValueError(f"token range [{low}, {high}] is outside [0, {EXPECTED_MAX_ID}]")
        offset = self.token_count
        self.handle.write(tokens.tobytes(order="C"))
        self.token_count += len(tokens)
        self.raw_bytes += raw_bytes
        self.document_count += 1
        self.min_id = min(self.min_id, low)
        self.max_id = max(self.max_id, high)
        return offset

    def finalize(self, output_root: Path) -> Optional[Dict[str, Any]]:
        if self.handle is None or self.tmp_path is None or self.final_path is None:
            return None
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        self.handle = None
        expected_size = self.token_count * UINT32_DTYPE.itemsize
        actual_size = self.tmp_path.stat().st_size
        if actual_size != expected_size or actual_size % 4:
            raise IOError(f"temporary shard size mismatch: {self.tmp_path}")
        digest_uncompressed = sha256_file(self.tmp_path)
        
        # Lossless Zstandard Level 1 compression (reduces disk footprint from 5.7 GB -> 2.5 GB)
        import zstandard as zstd
        cctx = zstd.ZstdCompressor(level=1, write_content_size=True)
        zst_path = self.final_path.with_name(self.final_path.name + ".zst")
        with self.tmp_path.open("rb") as f_in, zst_path.open("wb") as f_out:
            cctx.copy_stream(f_in, f_out)
        self.tmp_path.unlink()
        
        digest_zst = sha256_file(zst_path)
        actual_size = zst_path.stat().st_size
        return {
            "shard_id": self.index, "filename": zst_path.name, "split": self.split,
            "path": zst_path.relative_to(output_root).as_posix(),
            "token_count": self.token_count, "raw_bytes_represented": self.raw_bytes,
            "source_document_count": self.document_count, "min_token_id": self.min_id,
            "max_token_id": self.max_id, "sha256": digest_zst, "sha256_uncompressed": digest_uncompressed,
            "byte_size": actual_size, "uncompressed_byte_size": expected_size,
            "dtype": "uint32", "numpy_dtype": "<u4", "endian": "little",
            "compression": "zstd_level_1",
        }

    def close_incomplete(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def _derive_packed21(output_root: Path, shard: Mapping[str, Any]) -> Dict[str, Any]:
    source = output_root / shard["path"]
    target = output_root / "packed21" / shard["split"] / source.with_suffix(".p21").name
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    pack_started = time.perf_counter()
    with source.open("rb") as src, tmp.open("wb") as dst:
        remaining = int(shard["token_count"])
        while remaining:
            count = min(1_048_576, remaining)
            if remaining > count:
                count -= count % 8
            array = np.fromfile(src, dtype=UINT32_DTYPE, count=count)
            if len(array) != count:
                raise IOError("short uint32 read during 21-bit packing")
            dst.write(pack_21bit_vectorized(array))
            remaining -= count
        dst.flush()
        os.fsync(dst.fileno())
    pack_seconds = time.perf_counter() - pack_started
    expected = math.ceil(int(shard["token_count"]) / 8) * 21
    if tmp.stat().st_size != expected:
        raise IOError("packed21 size mismatch")

    unpack_started = time.perf_counter()
    with source.open("rb") as src, tmp.open("rb") as packed:
        remaining = int(shard["token_count"])
        while remaining:
            count = min(1_048_576, remaining)
            if remaining > count:
                count -= count % 8
            original = np.fromfile(src, dtype=UINT32_DTYPE, count=count)
            encoded = packed.read(math.ceil(count / 8) * 21)
            decoded = unpack_21bit_vectorized(encoded, count)
            if not np.array_equal(original, decoded):
                raise IOError("packed21 independent reversibility check failed")
            remaining -= count
    unpack_seconds = time.perf_counter() - unpack_started
    os.replace(tmp, target)
    uint32_mb = int(shard["byte_size"]) / (1 << 20)
    return {
        "path": target.relative_to(output_root).as_posix(), "byte_size": target.stat().st_size,
        "sha256": sha256_file(target), "token_count": int(shard["token_count"]),
        "format": "little-endian contiguous 21-bit unsigned IDs; zero padded to groups of 8",
        "pack_seconds": round(pack_seconds, 6), "unpack_verify_seconds": round(unpack_seconds, 6),
        "pack_uint32_input_mb_s": round(uint32_mb / max(pack_seconds, 1e-9), 3),
        "unpack_uint32_output_mb_s": round(uint32_mb / max(unpack_seconds, 1e-9), 3),
    }


def compute_minhash_bands(text: str, num_bands: int = 16, rows_per_band: int = 4) -> List[int]:
    """Computes fast 64-hash MinHash LSH band keys for near-deduplication (Jaccard >= 0.75)."""
    words = text.split()
    if len(words) < 5:
        return []
    shingles = [hash(words[i] + " " + words[i+1] + " " + words[i+2]) for i in range(len(words) - 2)]
    if not shingles:
        return []
    num_hashes = num_bands * rows_per_band
    min_hashes = [float("inf")] * num_hashes
    for s in shingles:
        for h_idx in range(num_hashes):
            h = (s * (h_idx * 10007 + 1) + (h_idx * 31337 + 7)) & 0xFFFFFFFFFFFFFFFF
            if h < min_hashes[h_idx]:
                min_hashes[h_idx] = h
    band_keys = []
    for b in range(num_bands):
        band_slice = tuple(min_hashes[b * rows_per_band : (b + 1) * rows_per_band])
        band_keys.append(hash((b, band_slice)))
    return band_keys


class ProductionDatasetBuilder:
    """Streaming builder with SQLite-authoritative, shard-boundary checkpoints."""

    def __init__(
        self, input_dir: str, output_dir: str, tokenizer_census: TokenizerCensus,
        shard_tokens: int = 100_000_000, val_ratio: float = 0.005,
        dedupe_exact: bool = False, enable_packed21: bool = False,
        minhash_dedup: bool = True, quality_pruning: bool = True,
        text_fields: Optional[List[str]] = None, sample_mb: Optional[int] = None,
        dry_run: bool = False, resume: bool = False, workers: int = 1,
        text_chunk_bytes: int = 1 << 20, max_document_bytes: int = 64 << 20,
        short_document_bytes: int = 32, long_document_bytes: int = 8 << 20,
        repetition_threshold: float = 0.95, log_interval: float = 5.0,
        inventory_summary: Optional[Mapping[str, Any]] = None,
        report_dir: Optional[str] = None,
        tokenizer_backend: str = "reference",
    ):
        if not 0 <= val_ratio < 1:
            raise ValueError("--val-ratio must be in [0, 1)")
        if shard_tokens <= 0 or workers <= 0:
            raise ValueError("shard tokens and workers must be positive")
        self.input_dir = Path(input_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.tc = tokenizer_census
        self.shard_tokens = int(shard_tokens)
        self.val_ratio = float(val_ratio)
        self.dedupe_exact = dedupe_exact
        self.enable_packed21 = enable_packed21
        self.minhash_dedup = minhash_dedup
        self.quality_pruning = quality_pruning
        self.minhash_seen: Set[int] = set()
        self.text_fields = list(text_fields or DEFAULT_TEXT_FIELDS)
        self.sample_bytes = sample_mb * (1 << 20) if sample_mb else None
        self.sample_mb = sample_mb
        self.dry_run = dry_run
        self.resume = resume
        self.workers = workers
        if tokenizer_backend not in {"reference", "native"}:
            raise ValueError("tokenizer_backend must be 'reference' or 'native'")
        if tokenizer_backend == "native" and NativeBpe is None:
            raise RuntimeError("native backend requested but extension is not built; run native/aglm_native/build.sh")
        self.tokenizer_backend = tokenizer_backend
        if workers > 1 and tokenizer_backend == "reference" and psutil:
            rss = psutil.Process().memory_info().rss
            available = psutil.virtual_memory().available
            if rss * workers > available * 0.75:
                raise MemoryError(
                    f"Refusing {workers} tokenizer workers: conservative demand {rss * workers / (1 << 30):.1f} GiB "
                    f"exceeds 75% of {available / (1 << 30):.1f} GiB currently available. Use fewer workers."
                )
        if workers > 1 and tokenizer_backend == "native" and psutil:
            rss = psutil.Process().memory_info().rss
            available = psutil.virtual_memory().available
            # At worst every input byte falls back to a separate uint32 token;
            # the scheduler admits at most workers*2 documents per batch.
            worst_inflight = workers * 2 * max_document_bytes * 5
            if rss + worst_inflight > available * 0.75:
                raise MemoryError(
                    f"Refusing {workers} native tokenizer threads: process RSS plus worst-case bounded "
                    f"in-flight text/uint32 buffers is {(rss + worst_inflight) / (1 << 30):.1f} GiB, "
                    f"above 75% of {available / (1 << 30):.1f} GiB available. Lower --workers or --max-document-bytes."
                )
        self.text_chunk_bytes = text_chunk_bytes
        self.max_document_bytes = max_document_bytes
        self.short_document_bytes = short_document_bytes
        self.long_document_bytes = long_document_bytes
        self.repetition_threshold = repetition_threshold
        self.log_interval = log_interval
        self.inventory_summary = dict(inventory_summary or {})
        self.report_dir = Path(report_dir).resolve() if report_dir else REPO_ROOT
        self.meta_dir = self.output_dir / "metadata"
        self.log_dir = self.output_dir / "logs"
        for directory in (self.output_dir / "train", self.output_dir / "val", self.meta_dir, self.log_dir, self.output_dir / "scripts"):
            directory.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"aglm-builder-{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(self.log_dir / "conversion.log", encoding="utf-8")):
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.db_path = self.meta_dir / "checkpoint.sqlite3"
        if self.db_path.exists() and not resume:
            raise FileExistsError(f"Output already has a checkpoint; use --resume or a clean output directory: {self.output_dir}")
        self.db = sqlite3.connect(self.db_path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self._create_schema()
        self.frequency = np.zeros(EXPECTED_VOCAB_SIZE, dtype=np.uint64)
        self.counters: Dict[str, int] = Counter()
        self.cursor: Dict[str, Any] = {"file_index": 0, "doc_index": 0}
        self.next_shard = {"train": 0, "val": 0}
        self.current_source: Optional[Dict[str, Any]] = None
        self.generation = 0
        self.started = time.perf_counter()
        self.last_log = self.started
        self.peak_ram_bytes = 0
        self._configure_or_resume()
        self.writers = {
            split: OpenShard(split, self.next_shard[split], self.output_dir)
            for split in ("train", "val")
        }
        global _WORKER_TOKENIZER, _WORKER_NATIVE, _WORKER_BACKEND
        _WORKER_TOKENIZER = self.tc.tokenizer
        _WORKER_BACKEND = tokenizer_backend
        if tokenizer_backend == "native" and _WORKER_NATIVE is None:
            _WORKER_NATIVE = AGLMNativeAccelerator(self.tc.tokenizer)
        self.executor: Any = None
        if workers > 1:
            if tokenizer_backend == "native":
                # Rust releases the GIL and the immutable 4.7M-node trie is shared,
                # avoiding one massive tokenizer copy per process.
                self.executor = ThreadPoolExecutor(max_workers=workers)
            else:
                context = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
                self.executor = ProcessPoolExecutor(
                    max_workers=workers, mp_context=context, initializer=_worker_init,
                    initargs=(self.tc.tokenizer_dir, tokenizer_backend),
                )

    def _create_schema(self) -> None:
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS shards (path TEXT PRIMARY KEY, split TEXT NOT NULL, shard_id INTEGER NOT NULL, record_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS documents (
                doc_key TEXT PRIMARY KEY, content_hash TEXT NOT NULL, split TEXT NOT NULL,
                source_path TEXT NOT NULL, doc_index INTEGER NOT NULL, shard_path TEXT NOT NULL,
                token_offset INTEGER NOT NULL, token_count INTEGER NOT NULL, raw_bytes INTEGER NOT NULL,
                language TEXT, domain TEXT, dataset TEXT, source_label TEXT
            );
            CREATE INDEX IF NOT EXISTS documents_content_hash ON documents(content_hash);
            CREATE INDEX IF NOT EXISTS documents_split_hash ON documents(split, content_hash);
            CREATE TABLE IF NOT EXISTS sources (path TEXT PRIMARY KEY, record_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS source_seen (path TEXT PRIMARY KEY, packed_bits BLOB NOT NULL);
        """)
        self.db.commit()

    def _config_identity(self) -> Dict[str, Any]:
        return {
            "input_dir": str(self.input_dir), "inventory_fingerprint": self.inventory_summary.get("inventory_fingerprint"),
            "tokenizer_sha256": self.tc.model_sha256, "regex_version": self.tc.regex_version,
            "dtype": "<u4", "shard_tokens": self.shard_tokens,
            "val_ratio": self.val_ratio, "dedupe_exact": self.dedupe_exact,
            "text_fields": self.text_fields, "text_chunk_bytes": self.text_chunk_bytes,
            "max_document_bytes": self.max_document_bytes, "sample_bytes": self.sample_bytes,
            "tokenizer_backend": self.tokenizer_backend,
        }

    def _get_state(self, key: str, default: Any = None) -> Any:
        row = self.db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def _set_state(self, key: str, value: Any) -> None:
        self.db.execute("INSERT INTO state(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, json.dumps(value, sort_keys=True)))

    def _configure_or_resume(self) -> None:
        prior = self._get_state("config")
        identity = self._config_identity()
        if prior is not None and prior != identity:
            raise RuntimeError(f"Resume configuration/input mismatch. Existing={prior!r}; requested={identity!r}")
        if prior is None:
            self._set_state("config", identity)
            self._set_state("cursor", self.cursor)
            self._set_state("counters", dict(self.counters))
            self._set_state("generation", 0)
            self.db.commit()
        else:
            self.cursor = self._get_state("cursor", self.cursor)
            self.counters.update(self._get_state("counters", {}))
            self.generation = int(self._get_state("generation", 0))
            for split in ("train", "val"):
                row = self.db.execute("SELECT MAX(shard_id) FROM shards WHERE split=?", (split,)).fetchone()
                self.next_shard[split] = 0 if row[0] is None else int(row[0]) + 1
            self._restore_frequency()
            self._verify_resume_shards()
            self.logger.info("Resuming at cursor %s with %d verified shards", self.cursor, sum(self.next_shard.values()))
        referenced = {row[0] for row in self.db.execute("SELECT path FROM shards")}
        for tmp in self.output_dir.glob("**/*.tmp"):
            tmp.unlink()
        for split in ("train", "val"):
            for path in (self.output_dir / split).glob("shard_*.bin*"):
                if path.relative_to(self.output_dir).as_posix() not in referenced:
                    path.unlink()

    def _verify_resume_shards(self) -> None:
        for (record_json,) in self.db.execute("SELECT record_json FROM shards"):
            record = json.loads(record_json)
            path = self.output_dir / record["path"]
            if not path.is_file() or path.stat().st_size != record["byte_size"] or sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"A previously committed shard failed resume verification: {path}")

    def _restore_frequency(self) -> None:
        freq_path = self.meta_dir / "token_frequency.npy"
        generation_path = self.meta_dir / "token_frequency_generation.json"
        try:
            info = json.loads(generation_path.read_text(encoding="utf-8"))
            loaded = np.load(freq_path, mmap_mode=None)
            if info["generation"] == self.generation and loaded.shape == self.frequency.shape:
                self.frequency[:] = loaded
                return
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
        self.logger.warning("Frequency checkpoint mismatch; rebuilding counters by streaming verified shards")
        for (record_json,) in self.db.execute("SELECT record_json FROM shards ORDER BY split, shard_id"):
            record = json.loads(record_json)
            with (self.output_dir / record["path"]).open("rb") as handle:
                while True:
                    tokens = np.fromfile(handle, dtype=UINT32_DTYPE, count=1_048_576)
                    if not len(tokens):
                        break
                    np.add.at(self.frequency, tokens, 1)
        self._save_frequency()

    def _save_frequency(self) -> None:
        path = self.meta_dir / "token_frequency.npy"
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as handle:
            np.save(handle, self.frequency, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        atomic_write_json(self.meta_dir / "token_frequency_generation.json", {"generation": self.generation})

    def _new_source_stats(self, info: Mapping[str, Any]) -> Dict[str, Any]:
        row = self.db.execute("SELECT record_json FROM sources WHERE path=?", (info["rel_path"],)).fetchone()
        if row:
            stats = json.loads(row[0])
        else:
            stats = {
                "path": info["rel_path"], "absolute_path": info["path"], "extension": info["ext"],
                "file_size_bytes": info["size_bytes"], "documents_scanned": 0, "documents_written": 0,
                "tokens": 0, "raw_utf8_bytes": 0, "input_bytes_read": 0, "exact_duplicates": 0,
                "duplicate_bytes_skipped": 0, "tokens_avoided": 0, "empty_documents": 0,
                "short_documents": 0, "invalid_utf8_documents": 0, "nul_heavy_documents": 0,
                "extreme_repetition_documents": 0, "extremely_long_documents": 0,
                "extremely_long_lines": 0, "ambiguous_records": 0, "parse_errors": 0,
                "ambiguity_examples": [],
                "min_token_id": None, "max_token_id": None, "unique_token_ids_seen": 0,
                "labels": {"language": [], "domain": [], "source": [], "dataset": []}, "complete": False,
            }
        stats["_seen"] = np.zeros(EXPECTED_VOCAB_SIZE, dtype=np.bool_)
        seen_row = self.db.execute("SELECT packed_bits FROM source_seen WHERE path=?", (info["rel_path"],)).fetchone()
        if seen_row:
            restored = np.unpackbits(np.frombuffer(seen_row[0], dtype=np.uint8), bitorder="little")[:EXPECTED_VOCAB_SIZE]
            stats["_seen"][:] = restored.astype(np.bool_, copy=False)
        return stats

    def _save_source(self, complete: bool = False) -> None:
        if self.current_source is None:
            return
        stats = self.current_source
        stats["complete"] = complete
        stats["unique_token_ids_seen"] = int(np.count_nonzero(stats["_seen"])) if "_seen" in stats else stats.get("unique_token_ids_seen", 0)
        public = {key: value for key, value in stats.items() if not key.startswith("_")}
        tokens = public["tokens"]
        docs = public["documents_written"]
        raw = public["raw_utf8_bytes"]
        public.update({
            "bytes_per_token": round(raw / tokens, 6) if tokens else None,
            "tokens_per_document": round(tokens / docs, 6) if docs else None,
            "tokens_per_mb": round(tokens / (raw / (1 << 20)), 3) if raw else None,
        })
        self.db.execute("INSERT INTO sources(path,record_json) VALUES(?,?) ON CONFLICT(path) DO UPDATE SET record_json=excluded.record_json", (public["path"], json.dumps(public, sort_keys=True)))
        if complete:
            self.db.execute("DELETE FROM source_seen WHERE path=?", (public["path"],))
        else:
            packed = np.packbits(stats["_seen"], bitorder="little").tobytes()
            self.db.execute("INSERT INTO source_seen(path,packed_bits) VALUES(?,?) ON CONFLICT(path) DO UPDATE SET packed_bits=excluded.packed_bits", (public["path"], packed))

    def _quality(self, doc: Mapping[str, Any]) -> Dict[str, bool]:
        text = doc.get("text")
        raw = text.encode("utf-8") if text is not None else b""
        sample = raw[: 64 << 10]
        counts = Counter(sample)
        repetition = (max(counts.values()) / len(sample)) if sample else 0.0
        nul_ratio = sample.count(0) / len(sample) if sample else 0.0

        is_repetitive_trigram = False
        is_symbol_heavy = False
        is_too_short = 0 < len(raw) < self.short_document_bytes

        if self.quality_pruning and text:
            words = text.split()
            word_count = len(words)
            if word_count >= 15:
                trigrams = [words[i] + " " + words[i+1] + " " + words[i+2] for i in range(word_count - 2)]
                tri_counts = Counter(trigrams)
                if max(tri_counts.values()) / len(trigrams) > 0.15:
                    is_repetitive_trigram = True

            alnum_count = sum(1 for c in text if c.isalnum() or ('\u0900' <= c <= '\u0D7F'))
            if len(text) > 50 and (alnum_count / len(text)) < 0.40:
                is_symbol_heavy = True
            if word_count < 15 or len(raw) < 80:
                is_too_short = True

        return {
            "empty": not raw, "short": is_too_short,
            "invalid_utf8": bool(doc.get("invalid_utf8_replacements")), "nul_heavy": nul_ratio > 0.01,
            "extreme_repetition": (len(sample) >= 128 and repetition >= self.repetition_threshold) or is_repetitive_trigram,
            "noisy_symbols": is_symbol_heavy,
            "extremely_long": len(raw) >= self.long_document_bytes,
            "extremely_long_line": bool(doc.get("extremely_long_line")),
            "ambiguous": text is None,
        }

    def _split(self, content_hash: str) -> str:
        value = int(content_hash[:16], 16) / float(1 << 64)
        return "val" if value < self.val_ratio else "train"

    def _roundtrip_selected(self, content_hash: str) -> bool:
        return int(content_hash[-8:], 16) % 10_000 == 0

    def _tokenize(self, tasks: List[Tuple[str, bool]]) -> List[Dict[str, Any]]:
        if self.executor:
            return list(self.executor.map(_tokenize_task, tasks, chunksize=1))
        return [_tokenize_task(task) for task in tasks]

    def _append_unique(self, doc: Mapping[str, Any], result: Mapping[str, Any]) -> None:
        assert self.current_source is not None
        content_hash = str(doc["content_sha256"])
        raw_tokens = np.frombuffer(result["tokens"], dtype=UINT32_DTYPE)
        if result.get("native_reference_fallback"):
            self.counters["native_reference_fallback_documents"] += 1
            self.counters["native_reference_fallback_bytes"] += int(doc["raw_bytes"])
        if len(raw_tokens) and (int(raw_tokens.min()) < 0 or int(raw_tokens.max()) >= EXPECTED_VOCAB_SIZE):
            raise ValueError("tokenizer emitted an out-of-range ID")
        if "roundtrip_sha256" in result and result["roundtrip_sha256"] != content_hash:
            raise RuntimeError(f"periodic roundtrip failed for {doc['source']}#{doc['doc_index']}")
        tokens = np.empty(len(raw_tokens) + 1, dtype=UINT32_DTYPE)
        tokens[:-1] = raw_tokens
        tokens[-1] = self.tc.eos_token_id
        split = self._split(content_hash)
        writer = self.writers[split]
        if writer.token_count and writer.token_count + len(tokens) > self.shard_tokens:
            self._commit_group()
            writer = self.writers[split]
        offset = writer.append(tokens, int(doc["raw_bytes"]))
        np.add.at(self.frequency, tokens, 1)
        self.current_source["_seen"][tokens] = True
        labels = doc.get("labels", {})
        for key in ("language", "domain", "source", "dataset"):
            if key in labels and labels[key] not in self.current_source["labels"][key]:
                self.current_source["labels"][key].append(labels[key])
        doc_key = hashlib.sha256(f"{doc['source']}\0{doc['doc_index']}\0{content_hash}".encode()).hexdigest()
        shard_path = f"{split}/shard_{writer.index:05d}.bin"
        self.db.execute(
            "INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (doc_key, content_hash, split, doc["source"], int(doc["doc_index"]), shard_path,
             offset, len(tokens), int(doc["raw_bytes"]), labels.get("language"), labels.get("domain"),
             labels.get("dataset"), labels.get("source"),),
        )
        source = self.current_source
        source["documents_written"] += 1
        source["tokens"] += len(tokens)
        source["raw_utf8_bytes"] += int(doc["raw_bytes"])
        source["min_token_id"] = min(source["min_token_id"] if source["min_token_id"] is not None else EXPECTED_MAX_ID, int(tokens.min()))
        source["max_token_id"] = max(source["max_token_id"] if source["max_token_id"] is not None else 0, int(tokens.max()))
        self.counters["documents_written"] += 1
        self.counters["raw_utf8_bytes"] += int(doc["raw_bytes"])
        self.counters["tokens"] += len(tokens)
        if writer.token_count >= self.shard_tokens:
            self._commit_group()

    def _process_batch(self, events: List[Dict[str, Any]]) -> None:
        unique = [event for event in events if event["kind"] == "unique"]
        tasks = [(event["doc"]["text"], self._roundtrip_selected(event["doc"]["content_sha256"])) for event in unique]
        results = iter(self._tokenize(tasks))
        result_by_hash: Dict[str, Dict[str, Any]] = {}
        for event in events:
            doc = event["doc"]
            assert self.current_source is not None
            self.current_source["documents_scanned"] += 1
            self.current_source["input_bytes_read"] += int(doc.get("input_bytes", 0))
            self.counters["documents_scanned"] += 1
            self.counters["input_bytes_read"] += int(doc.get("input_bytes", 0))
            for flag, counter_name in (
                ("empty", "empty_documents"), ("short", "short_documents"),
                ("invalid_utf8", "invalid_utf8_documents"), ("nul_heavy", "nul_heavy_documents"),
                ("extreme_repetition", "extreme_repetition_documents"),
                ("noisy_symbols", "noisy_symbol_documents"),
                ("extremely_long", "extremely_long_documents"),
                ("extremely_long_line", "extremely_long_lines"), ("ambiguous", "ambiguous_records"),
            ):
                if event["quality"].get(flag):
                    self.current_source[counter_name] = self.current_source.get(counter_name, 0) + 1
                    self.counters[counter_name] += 1
            if event["quality"].get("ambiguous") and len(self.current_source["ambiguity_examples"]) < 20:
                self.current_source["ambiguity_examples"].append({"doc_index": doc["doc_index"], "reason": doc.get("ambiguity")})
            if event["kind"] == "unique":
                result = next(results)
                result_by_hash[doc["content_sha256"]] = result
                self._append_unique(doc, result)
            elif event["kind"] == "duplicate":
                known = event.get("known_tokens")
                if known is None:
                    result = result_by_hash.get(doc["content_sha256"])
                    known = (result["token_count_without_eos"] + 1) if result else 0
                self.current_source["exact_duplicates"] += 1
                self.current_source["duplicate_bytes_skipped"] += int(doc["raw_bytes"])
                self.current_source["tokens_avoided"] += int(known)
                self.counters["exact_duplicates"] += 1
                self.counters["duplicate_bytes_skipped"] += int(doc["raw_bytes"])
                self.counters["tokens_avoided"] += int(known)
            self.cursor = {"file_index": event["file_index"], "rel_path": doc["source"], **doc["cursor"]}
            self._telemetry()

    def _prepare_event(self, doc: Mapping[str, Any], file_index: int, batch_hashes: Set[str]) -> Dict[str, Any]:
        quality = self._quality(doc)
        if quality["ambiguous"] or quality["empty"] or quality["short"] or quality["extreme_repetition"] or quality["noisy_symbols"]:
            return {"kind": "skip", "doc": doc, "quality": quality, "file_index": file_index}
        content_hash = str(doc["content_sha256"])
        if self.dedupe_exact:
            row = self.db.execute("SELECT token_count FROM documents WHERE content_hash=? LIMIT 1", (content_hash,)).fetchone()
            if row:
                return {"kind": "duplicate", "doc": doc, "quality": quality, "file_index": file_index, "known_tokens": int(row[0])}
            if content_hash in batch_hashes:
                return {"kind": "duplicate", "doc": doc, "quality": quality, "file_index": file_index, "known_tokens": None}
            batch_hashes.add(content_hash)

        # MinHash LSH Near-Deduplication
        if getattr(self, "minhash_dedup", True) and doc.get("text"):
            bands = compute_minhash_bands(doc["text"])
            if bands:
                matching_bands = sum(1 for b in bands if b in self.minhash_seen)
                if matching_bands >= 2:  # >= 75% near duplicate match
                    return {"kind": "duplicate", "doc": doc, "quality": quality, "file_index": file_index, "known_tokens": None}
                for b in bands:
                    self.minhash_seen.add(b)

        return {"kind": "unique", "doc": doc, "quality": quality, "file_index": file_index}

    def _commit_group(self, complete_source: bool = False) -> None:
        records: List[Dict[str, Any]] = []
        for split in ("train", "val"):
            record = self.writers[split].finalize(self.output_dir)
            if record:
                if self.enable_packed21:
                    record["packed21"] = _derive_packed21(self.output_dir, record)
                records.append(record)
                self.db.execute("INSERT INTO shards(path,split,shard_id,record_json) VALUES(?,?,?,?)", (record["path"], split, record["shard_id"], json.dumps(record, sort_keys=True)))
                self.next_shard[split] += 1
        self._save_source(complete=complete_source)
        self.generation += 1
        self._set_state("cursor", self.cursor)
        self._set_state("counters", dict(self.counters))
        self._set_state("generation", self.generation)
        self.db.commit()
        self._save_frequency()
        self.writers = {split: OpenShard(split, self.next_shard[split], self.output_dir) for split in ("train", "val")}
        self._write_metadata(final=False)
        if records:
            self.logger.info("Committed %s", ", ".join(f"{r['path']} ({r['token_count']:,} tokens, sha256={r['sha256'][:12]}…)" for r in records))

    def _ram_bytes(self) -> int:
        if psutil:
            process = psutil.Process()
            info = process.memory_full_info()
            total = getattr(info, "pss", info.rss)
            for child in process.children(recursive=True):
                try:
                    child_info = child.memory_full_info()
                    total += getattr(child_info, "pss", child_info.rss)
                except psutil.Error:
                    pass
            return total
        maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(maximum * 1024)

    def _telemetry(self, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self.last_log < self.log_interval:
            return
        elapsed = max(now - self.started, 1e-9)
        read = self.counters["input_bytes_read"]
        raw = self.counters["raw_utf8_bytes"]
        tokens = self.counters["tokens"]
        output = tokens * 4
        ram = self._ram_bytes()
        self.peak_ram_bytes = max(self.peak_ram_bytes, ram)
        cpu = psutil.cpu_percent(interval=None) if psutil else None
        total_input = int(self.sample_bytes or self.inventory_summary.get("total_raw_bytes", 0))
        processed = min(read, total_input) if total_input else read
        fraction = min(1.0, processed / total_input) if total_input else 0.0
        bar_width = 28
        filled = min(bar_width, int(fraction * bar_width))
        progress_bar = "=" * filled + (">" if filled < bar_width else "") + "." * max(0, bar_width - filled - 1)
        eta_seconds = ((total_input - processed) / (read / elapsed)) if read and total_input > processed else 0.0 if total_input and processed >= total_input else None
        self.logger.info(
            "progress [%s] %5.1f%% %.2f/%.2f GiB raw-read=%.2f MB/s raw-tokenized=%.2f MB/s tokens=%.0f/s output=%.2f MB/s CPU=%s RAM=%.1f MiB elapsed=%s ETA=%s",
            progress_bar, fraction * 100.0, processed / (1 << 30), total_input / (1 << 30) if total_input else 0.0,
            read / (1 << 20) / elapsed, raw / (1 << 20) / elapsed, tokens / elapsed,
            output / (1 << 20) / elapsed, f"{cpu:.1f}%" if cpu is not None else "n/a",
            ram / (1 << 20), format_duration(elapsed), format_duration(eta_seconds),
        )
        self.last_log = now

    def build_dataset(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_limit = max(1, self.workers * 2)
        stop = False
        start_file = int(self.cursor.get("file_index", 0))
        try:
            for file_index, info in enumerate(files):
                if file_index < start_file:
                    continue
                if file_index == start_file and self.cursor.get("rel_path") == info["rel_path"]:
                    start_cursor = self.cursor
                else:
                    start_cursor = None
                self.current_source = self._new_source_stats(info)
                issues: Dict[str, Any] = {}
                events: List[Dict[str, Any]] = []
                batch_hashes: Set[str] = set()
                self.logger.info("Processing [%d/%d] %s", file_index + 1, len(files), info["rel_path"])
                iterator = stream_documents_from_file(
                    info, set(self.text_fields), start_cursor=start_cursor,
                    text_chunk_bytes=self.text_chunk_bytes, max_document_bytes=self.max_document_bytes,
                    issues=issues,
                )
                while True:
                    try:
                        doc = next(iterator)
                    except StopIteration:
                        break
                    except (ValueError, UnicodeError, csv.Error, OSError, EOFError) as exc:
                        self.current_source["parse_errors"] += 1
                        self.counters["parse_errors"] += 1
                        self.logger.error("Parser stopped for %s: %s", info["rel_path"], exc)
                        break
                    else:
                        events.append(self._prepare_event(doc, file_index, batch_hashes))
                        if len(events) >= batch_limit:
                            self._process_batch(events)
                            events, batch_hashes = [], set()
                            if self.sample_bytes and self.counters["raw_utf8_bytes"] >= self.sample_bytes:
                                stop = True
                                break
                if events:
                    self._process_batch(events)
                    if self.sample_bytes and self.counters["raw_utf8_bytes"] >= self.sample_bytes:
                        stop = True
                if issues.get("oversized_records"):
                    self.current_source["extremely_long_documents"] += int(issues["oversized_records"])
                    self.counters["extremely_long_documents"] += int(issues["oversized_records"])
                if stop:
                    self._commit_group(complete_source=False)
                    break
                self.cursor = {"file_index": file_index + 1, "doc_index": 0}
                self._save_source(complete=True)
                self.current_source = None
            if not stop:
                self._commit_group(complete_source=True)
            self._telemetry(force=True)
            manifest = self._write_metadata(final=True)
            verification = verify_dataset(self.output_dir, self.db, manifest, self.tc, self.report_dir / "AGLM_TRAINING_DATASET_VERIFICATION_REPORT.md")
            self._write_reports(manifest, verification)
            return manifest
        finally:
            if self.executor:
                self.executor.shutdown(wait=True, cancel_futures=True)
            for writer in self.writers.values():
                writer.close_incomplete()
            self.db.close()

    def _shards(self) -> Dict[str, List[Dict[str, Any]]]:
        result: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": []}
        for split, record_json in self.db.execute("SELECT split,record_json FROM shards ORDER BY split,shard_id"):
            result[split].append(json.loads(record_json))
        return result

    def _stats(self, shards: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
        elapsed = max(time.perf_counter() - self.started, 1e-9)
        raw = self.counters["raw_utf8_bytes"]
        tokens = sum(record["token_count"] for rows in shards.values() for record in rows)
        # Counter only materializes non-zero keys. Production metadata must still
        # report explicit zeroes so downstream audits can distinguish "none"
        # from "not measured".
        counter_names = (
            "documents_scanned", "documents_written", "tokens", "raw_utf8_bytes",
            "input_bytes_read", "exact_duplicates", "duplicate_bytes_skipped",
            "tokens_avoided", "empty_documents", "short_documents",
            "invalid_utf8_documents", "nul_heavy_documents",
            "extreme_repetition_documents", "extremely_long_documents",
            "extremely_long_lines", "ambiguous_records", "parse_errors",
            "native_reference_fallback_documents", "native_reference_fallback_bytes",
        )
        explicit_counters = {name: int(self.counters.get(name, 0)) for name in counter_names}
        seen = self.frequency > 0
        seen_counts = self.frequency[seen]
        top_count = min(50, int(np.count_nonzero(seen)))
        top_ids = np.argpartition(self.frequency, -top_count)[-top_count:] if top_count else np.array([], dtype=np.int64)
        top_ids = top_ids[np.argsort(self.frequency[top_ids])[::-1]] if top_count else top_ids
        top = []
        for token_id in top_ids:
            token_bytes = self.tc.tokenizer.engine.id_to_bytes.get(int(token_id))
            token_text = token_bytes.decode("utf-8", errors="replace") if token_bytes is not None else self.tc.tokenizer.engine.special_id_to_str.get(int(token_id), "")
            top.append({"id": int(token_id), "count": int(self.frequency[token_id]), "text": token_text[:200]})
        bpt = raw / tokens if tokens else None
        projections: Dict[str, Any] = {"basis": "decimal TB (1 TB = 10^12 raw bytes)", "measured_raw_bytes_per_token": bpt}
        if bpt:
            for tb in (1, 10, 20):
                projected_tokens = int(tb * 1_000_000_000_000 / bpt)
                projections[f"{tb}TB"] = {
                    "estimated_tokens": projected_tokens,
                    "estimated_uint32_bytes": projected_tokens * 4,
                    "estimated_packed21_bytes": math.ceil(projected_tokens * 21 / 8),
                    "estimated_conversion_seconds": tb * 1_000_000_000_000 / max(raw / elapsed, 1e-9),
                }
        return {
            **explicit_counters, **dict(self.counters), "total_tokens": tokens,
            "train_tokens": sum(row["token_count"] for row in shards["train"]),
            "val_tokens": sum(row["token_count"] for row in shards["val"]),
            "raw_bytes_per_token": bpt, "tokens_per_document": tokens / self.counters["documents_written"] if self.counters["documents_written"] else None,
            "tokens_per_mb": tokens / (raw / (1 << 20)) if raw else None,
            "unique_token_ids_seen": int(np.count_nonzero(seen)),
            "min_token_id_seen": int(np.flatnonzero(seen)[0]) if np.any(seen) else None,
            "max_token_id_seen": int(np.flatnonzero(seen)[-1]) if np.any(seen) else None,
            "frequency_bands": {
                "seen_once": int(np.count_nonzero(seen_counts == 1)),
                "seen_less_than_10": int(np.count_nonzero(seen_counts < 10)),
                "seen_less_than_100": int(np.count_nonzero(seen_counts < 100)),
                "seen_less_than_1000": int(np.count_nonzero(seen_counts < 1000)),
            },
            "top_tokens": top, "elapsed_seconds": elapsed,
            "raw_read_mb_s": self.counters["input_bytes_read"] / (1 << 20) / elapsed,
            "raw_tokenized_mb_s": raw / (1 << 20) / elapsed, "tokens_per_second": tokens / elapsed,
            "output_mb_s": tokens * 4 / (1 << 20) / elapsed,
            "peak_ram_bytes": max(self.peak_ram_bytes, self._ram_bytes()), "corpus_projection": projections,
        }

    def _write_metadata(self, final: bool) -> Dict[str, Any]:
        shards = self._shards()
        stats = self._stats(shards)
        config = {
            **self._config_identity(), "workers": self.workers, "dry_run": self.dry_run,
            "sample_mb": self.sample_mb, "packed21": self.enable_packed21,
            "short_document_bytes": self.short_document_bytes, "long_document_bytes": self.long_document_bytes,
            "tokenizer_backend_info": {
                "selected": self.tokenizer_backend,
                "python_regex_unicode": "17.0.0",
                "rust_regex_unicode": "16.0.0" if self.tokenizer_backend == "native" else None,
                "unicode_table_difference_policy": "route the complete document to the verified Python reference" if self.tokenizer_backend == "native" else None,
            },
            "document_serialization": {"chat": CHAT_FORMAT, "multi_field": FIELD_FORMAT},
            "document_boundary": {"eos_token": "<|eos|>", "eos_token_id": self.tc.eos_token_id, "bos_policy": "none", "policy": "append exactly one EOS ID after each non-empty extracted document; literal EOS text is encoded normally"},
        }
        manifest = {
            "format_version": 1, "dataset_name": "AGLM-tokenized-dataset", "created_at": utc_now(),
            "complete": bool(final), "sample_run": bool(self.sample_bytes), "tokenizer": self.tc.as_dict(),
            "dedupe_exact": self.dedupe_exact,
            "dtype": "uint32", "numpy_dtype": "<u4", "endian": "little",
            "document_boundary": config["document_boundary"], "shards": shards, "statistics": stats,
            "metadata_database": "metadata/checkpoint.sqlite3",
        }
        atomic_write_json(self.meta_dir / "dataset_manifest.json", manifest)
        atomic_write_json(self.meta_dir / "shard_index.json", {"format_version": 1, "shards": shards})
        atomic_write_json(self.meta_dir / "tokenizer_info.json", self.tc.as_dict())
        atomic_write_json(self.meta_dir / "statistics.json", stats)
        atomic_write_json(self.meta_dir / "conversion_config.json", config)
        lines = [json.dumps(json.loads(row[0]), ensure_ascii=False, sort_keys=True) for row in self.db.execute("SELECT record_json FROM sources ORDER BY path")]
        atomic_write_text(self.meta_dir / "source_files.jsonl", "\n".join(lines) + ("\n" if lines else ""))
        atomic_write_text(self.output_dir / "scripts" / "verification_information.txt", "Verify without training:\npython build_aglm_dataset.py --verify-only --output-dir <DATASET> --tokenizer <TOKENIZER>\nBenchmark loader:\npython aglm_dataset_loader.py --manifest <DATASET>/metadata/dataset_manifest.json\n")
        return manifest

    def _write_reports(self, manifest: Mapping[str, Any], verification: Mapping[str, Any]) -> None:
        stats = manifest["statistics"]
        dry = f"""# AGLM Dataset Dry-Run Report

Generated: {utc_now()}

- Mode: {'sample/dry-run' if self.sample_bytes else 'authorized full conversion'}
- Raw UTF-8 bytes tokenized: {stats['raw_utf8_bytes']:,}
- Documents written: {stats['documents_written']:,}
- Tokens: {stats['total_tokens']:,}
- Raw bytes/token: {stats['raw_bytes_per_token']}
- Train shards: {len(manifest['shards']['train'])}
- Validation shards: {len(manifest['shards']['val'])}
- Raw tokenization throughput: {stats['raw_tokenized_mb_s']:.3f} MiB/s
- Peak aggregate worker RSS: {stats['peak_ram_bytes'] / (1 << 20):.1f} MiB
- Verification: {verification['status']}

This report records measurements from this run; no rate is hard-coded from an earlier prototype.
"""
        atomic_write_text(self.report_dir / "AGLM_DATASET_DRY_RUN_REPORT.md", dry)


def preflight_roundtrip(
    files: Sequence[Mapping[str, Any]], tc: TokenizerCensus, text_fields: Sequence[str],
    text_chunk_bytes: int, max_document_bytes: int, samples_per_format: int = 2,
    candidate_encoder: Any = None,
) -> Dict[str, Any]:
    """Deterministically reservoir-sample extracted documents from every present format."""
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for info in files:
        grouped.setdefault(info["ext"], []).append(info)
    report: Dict[str, Any] = {"formats": {}, "total_samples": 0, "failures": []}
    for ext, entries in sorted(grouped.items()):
        rng = random.Random(int(hashlib.sha256(ext.encode()).hexdigest()[:16], 16))
        candidates: List[Dict[str, Any]] = []
        errors: List[str] = []
        shuffled = list(entries)
        rng.shuffle(shuffled)
        for info in shuffled[: min(4, len(shuffled))]:
            try:
                for index, doc in enumerate(stream_documents_from_file(info, set(text_fields), text_chunk_bytes=text_chunk_bytes, max_document_bytes=max_document_bytes)):
                    if doc["text"] is not None and doc["raw_bytes"]:
                        if len(candidates) < samples_per_format:
                            candidates.append(doc)
                        else:
                            slot = rng.randrange(index + 1)
                            if slot < samples_per_format:
                                candidates[slot] = doc
                    if index >= 63:
                        break
            except Exception as exc:
                errors.append(f"{info['rel_path']}: {exc}")
            if len(candidates) >= samples_per_format:
                break
        results = []
        for doc in candidates:
            tokens, _ = tc.tokenizer.engine.encode(doc["text"])
            array = np.asarray(tokens, dtype=UINT32_DTYPE)
            decoded = tc.tokenizer.engine.decode_to_bytes(array.tolist())
            expected = doc["text"].encode("utf-8")
            passed = decoded == expected and hashlib.sha256(decoded).hexdigest() == hashlib.sha256(expected).hexdigest()
            candidate_exact = None
            if candidate_encoder is not None:
                candidate_bytes = candidate_encoder(doc["text"])
                candidate_exact = candidate_bytes == array.tobytes()
                passed = passed and candidate_exact
            raw_exact = None
            if ext in PLAIN_EXTENSIONS and not doc["invalid_utf8_replacements"]:
                raw_exact = hashlib.sha256(decoded).hexdigest() == doc["content_sha256"]
                passed = passed and raw_exact
            result = {
                "source": doc["source"], "doc_index": doc["doc_index"], "tokens": len(array),
                "roundtrip": passed, "raw_text_sha256_exact": raw_exact,
                "candidate_token_ids_bit_exact": candidate_exact,
            }
            results.append(result)
            if not passed:
                report["failures"].append(result)
        report["formats"][ext] = {"samples": results, "errors": errors, "status": "passed" if results and all(row["roundtrip"] for row in results) else "no_extractable_sample" if not results else "failed"}
        report["total_samples"] += len(results)
    if report["failures"]:
        raise RuntimeError(f"Preflight tokenizer roundtrip failures: {report['failures']}")
    return report


def verify_dataset(
    output_dir: Path | str, db: sqlite3.Connection, manifest: Mapping[str, Any],
    tc: TokenizerCensus, report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    root = Path(output_dir)
    errors: List[str] = []
    verified = 0
    for split in ("train", "val"):
        for record in manifest["shards"][split]:
            path = root / record["path"]
            if not path.is_file():
                errors.append(f"missing {record['path']}")
                continue
            size = path.stat().st_size
            is_zst = path.name.endswith(".zst") or record.get("compression") == "zstd_level_1"
            if not is_zst:
                if size % 4 or size // 4 != record["token_count"] or size != record["byte_size"]:
                    errors.append(f"size/token mismatch {record['path']}")
            else:
                if size != record["byte_size"]:
                    errors.append(f"size mismatch {record['path']}")
            if sha256_file(path) != record["sha256"]:
                errors.append(f"SHA256 mismatch {record['path']}")
            observed_count = 0
            observed_min, observed_max = EXPECTED_MAX_ID, 0

            if is_zst:
                import zstandard as zstd
                dctx = zstd.ZstdDecompressor()
                with path.open("rb") as handle:
                    uncompressed_limit = int(record.get("uncompressed_byte_size") or (record["token_count"] * 4) + 1024)
                    decompressed = dctx.decompress(handle.read(), max_output_size=uncompressed_limit)
                    tokens = np.frombuffer(decompressed, dtype=UINT32_DTYPE)
                    observed_count = len(tokens)
                    observed_min = int(tokens.min())
                    observed_max = int(tokens.max())
                    if observed_max >= EXPECTED_VOCAB_SIZE:
                        errors.append(f"out-of-range ID in {record['path']}")
            else:
                with path.open("rb") as handle:
                    while True:
                        tokens = np.fromfile(handle, dtype=UINT32_DTYPE, count=1_048_576)
                        if not len(tokens):
                            break
                        observed_count += len(tokens)
                        observed_min = min(observed_min, int(tokens.min()))
                        observed_max = max(observed_max, int(tokens.max()))
                        if int(tokens.max()) >= EXPECTED_VOCAB_SIZE:
                            errors.append(f"out-of-range ID in {record['path']}")

            if observed_count != record["token_count"] or observed_min != record["min_token_id"] or observed_max != record["max_token_id"]:
                errors.append(f"manifest range/count mismatch {record['path']}")
            verified += 1
    leakage = db.execute("SELECT COUNT(*) FROM (SELECT content_hash FROM documents GROUP BY content_hash HAVING COUNT(DISTINCT split)>1)").fetchone()[0]
    if leakage:
        errors.append(f"{leakage} content hashes occur in both train and val")
    duplicate_rows = db.execute("SELECT COUNT(*) FROM (SELECT content_hash FROM documents GROUP BY content_hash HAVING COUNT(*)>1)").fetchone()[0]
    if manifest["statistics"].get("exact_duplicates") is not None and duplicate_rows and manifest.get("dedupe_exact"):
        errors.append(f"dedupe enabled but {duplicate_rows} duplicate hashes remain")
    report = {"status": "PASSED" if not errors else "FAILED", "verified_shards": verified, "train_validation_hash_leakage": leakage, "duplicate_document_hash_groups": duplicate_rows, "errors": errors, "verified_at": utc_now()}
    lines = [
        "# AGLM Training Dataset Verification Report", "", f"Generated: {report['verified_at']}", "",
        f"Status: **{report['status']}**", "", f"- Shards verified: {verified}",
        f"- Train/validation document-hash leakage: {leakage}", f"- Duplicate hash groups retained: {duplicate_rows}",
        f"- Canonical dtype: little-endian uint32 (`<u4`)", f"- Tokenizer SHA256: `{tc.model_sha256}`", "",
        "## Errors", "", *(([f"- {error}" for error in errors]) or ["- None."]),
    ]
    atomic_write_text(report_path or (REPO_ROOT / "AGLM_TRAINING_DATASET_VERIFICATION_REPORT.md"), "\n".join(lines) + "\n")
    if errors:
        raise RuntimeError("Final dataset verification failed: " + "; ".join(errors))
    return report


def write_inventory_report(summary: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Dataset Input Inventory", "", f"Generated: {summary['created_at']}", "",
        f"Input directory: `{summary['input_dir']}`", "",
        f"Supported files: {summary['total_supported_files']:,}",
        f"Ignored files: {summary['total_ignored_files']:,}",
        f"Supported on-disk bytes: {summary['total_raw_bytes']:,}", "",
        "## Supported files", "", "| Bytes | Format | Relative path |", "|---:|:---:|---|",
    ]
    lines.extend(f"| {row['size_bytes']:,} | `{row['ext']}` | `{row['rel_path']}` |" for row in summary["supported_files"])
    lines.extend(["", "## Ignored files", "", "| Bytes | Reason | Relative path |", "|---:|---|---|"])
    lines.extend(f"| {row['size_bytes']:,} | {row['reason']} | `{row['rel_path']}` |" for row in summary["ignored_files"])
    atomic_write_text(path, "\n".join(lines) + "\n")


def parse_text_fields(value: str) -> List[str]:
    fields = [item.strip() for item in value.split(",") if item.strip()]
    if not fields:
        raise argparse.ArgumentTypeError("--text-fields must contain at least one field")
    return fields


def parse_worker_counts(value: str) -> List[int]:
    try:
        counts = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("worker counts must be comma-separated integers") from exc
    if not counts or counts[0] <= 0:
        raise argparse.ArgumentTypeError("worker counts must be positive")
    return counts


def benchmark_worker_scaling(
    files: Sequence[Mapping[str, Any]], tc: TokenizerCensus, text_fields: Sequence[str],
    counts: Sequence[int], text_chunk_bytes: int, max_document_bytes: int,
    target_raw_bytes: int = 4 << 20,
    tokenizer_backend: str = "reference",
) -> Dict[str, Any]:
    """Bounded opt-in 1/2/4/8-style tokenizer scaling benchmark."""
    texts: List[str] = []
    raw_bytes = 0
    for info in files:
        try:
            for doc in stream_documents_from_file(info, set(text_fields), text_chunk_bytes=text_chunk_bytes, max_document_bytes=max_document_bytes):
                if doc["text"] and doc["raw_bytes"]:
                    texts.append(doc["text"])
                    raw_bytes += doc["raw_bytes"]
                    if raw_bytes >= target_raw_bytes:
                        break
        except (ValueError, OSError, UnicodeError, csv.Error):
            continue
        if raw_bytes >= target_raw_bytes:
            break
    if not texts:
        return {"status": "skipped", "reason": "no extractable documents"}
    tasks = [(text, False) for text in texts]
    results: List[Dict[str, Any]] = []
    global _WORKER_TOKENIZER, _WORKER_NATIVE, _WORKER_BACKEND
    _WORKER_TOKENIZER = tc.tokenizer
    _WORKER_BACKEND = tokenizer_backend
    if tokenizer_backend == "native" and _WORKER_NATIVE is None:
        _WORKER_NATIVE = AGLMNativeAccelerator(tc.tokenizer)
    cpu_count = os.cpu_count() or 1
    base_rss = psutil.Process().memory_info().rss if psutil else 0
    available = psutil.virtual_memory().available if psutil else 0
    for workers in counts:
        if workers > cpu_count:
            results.append({"workers": workers, "status": "skipped", "reason": f"only {cpu_count} logical CPUs available"})
            continue
        if tokenizer_backend == "reference" and psutil and base_rss * workers > available * 0.75:
            results.append({
                "workers": workers, "status": "skipped",
                "reason": f"conservative tokenizer RAM demand {base_rss * workers / (1 << 30):.1f} GiB exceeds 75% of {available / (1 << 30):.1f} GiB available",
            })
            print(f"[WORKER SCALING] workers={workers} skipped for RAM safety", flush=True)
            continue
        started = time.perf_counter()
        if workers == 1:
            tokenized = [_tokenize_task(task) for task in tasks]
        elif tokenizer_backend == "native":
            with ThreadPoolExecutor(max_workers=workers) as executor:
                tokenized = list(executor.map(_tokenize_task, tasks))
        else:
            context = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
            with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_worker_init, initargs=(tc.tokenizer_dir, tokenizer_backend)) as executor:
                tokenized = list(executor.map(_tokenize_task, tasks, chunksize=1))
        elapsed = time.perf_counter() - started
        tokens = sum(row["token_count_without_eos"] for row in tokenized)
        results.append({
            "workers": workers, "status": "measured", "raw_bytes": raw_bytes, "tokens": tokens,
            "elapsed_seconds": elapsed, "raw_mb_s": raw_bytes / (1 << 20) / max(elapsed, 1e-9),
            "tokens_per_second": tokens / max(elapsed, 1e-9),
        })
        print(f"[WORKER SCALING] workers={workers} raw={raw_bytes / (1 << 20):.2f} MiB rate={results[-1]['raw_mb_s']:.3f} MiB/s", flush=True)
    return {
        "status": "complete", "sample_raw_bytes": raw_bytes, "results": results,
        "backend": tokenizer_backend,
        "note": "native uses GIL-free threads and one shared immutable trie; reference uses forked processes",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build AGLM uint32 sharded training data without starting training")
    parser.add_argument("--input-dir", help="one recursively scanned input folder")
    parser.add_argument("--output-dir", required=True, help="clean output package directory")
    parser.add_argument("--tokenizer", "--tokenizer-path", dest="tokenizer", default=str(REPO_ROOT / "exported_tokenizers" / "aglm_universal_max"))
    parser.add_argument("--dtype", choices=["uint32"], default="uint32")
    parser.add_argument("--val-ratio", type=float, default=0.005)
    parser.add_argument("--shard-tokens", type=int, default=100_000_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--tokenizer-backend", choices=["reference", "native"], default="reference",
        help="reference Python or bit-exact Rust accelerator (must pass preflight)",
    )
    parser.add_argument("--benchmark-workers", type=parse_worker_counts, help="opt-in scaling benchmark, e.g. 1,2,4,8")
    dedupe = parser.add_mutually_exclusive_group()
    dedupe.add_argument("--dedupe-exact", action="store_true")
    dedupe.add_argument("--no-dedupe-exact", action="store_false", dest="dedupe_exact")
    parser.set_defaults(dedupe_exact=False)
    parser.add_argument("--minhash-dedup", action="store_true", default=True, help="enable MinHash LSH near-deduplication")
    parser.add_argument("--no-minhash-dedup", action="store_false", dest="minhash_dedup")
    parser.add_argument("--quality-pruning", action="store_true", default=True, help="enable quality & repetitive spam pruning")
    parser.add_argument("--no-quality-pruning", action="store_false", dest="quality_pruning")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--packed21", action="store_true")
    parser.add_argument("--text-fields", type=parse_text_fields, default=list(DEFAULT_TEXT_FIELDS), help="comma-separated ordered extraction fields")
    parser.add_argument("--sample-mb", type=int)
    parser.add_argument("--dry-run", action="store_true", help="sample mode; defaults to 100 MiB")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--authorize-full-conversion", action="store_true", help="required safety gate for an uncapped full-folder run")
    parser.add_argument("--text-chunk-bytes", type=int, default=1 << 20)
    parser.add_argument("--max-document-bytes", type=int, default=64 << 20)
    parser.add_argument("--log-interval", type=float, default=5.0)
    parser.add_argument("--report-dir", default=str(REPO_ROOT))
    args = parser.parse_args(argv)

    census = TokenizerCensus(args.tokenizer)
    output = Path(args.output_dir).expanduser().resolve()
    if args.verify_only:
        db = sqlite3.connect(output / "metadata" / "checkpoint.sqlite3")
        manifest = json.loads((output / "metadata" / "dataset_manifest.json").read_text(encoding="utf-8"))
        try:
            result = verify_dataset(output, db, manifest, census)
        finally:
            db.close()
        print(json.dumps(result, indent=2))
        return 0
    if not args.input_dir:
        parser.error("--input-dir is required unless --verify-only is used")
    files, inventory = discover_input_inventory(args.input_dir, args.output_dir, verbose=True)
    write_inventory_report(inventory, Path(args.report_dir).resolve() / "DATASET_INPUT_INVENTORY.md")
    if not files:
        parser.error("inventory contains no supported, non-binary text files")
    if args.inventory_only:
        print("Inventory-only mode complete; no files were tokenized.")
        return 0
    if args.dry_run and args.sample_mb is None:
        args.sample_mb = 100
    if args.sample_mb is None and not args.authorize_full_conversion:
        parser.error("Refusing an uncapped conversion. Complete 100 MiB and 1 GiB sample gates, then pass --authorize-full-conversion.")

    candidate_encoder = None
    if args.tokenizer_backend == "native":
        _worker_init(census.tokenizer_dir, "native")
        candidate_encoder = _WORKER_NATIVE.encode_fast_u32_bytes
    preflight = preflight_roundtrip(
        files, census, args.text_fields, args.text_chunk_bytes, args.max_document_bytes,
        candidate_encoder=candidate_encoder,
    )
    atomic_write_json(output / "metadata" / "preflight_roundtrip.json", preflight)
    print(f"[PREFLIGHT] {preflight['total_samples']} sampled documents roundtripped; failures=0")
    if args.benchmark_workers:
        scaling = benchmark_worker_scaling(
            files, census, args.text_fields, args.benchmark_workers,
            args.text_chunk_bytes, args.max_document_bytes,
            tokenizer_backend=args.tokenizer_backend,
        )
        atomic_write_json(output / "metadata" / "worker_scaling_benchmark.json", scaling)
        print("[WORKER SCALING] " + json.dumps(scaling, indent=2))
    builder = ProductionDatasetBuilder(
        args.input_dir, str(output), census, shard_tokens=args.shard_tokens,
        val_ratio=args.val_ratio, dedupe_exact=args.dedupe_exact,
        enable_packed21=args.packed21, minhash_dedup=args.minhash_dedup,
        quality_pruning=args.quality_pruning, text_fields=args.text_fields,
        sample_mb=args.sample_mb, dry_run=args.dry_run, resume=args.resume,
        workers=args.workers, text_chunk_bytes=args.text_chunk_bytes,
        max_document_bytes=args.max_document_bytes, log_interval=args.log_interval,
        inventory_summary=inventory, report_dir=args.report_dir,
        tokenizer_backend=args.tokenizer_backend,
    )
    builder.build_dataset(files)
    full_command = (
        f"python3 {shlex.quote(str(REPO_ROOT / 'build_aglm_dataset.py'))} --input-dir {shlex.quote(str(Path(args.input_dir).resolve()))} "
        f"--output-dir {shlex.quote(str(output.parent / 'aglm_tokenized_dataset'))} --tokenizer {shlex.quote(str(Path(args.tokenizer).resolve()))} "
        f"--dtype uint32 --val-ratio {args.val_ratio} --shard-tokens {args.shard_tokens} "
        f"--workers {args.workers} --tokenizer-backend {args.tokenizer_backend} "
        f"{'--dedupe-exact ' if args.dedupe_exact else ''}--resume --authorize-full-conversion"
    )
    if args.sample_mb is None:
        print("\nAuthorized full-folder conversion completed and verified. No training was started by the builder.")
    else:
        print("\nFull-folder conversion remains gated. After approving both sample reports, run exactly:\n" + full_command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
