#!/usr/bin/env python3
"""Compare frozen-vocabulary whole-document minima with production segmentation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Tuple

import numpy as np

from aglm_tokenizer.core.tokenizer import AGLMUniversalTokenizer
from aglm_tokenizer.native import AGLMNativeAccelerator
from build_aglm_dataset import _stream_byte_chunks


VOCAB_SIZE = 1_551_017
VOCAB_SHA256 = "1f865241d0f3ebcc41bc2e75de8eb6ef190dd23e2fa5a444f69d40ad250c74eb"
ARTIFACT_HASHES = {
    "aglm_vocab.json": "48d47c68673e3eabecdba9135deaf67a2068d3a8b9c83d12d13b55555860a8d9",
    "aglm_vocab.json.gz": VOCAB_SHA256,
    "manifest.json": "71d4b419d9efb117bf558313224ced76b0880ff7fad15ef5fd36fa4d3ff8aee0",
}


def artifact_hashes(tokenizer_dir: Path) -> Dict[str, str]:
    return {
        name: hashlib.sha256((tokenizer_dir / name).read_bytes()).hexdigest()
        for name in ARTIFACT_HASHES
    }


def verify_frozen(tokenizer_dir: Path) -> Dict[str, str]:
    observed = artifact_hashes(tokenizer_dir)
    if observed != ARTIFACT_HASHES:
        raise RuntimeError(f"frozen vocabulary artifact mismatch: {observed}")
    return observed


def iter_sample_documents(path: Path, target_bytes: int, chunk_bytes: int) -> Iterator[Tuple[int, str, int]]:
    consumed = 0
    with path.open("rb") as handle:
        for index, (raw, _, _) in enumerate(_stream_byte_chunks(handle, chunk_bytes, 0)):
            remaining = target_bytes - consumed
            if remaining <= 0:
                break
            if len(raw) > remaining:
                raw = raw[:remaining]
                # Do not manufacture U+FFFD at the audit boundary. At most three
                # bytes are removed from a valid UTF-8 source prefix.
                while raw:
                    try:
                        text = raw.decode("utf-8")
                        break
                    except UnicodeDecodeError as exc:
                        if exc.reason == "unexpected end of data" and len(raw) - exc.start <= 4:
                            raw = raw[:exc.start]
                            continue
                        text = raw.decode("utf-8", errors="replace")
                        break
                else:
                    text = ""
            else:
                text = raw.decode("utf-8", errors="replace")
            normalized_bytes = len(text.encode("utf-8"))
            if normalized_bytes:
                yield index, text, normalized_bytes
                consumed += normalized_bytes
            if consumed >= target_bytes:
                break


def run_pass(
    source: Path,
    target_bytes: int,
    chunk_bytes: int,
    workers: int,
    encode: Callable[[str], bytes],
    expected: List[Dict[str, Any]] | None = None,
    tokenizer: AGLMUniversalTokenizer | None = None,
) -> Dict[str, Any]:
    started = time.perf_counter()
    raw_total = 0
    token_total = 0
    document_rows: List[Dict[str, Any]] = []
    global_digest = hashlib.sha256()
    roundtrip_samples = 0
    batch: List[Tuple[int, str, int]] = []

    def consume(rows: List[Tuple[int, str, int]], pool: ThreadPoolExecutor) -> None:
        nonlocal raw_total, token_total, roundtrip_samples
        encoded = pool.map(encode, (row[1] for row in rows))
        for (index, text, raw_bytes), token_bytes in zip(rows, encoded):
            ids = np.frombuffer(token_bytes, dtype="<u4")
            if ids.size and int(ids.max()) >= VOCAB_SIZE:
                raise RuntimeError(f"out-of-range ID in document {index}")
            digest = hashlib.sha256(token_bytes).hexdigest()
            global_digest.update(token_bytes)
            row = {"document_index": index, "raw_bytes": raw_bytes, "tokens": len(ids), "sha256": digest}
            document_rows.append(row)
            raw_total += raw_bytes
            token_total += len(ids)
            if tokenizer is not None and index % 127 == 0:
                decoded = tokenizer.decode_to_bytes(ids.tolist())
                if decoded != text.encode("utf-8"):
                    raise RuntimeError(f"minimum segmentation roundtrip failed at document {index}")
                roundtrip_samples += 1

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row in iter_sample_documents(source, target_bytes, chunk_bytes):
            batch.append(row)
            if len(batch) >= workers * 2:
                consume(batch, pool)
                batch = []
        if batch:
            consume(batch, pool)
    elapsed = time.perf_counter() - started
    if expected is not None:
        if len(expected) != len(document_rows):
            raise RuntimeError("comparison pass document count changed")
        for production, minimal in zip(expected, document_rows):
            if production["document_index"] != minimal["document_index"] or production["raw_bytes"] != minimal["raw_bytes"]:
                raise RuntimeError("comparison pass input changed")
            if minimal["tokens"] > production["tokens"]:
                raise RuntimeError("whole-document minimum exceeded production token count")
    return {
        "raw_bytes": raw_total,
        "documents": len(document_rows),
        "tokens": token_total,
        "raw_bytes_per_token": raw_total / token_total,
        "elapsed_seconds": elapsed,
        "raw_mib_per_second": raw_total / (1 << 20) / elapsed,
        "token_stream_sha256": global_digest.hexdigest(),
        "roundtrip_samples": roundtrip_samples,
        "document_rows": document_rows,
    }


def atomic_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--sample-bytes", type=int, default=1 << 30)
    parser.add_argument("--chunk-bytes", type=int, default=1 << 20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", default="benchmark_results/minimal_segmentation_1g.json")
    args = parser.parse_args()
    source = Path(args.source).expanduser().resolve()
    tokenizer_dir = Path(args.tokenizer).expanduser().resolve()
    before = verify_frozen(tokenizer_dir)
    tokenizer = AGLMUniversalTokenizer.load(str(tokenizer_dir))
    if tokenizer.vocab_size != VOCAB_SIZE:
        raise RuntimeError(f"vocabulary size changed: {tokenizer.vocab_size}")
    native = AGLMNativeAccelerator(tokenizer)

    print("Running production segmentation pass...", flush=True)
    production = run_pass(
        source, args.sample_bytes, args.chunk_bytes, args.workers,
        native.encode_fast_u32_bytes,
    )
    print("Running whole-document exact-minimum pass...", flush=True)
    minimum = run_pass(
        source, args.sample_bytes, args.chunk_bytes, args.workers,
        native.encode_minimal_u32_bytes, expected=production["document_rows"], tokenizer=tokenizer,
    )
    production_rows = production.pop("document_rows")
    minimum_rows = minimum.pop("document_rows")
    lower = sum(m["tokens"] < p["tokens"] for p, m in zip(production_rows, minimum_rows))
    identical = sum(
        m["tokens"] == p["tokens"] and m["sha256"] == p["sha256"]
        for p, m in zip(production_rows, minimum_rows)
    )
    same_count_different_ids = sum(
        m["tokens"] == p["tokens"] and m["sha256"] != p["sha256"]
        for p, m in zip(production_rows, minimum_rows)
    )
    tokens_saved = production["tokens"] - minimum["tokens"]
    after = verify_frozen(tokenizer_dir)
    result = {
        "status": "PASSED",
        "scope": "exact minimum independently computed within each bounded source document; no path may cross a document boundary",
        "source": str(source),
        "sample_target_bytes": args.sample_bytes,
        "chunk_bytes": args.chunk_bytes,
        "workers": args.workers,
        "vocabulary": {"size": VOCAB_SIZE, "artifact_hashes_before": before, "artifact_hashes_after": after, "modified": before != after},
        "production": production,
        "whole_document_minimum": minimum,
        "comparison": {
            "documents_with_fewer_tokens": lower,
            "documents_with_identical_ids": identical,
            "documents_with_equal_count_but_different_ids": same_count_different_ids,
            "tokens_saved": tokens_saved,
            "token_reduction_fraction": tokens_saved / production["tokens"],
            "minimum_is_never_longer": True,
        },
    }
    output = Path(args.output).expanduser().resolve()
    atomic_json(output, result)
    report = output.with_suffix(".md")
    comparison = result["comparison"]
    report.write_text(
        "# Frozen AGLM Minimum-Segmentation Audit\n\n"
        f"Status: **{result['status']}**  \nVocabulary: **{VOCAB_SIZE:,}** IDs (unchanged)  \n"
        f"Sample: **{production['raw_bytes']:,}** UTF-8 bytes across {production['documents']:,} bounded documents\n\n"
        "| Segmentation | Tokens | Raw bytes/token | Raw MiB/s | Token stream SHA256 |\n"
        "|---|---:|---:|---:|---|\n"
        f"| Production regex-bounded | {production['tokens']:,} | {production['raw_bytes_per_token']:.6f} | {production['raw_mib_per_second']:.3f} | `{production['token_stream_sha256']}` |\n"
        f"| Whole-document exact minimum | {minimum['tokens']:,} | {minimum['raw_bytes_per_token']:.6f} | {minimum['raw_mib_per_second']:.3f} | `{minimum['token_stream_sha256']}` |\n\n"
        f"- Documents with fewer tokens: {comparison['documents_with_fewer_tokens']:,}\n"
        f"- Tokens removed by relaxing only regex boundaries: {comparison['tokens_saved']:,} ({comparison['token_reduction_fraction'] * 100:.4f}%)\n"
        f"- Equal-count but different-ID segmentations: {comparison['documents_with_equal_count_but_different_ids']:,}\n"
        f"- Sampled exact byte roundtrips: {minimum['roundtrip_samples']:,}\n"
        "- No segmentation crosses the builder's document boundary.\n"
        "- This is position accounting only; it is not a model speed claim.\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in result.items() if key != "vocabulary"}, indent=2), flush=True)
    print(f"Reports: {output} and {report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
