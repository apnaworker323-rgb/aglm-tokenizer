#!/usr/bin/env python3
"""Reproducible profiling harness for the exact AGLM reference tokenizer.

The 1 GiB benchmark intentionally holds the raw sample in RAM to remove storage
from the timed tokenizer measurements. Production conversion remains streaming.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import multiprocessing as mp
import os
import platform
import pstats
import resource
import shutil
import tempfile
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from aglm_tokenizer.core.script_handlers import ScriptSegmenter
from aglm_tokenizer.core.tokenizer import AGLMUniversalTokenizer
from build_aglm_dataset import EXPECTED_VOCAB_SIZE, TokenizerCensus, atomic_write_json, atomic_write_text

try:
    from aglm_tokenizer.native import AGLMNativeAccelerator, NativeBpe
except ImportError:  # pragma: no cover - native profiling is opt-in
    AGLMNativeAccelerator = None
    NativeBpe = None


UINT32_LE = np.dtype("<u4")
_TOKENIZER: Optional[AGLMUniversalTokenizer] = None
_NATIVE: Any = None


def _init_worker(tokenizer_path: str) -> None:
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AGLMUniversalTokenizer.load(tokenizer_path)


def _tokenize_summary(text: str) -> Tuple[int, bytes]:
    if _TOKENIZER is None:
        raise RuntimeError("worker tokenizer is not initialized")
    ids, _ = _TOKENIZER.engine.encode(text)
    array = np.asarray(ids, dtype=UINT32_LE)
    return len(array), hashlib.sha256(array.tobytes()).digest()


def _tokenize_bytes(text: str) -> bytes:
    if _TOKENIZER is None:
        raise RuntimeError("worker tokenizer is not initialized")
    ids, _ = _TOKENIZER.engine.encode(text)
    return np.asarray(ids, dtype=UINT32_LE).tobytes()


def _init_native_worker(tokenizer_path: str) -> None:
    global _TOKENIZER, _NATIVE
    if _TOKENIZER is None:
        _TOKENIZER = AGLMUniversalTokenizer.load(tokenizer_path)
    if _NATIVE is None:
        if AGLMNativeAccelerator is None:
            raise RuntimeError("native extension is unavailable")
        _NATIVE = AGLMNativeAccelerator(_TOKENIZER)


def _native_output(text: str) -> Tuple[bytes, bool, int]:
    if _NATIVE is None:
        raise RuntimeError("native tokenizer worker is not initialized")
    fallback = _NATIVE.requires_reference_fallback(text)
    if fallback:
        if _TOKENIZER is None:  # pragma: no cover - defensive
            raise RuntimeError("reference tokenizer worker is not initialized")
        raw = np.asarray(_TOKENIZER.encode(text), dtype=UINT32_LE).tobytes()
    else:
        raw = _NATIVE.engine.encode_text_fast_u32(text)
    return raw, fallback, len(text.encode("utf-8"))


def _ipc_echo(text: str) -> Tuple[int, bytes]:
    raw = text.encode("utf-8")
    return len(raw), hashlib.sha256(raw).digest()


def _combine_results(results: Iterable[Tuple[int, bytes]]) -> Tuple[int, str]:
    count = 0
    digest = hashlib.sha256()
    for tokens, chunk_digest in results:
        count += tokens
        digest.update(chunk_digest)
    return count, digest.hexdigest()


def load_raw_sample(path: Path, target_bytes: int, chunk_bytes: int) -> Tuple[List[bytes], Dict[str, Any]]:
    chunks: List[bytes] = []
    remaining = target_bytes
    digest = hashlib.sha256()
    started = time.perf_counter()
    with path.open("rb", buffering=8 << 20) as handle:
        while remaining:
            raw = handle.read(min(chunk_bytes, remaining))
            if not raw:
                break
            # Avoid splitting a valid UTF-8 sequence at a benchmark chunk boundary.
            if remaining > len(raw):
                try:
                    raw.decode("utf-8")
                except UnicodeDecodeError as exc:
                    if exc.reason != "unexpected end of data":
                        raise
                    pushback = len(raw) - exc.start
                    handle.seek(-pushback, os.SEEK_CUR)
                    raw = raw[:exc.start]
            chunks.append(raw)
            digest.update(raw)
            remaining -= len(raw)
    elapsed = time.perf_counter() - started
    total = sum(map(len, chunks))
    return chunks, {
        "bytes": total, "seconds": elapsed, "mib_per_second": total / (1 << 20) / max(elapsed, 1e-9),
        "sha256": digest.hexdigest(), "chunks": len(chunks),
    }


def parse_utf8(raw_chunks: Sequence[bytes]) -> Tuple[List[str], Dict[str, Any]]:
    started = time.perf_counter()
    texts = [chunk.decode("utf-8") for chunk in raw_chunks]
    elapsed = time.perf_counter() - started
    total = sum(len(chunk) for chunk in raw_chunks)
    return texts, {"bytes": total, "seconds": elapsed, "mib_per_second": total / (1 << 20) / max(elapsed, 1e-9)}


def prefix_by_bytes(texts: Sequence[str], byte_limit: int) -> List[str]:
    selected: List[str] = []
    total = 0
    for text in texts:
        raw_bytes = len(text.encode("utf-8"))
        if total and total + raw_bytes > byte_limit:
            break
        selected.append(text)
        total += raw_bytes
    return selected


def stratified_by_bytes(texts: Sequence[str], byte_limit: int, slots: int = 64) -> List[str]:
    """Deterministically cover the full in-memory sample, not just its prefix."""
    if not texts or byte_limit <= 0:
        return []
    indices = sorted({min(len(texts) - 1, int(index * len(texts) / slots)) for index in range(slots)})
    selected: List[str] = []
    total = 0
    for index in indices:
        text = texts[index]
        size = len(text.encode("utf-8"))
        if selected and total + size > byte_limit:
            break
        selected.append(text)
        total += size
    return selected


def timed_stage(name: str, raw_bytes: int, function: Any) -> Tuple[Any, Dict[str, Any]]:
    started = time.perf_counter()
    result = function()
    elapsed = time.perf_counter() - started
    return result, {"name": name, "seconds": elapsed, "raw_bytes": raw_bytes, "mib_per_second": raw_bytes / (1 << 20) / max(elapsed, 1e-9)}


def profile_reference_stages(tc: TokenizerCensus, texts: Sequence[str], output_dir: Path) -> Dict[str, Any]:
    raw_bytes = sum(len(text.encode("utf-8")) for text in texts)
    stages: Dict[str, Any] = {}

    _, stages["text_extraction"] = timed_stage(
        "plain-text extraction/document metadata", raw_bytes,
        lambda: [{"text": text, "bytes": len(text.encode("utf-8"))} for text in texts],
    )
    _, stages["dedup_sha256"] = timed_stage(
        "exact-dedup SHA256", raw_bytes,
        lambda: [hashlib.sha256(text.encode("utf-8")).digest() for text in texts],
    )
    segments, stages["pre_tokenization"] = timed_stage(
        "regex/script pre-tokenization", raw_bytes,
        lambda: [ScriptSegmenter.pre_tokenize(text) for text in texts],
    )

    flat_segments = [segment for group in segments for segment in group]
    segment_bytes = sum(len(segment.encode("utf-8")) for segment in flat_segments)
    encoded_lists, stages["trie_bpe_dp"] = timed_stage(
        "trie lookup + shortest-path DP/backtracking", segment_bytes,
        lambda: [tc.tokenizer.engine.encode_segment(segment.encode("utf-8"))[0] for segment in flat_segments],
    )
    token_count = sum(map(len, encoded_lists))
    arrays, stages["uint32_conversion"] = timed_stage(
        "Python IDs to little-endian uint32", raw_bytes,
        lambda: [np.asarray(ids, dtype=UINT32_LE) for ids in encoded_lists],
    )
    output_bytes = sum(array.nbytes for array in arrays)
    tmp = output_dir / "stage_write_test.bin"

    def write_and_fsync() -> None:
        with tmp.open("wb") as handle:
            for array in arrays:
                handle.write(array.tobytes())
            handle.flush()
            os.fsync(handle.fileno())

    _, stages["shard_write_fsync"] = timed_stage("shard write + flush + fsync", output_bytes, write_and_fsync)
    _, stages["shard_sha256"] = timed_stage("shard SHA256 readback", output_bytes, lambda: hashlib.sha256(tmp.read_bytes()).digest())
    tmp.unlink(missing_ok=True)

    # Allocation profiling is intentionally small because tracemalloc changes timing.
    allocation_texts = prefix_by_bytes(texts, min(raw_bytes, 1 << 20))
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    allocation_started = time.perf_counter()
    allocation_tokens = [tc.tokenizer.engine.encode(text)[0] for text in allocation_texts]
    allocation_seconds = time.perf_counter() - allocation_started
    after = tracemalloc.take_snapshot()
    current, peak = tracemalloc.get_traced_memory()
    diff = after.compare_to(before, "lineno")
    tracemalloc.stop()
    stages["python_allocation"] = {
        "profiled_raw_bytes": sum(len(text.encode("utf-8")) for text in allocation_texts),
        "seconds_with_tracemalloc": allocation_seconds, "current_bytes": current, "peak_bytes": peak,
        "net_allocated_bytes": sum(item.size_diff for item in diff),
        "token_count_retained": sum(map(len, allocation_tokens)),
        "top_allocation_sites": [str(item) for item in diff[:15]],
    }

    profiler = cProfile.Profile()
    cprofile_texts = prefix_by_bytes(texts, min(raw_bytes, 8 << 20))
    profiler.enable()
    for text in cprofile_texts:
        tc.tokenizer.engine.encode(text)
    profiler.disable()
    profile_path = output_dir / "reference_tokenizer.cprofile"
    profiler.dump_stats(profile_path)
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative")
    stats.print_stats(40)
    cprofile_text = stream.getvalue()
    atomic_write_text(output_dir / "reference_tokenizer_cprofile.txt", cprofile_text)

    # The reference implementation performs dynamic-programming relaxations, not
    # traditional runtime BPE merge operations. Count trie matches/relaxations on
    # a bounded diagnostic prefix without adding timers to every byte in the main run.
    diagnostic = prefix_by_bytes(texts, min(raw_bytes, 256 << 10))
    match_count = relaxation_count = byte_positions = 0
    for text in diagnostic:
        for segment in ScriptSegmenter.pre_tokenize(text):
            data = segment.encode("utf-8")
            costs = [10**9] * (len(data) + 1)
            costs[0] = 0
            for index in range(len(data)):
                if costs[index] == 10**9:
                    continue
                matches = tc.tokenizer.engine.trie.all_prefix_matches(data, index)
                byte_positions += 1
                match_count += len(matches)
                for _, length in matches:
                    if costs[index] + 1 < costs[index + length]:
                        costs[index + length] = costs[index] + 1
                        relaxation_count += 1
    return {
        "profiled_raw_bytes": raw_bytes, "segments": len(flat_segments), "tokens": token_count,
        "stages": stages, "cprofile_path": str(profile_path), "cprofile_top40": cprofile_text,
        "runtime_algorithm": {
            "traditional_bpe_merge_operations": 0,
            "explanation": "The current encoder does not apply ranked pair merges at runtime; it finds all trie prefixes and solves a minimum-token shortest path.",
            "diagnostic_byte_positions": byte_positions, "diagnostic_trie_matches": match_count,
            "diagnostic_successful_dp_relaxations": relaxation_count,
        },
    }


def run_mode(
    mode: str, workers: int, texts: Sequence[str], tokenizer_path: str,
    production_bytes_result: bool = False,
) -> Dict[str, Any]:
    raw_bytes = sum(len(text.encode("utf-8")) for text in texts)
    started = time.perf_counter()
    if mode == "single":
        results = [_tokenize_bytes(text) if production_bytes_result else _tokenize_summary(text) for text in texts]
    elif mode == "threads":
        function = _tokenize_bytes if production_bytes_result else _tokenize_summary
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(function, texts))
    elif mode == "processes":
        function = _tokenize_bytes if production_bytes_result else _tokenize_summary
        context = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
        with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_init_worker, initargs=(tokenizer_path,)) as executor:
            results = list(executor.map(function, texts, chunksize=1))
    else:
        raise ValueError(mode)
    elapsed = time.perf_counter() - started
    if production_bytes_result:
        token_count = sum(len(value) // 4 for value in results)
        digest = hashlib.sha256()
        for value in results:
            digest.update(hashlib.sha256(value).digest())
        output_digest = digest.hexdigest()
    else:
        token_count, output_digest = _combine_results(results)
    return {
        "mode": mode, "workers": workers, "raw_bytes": raw_bytes, "tokens": token_count,
        "seconds": elapsed, "raw_mib_per_second": raw_bytes / (1 << 20) / max(elapsed, 1e-9),
        "tokens_per_second": token_count / max(elapsed, 1e-9), "ordered_chunk_digest": output_digest,
        "returns_uint32_bytes": production_bytes_result,
    }


def run_native_mode(mode: str, workers: int, texts: Sequence[str], tokenizer_path: str) -> Dict[str, Any]:
    """Benchmark the exact native uint32 path without retaining token arrays."""
    raw_bytes = sum(len(text.encode("utf-8")) for text in texts)
    started = time.perf_counter()
    if mode == "native-single":
        results = list(map(_native_output, texts))
    elif mode == "native-threads":
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_native_output, texts))
    elif mode == "native-processes":
        context = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=context, initializer=_init_native_worker,
            initargs=(tokenizer_path,),
        ) as executor:
            results = list(executor.map(_native_output, texts, chunksize=1))
    else:
        raise ValueError(mode)
    elapsed = time.perf_counter() - started
    token_count = 0
    digest_builder = hashlib.sha256()
    fallback_chunks = fallback_bytes = 0
    validation_started = time.perf_counter()
    for token_bytes, fallback, chunk_bytes in results:
        token_count += len(token_bytes) // 4
        digest_builder.update(hashlib.sha256(token_bytes).digest())
        if fallback:
            fallback_chunks += 1
            fallback_bytes += chunk_bytes
    digest = digest_builder.hexdigest()
    validation_seconds = time.perf_counter() - validation_started
    return {
        "mode": mode, "workers": workers, "raw_bytes": raw_bytes,
        "tokens": token_count, "seconds": elapsed,
        "raw_mib_per_second": raw_bytes / (1 << 20) / max(elapsed, 1e-9),
        "tokens_per_second": token_count / max(elapsed, 1e-9),
        "ordered_chunk_digest": digest, "returns_uint32_bytes": True,
        "unicode_reference_fallback_chunks": fallback_chunks,
        "unicode_reference_fallback_raw_bytes": fallback_bytes,
        "post_timing_validation_sha256_seconds": validation_seconds,
    }


def benchmark_ipc(texts: Sequence[str], workers: int) -> Dict[str, Any]:
    raw_bytes = sum(len(text.encode("utf-8")) for text in texts)
    context = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        results = list(executor.map(_ipc_echo, texts, chunksize=1))
    elapsed = time.perf_counter() - started
    return {
        "workers": workers, "raw_bytes": raw_bytes, "seconds": elapsed,
        "one_way_payload_mib_per_second": raw_bytes / (1 << 20) / max(elapsed, 1e-9),
        "records": len(results), "note": "text pickling + dispatch + small result; excludes tokenization",
    }


def regression_suite() -> List[str]:
    return [
        "ASCII and whitespace:\talpha  beta\r\nfinal\n",
        "हिन्दी: नमस्ते दुनिया। বাংলা ગુજરાતી ਪੰਜਾਬੀ ଓଡ଼ିଆ தமிழ் తెలుగు ಕನ್ನಡ മലയാളം",
        "Romanized Indic: mera naam Akash hai; nenu baagunnanu; enikku sukham aanu.",
        "中文分词测试。日本語のテスト。한국어 토크나이저 테스트.",
        "العربية مع التشكيل، עברית, Ελληνικά, Русский язык.",
        "👩🏽‍💻🚀 family 👨‍👩‍👧‍👦 symbols ©™✓\ufe0f",
        "def quick_sort(xs: list[int]) -> list[int]:\n    return sorted(xs)  # λ\n",
        "SELECT a.id, COUNT(*) FROM t AS a WHERE a.x >= 10 GROUP BY a.id;",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hello\\nworld\"}]}",
        "actual controls are text here: \x00 \x01 ÿ <|eos|> literal",
    ]


def verify_candidate(reference: AGLMUniversalTokenizer, candidate_encode: Any) -> Dict[str, Any]:
    cases = []
    for index, text in enumerate(regression_suite()):
        reference_ids = np.asarray(reference.encode(text), dtype=UINT32_LE)
        candidate_ids = np.asarray(candidate_encode(text), dtype=UINT32_LE)
        identical = np.array_equal(reference_ids, candidate_ids)
        cases.append({
            "case": index, "raw_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "token_count": len(reference_ids), "reference_ids_sha256": hashlib.sha256(reference_ids.tobytes()).hexdigest(),
            "identical": identical,
        })
        if not identical:
            raise AssertionError(f"candidate token IDs differ on regression case {index}")
    return {"status": "PASSED", "cases": cases}


def verify_native_corpus(reference: AGLMUniversalTokenizer, native: Any, texts: Sequence[str]) -> Dict[str, Any]:
    checked_bytes = checked_tokens = 0
    digest = hashlib.sha256()
    for index, text in enumerate(texts):
        expected = np.asarray(reference.encode(text), dtype=UINT32_LE).tobytes()
        actual = native.encode_fast_u32_bytes(text)
        if expected != actual:
            expected_ids = np.frombuffer(expected, dtype=UINT32_LE)
            actual_ids = np.frombuffer(actual, dtype=UINT32_LE)
            mismatch = next(
                (position for position, pair in enumerate(zip(expected_ids, actual_ids)) if pair[0] != pair[1]),
                min(len(expected_ids), len(actual_ids)),
            )
            raise AssertionError(f"native token IDs differ on corpus chunk {index} at token {mismatch}")
        digest.update(hashlib.sha256(actual).digest())
        checked_bytes += len(text.encode("utf-8"))
        checked_tokens += len(actual) // 4
    return {
        "status": "PASSED", "raw_bytes": checked_bytes, "tokens": checked_tokens,
        "chunks": len(texts), "ordered_uint32_digest": digest.hexdigest(),
    }


def write_report(path: Path, result: Dict[str, Any]) -> None:
    modes = result.get("mode_benchmarks", [])
    lines = [
        "# AGLM Tokenizer Performance Profile", "", f"Generated: {result['generated_at']}", "",
        "## In-memory tokenizer throughput", "", "| Mode | Workers | Raw MiB/s | Tokens/s | Sample bytes |", "|---|---:|---:|---:|---:|",
    ]
    lines.extend(f"| {row['mode']} | {row['workers']} | {row['raw_mib_per_second']:.3f} | {row['tokens_per_second']:.0f} | {row['raw_bytes']:,} |" for row in modes)
    native = result.get("native_accelerator")
    if native:
        lines.extend(["", "## Exact Rust accelerator", "", "| Mode | Workers | Raw MiB/s | Tokens/s | Sample bytes |", "|---|---:|---:|---:|---:|"])
        lines.extend(
            f"| {row['mode']} | {row['workers']} | {row['raw_mib_per_second']:.3f} | {row['tokens_per_second']:.0f} | {row['raw_bytes']:,} |"
            for row in native["benchmarks"]
        )
        fallback = native["benchmarks"][0]
        lines.extend([
            "",
            f"Regression equivalence: **{native['regression_equivalence']['status']}**. Corpus equivalence: **{native['corpus_equivalence']['status']}** over {native['corpus_equivalence']['raw_bytes']:,} raw bytes.",
            f"Unicode-version safety fallback: {fallback['unicode_reference_fallback_chunks']:,} chunks / {fallback['unicode_reference_fallback_raw_bytes']:,} raw bytes in the measured sample.",
            "The production candidate returns little-endian uint32 bytes directly, releases the GIL during Rust tokenization, and shares one immutable trie across threads.",
        ])
    lines.extend(["", "## Stage profile", "", "| Stage | Seconds | MiB/s |", "|---|---:|---:|"])
    for name, row in result.get("stage_profile", {}).get("stages", {}).items():
        if "seconds" in row:
            lines.append(f"| {name} | {row['seconds']:.6f} | {row.get('mib_per_second', 0):.3f} |")
    lines.extend([
        "", "## Interpretation", "",
        "The reference runtime is not a conventional ranked-merge BPE. It performs regex pre-tokenization, enumerates every trie prefix at each byte position, and solves a shortest-path dynamic program. Therefore the requested ‘merge operations’ bucket is zero; DP relaxation/backtracking is the corresponding cost.",
        "The native design preserves that exact algorithm. Its safe speedups come from a linear-time Rust regex with an explicit exact whitespace-boundary repair, compact trie storage, constant-time root transitions, whole-segment terminal fast paths, reusable DP buffers, direct uint32 output, and GIL-free execution.",
        "", "Any native candidate is rejected unless every uint32 token ID is identical to the Python reference on the multilingual/code suite and corpus samples.",
    ])
    atomic_write_text(path, "\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile the exact AGLM tokenizer without training")
    parser.add_argument("--sample-file", required=True)
    parser.add_argument("--tokenizer", default="exported_tokenizers/aglm_universal_max")
    parser.add_argument("--sample-bytes", type=int, default=1 << 30)
    parser.add_argument("--chunk-bytes", type=int, default=64 << 10)
    parser.add_argument("--stage-profile-bytes", type=int, default=16 << 20)
    parser.add_argument("--mode-benchmark-bytes", type=int, default=64 << 20)
    parser.add_argument("--full-in-memory-single", action="store_true", help="run the reference tokenizer over the entire in-memory sample")
    parser.add_argument("--native", action="store_true", help="benchmark the optional exact Rust accelerator")
    parser.add_argument("--native-full-in-memory", action="store_true", help="run native single/thread modes over the entire in-memory sample")
    parser.add_argument("--native-processes", action="store_true", help="also benchmark native multiprocessing (higher RAM)")
    parser.add_argument("--workers", default="1,2,4")
    parser.add_argument("--output-dir", default="benchmark_results/tokenizer_profile")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tc = TokenizerCensus(args.tokenizer)
    global _TOKENIZER, _NATIVE
    _TOKENIZER = tc.tokenizer

    raw_chunks, disk = load_raw_sample(Path(args.sample_file), args.sample_bytes, args.chunk_bytes)
    texts, utf8 = parse_utf8(raw_chunks)
    del raw_chunks
    stage_texts = prefix_by_bytes(texts, args.stage_profile_bytes)
    stage_profile = profile_reference_stages(tc, stage_texts, output)
    benchmark_texts = prefix_by_bytes(texts, args.mode_benchmark_bytes)
    workers = sorted({int(value) for value in args.workers.split(",")})
    modes: List[Dict[str, Any]] = [run_mode("single", 1, benchmark_texts, tc.tokenizer_dir)]
    for count in workers:
        if count > 1:
            modes.append(run_mode("threads", count, benchmark_texts, tc.tokenizer_dir))
            modes.append(run_mode("processes", count, benchmark_texts, tc.tokenizer_dir))
    production_ipc_texts = prefix_by_bytes(texts, min(args.mode_benchmark_bytes, 16 << 20))
    production_process = run_mode("processes", max(workers), production_ipc_texts, tc.tokenizer_dir, production_bytes_result=True)
    ipc = benchmark_ipc(production_ipc_texts, max(workers))
    full_single = run_mode("single", 1, texts, tc.tokenizer_dir) if args.full_in_memory_single else None
    regression = verify_candidate(tc.tokenizer, tc.tokenizer.encode)
    native_result = None
    if args.native or args.native_full_in_memory:
        if NativeBpe is None or AGLMNativeAccelerator is None:
            raise RuntimeError("native profiling requested, but the extension is not built; run native/aglm_native/build.sh")
        peak_before_native_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        native_started = time.perf_counter()
        _NATIVE = AGLMNativeAccelerator(tc.tokenizer)
        native_load_seconds = time.perf_counter() - native_started
        peak_after_native_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        native_regression = verify_candidate(tc.tokenizer, _NATIVE.encode_fast)
        equivalence_texts = stratified_by_bytes(texts, 4 << 20)
        native_corpus = verify_native_corpus(tc.tokenizer, _NATIVE, equivalence_texts)
        native_texts = texts if args.native_full_in_memory else benchmark_texts
        native_modes: List[Dict[str, Any]] = [run_native_mode("native-single", 1, native_texts, tc.tokenizer_dir)]
        for count in workers:
            if count > 1:
                native_modes.append(run_native_mode("native-threads", count, native_texts, tc.tokenizer_dir))
                if args.native_processes:
                    native_modes.append(run_native_mode("native-processes", count, native_texts, tc.tokenizer_dir))
        native_result = {
            "native_trie_construction_seconds": native_load_seconds,
            "peak_rss_before_native_kib": peak_before_native_kib,
            "peak_rss_after_native_kib": peak_after_native_kib,
            "node_count": _NATIVE.engine.node_count, "edge_count": _NATIVE.engine.edge_count,
            "regression_equivalence": native_regression,
            "corpus_equivalence": native_corpus, "benchmarks": native_modes,
        }
    result = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "platform": platform.platform(), "python": platform.python_version(),
            "logical_cpus": os.cpu_count(), "py_spy": shutil.which("py-spy"),
            "perf": shutil.which("perf"), "cargo": shutil.which("cargo"),
            "gxx": shutil.which("g++"),
        },
        "tokenizer": tc.as_dict(), "sample_file": str(Path(args.sample_file).resolve()),
        "sample_in_memory_bytes": sum(len(text.encode("utf-8")) for text in texts),
        "disk_read": disk, "utf8_parsing": utf8, "stage_profile": stage_profile,
        "mode_benchmarks": modes, "production_process_with_uint32_ipc": production_process,
        "multiprocessing_ipc_only": ipc, "full_in_memory_single": full_single,
        "regression_suite": regression, "native_accelerator": native_result,
    }
    atomic_write_json(output / "profile_results.json", result)
    write_report(Path("AGLM_TOKENIZER_PERFORMANCE_REPORT.md").resolve(), result)
    print(json.dumps({key: result[key] for key in ("sample_in_memory_bytes", "disk_read", "utf8_parsing", "mode_benchmarks", "production_process_with_uint32_ipc", "multiprocessing_ipc_only", "full_in_memory_single", "native_accelerator")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
