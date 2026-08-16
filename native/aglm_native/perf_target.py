#!/usr/bin/env python3
"""Small repeatable target for Linux perf profiling of the Rust hot path."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aglm_tokenizer.core.tokenizer import AGLMUniversalTokenizer
from aglm_tokenizer.native import AGLMNativeAccelerator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-file", required=True)
    parser.add_argument("--tokenizer", default="exported_tokenizers/aglm_universal_max")
    parser.add_argument("--sample-bytes", type=int, default=8 << 20)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    tokenizer = AGLMUniversalTokenizer.load(args.tokenizer)
    native = AGLMNativeAccelerator(tokenizer)
    with Path(args.sample_file).open("rb") as handle:
        text = handle.read(args.sample_bytes).decode("utf-8")
    digest = hashlib.sha256()
    started = time.perf_counter()
    for _ in range(args.iterations):
        digest.update(native.encode_fast_u32_bytes(text))
    elapsed = time.perf_counter() - started
    raw_bytes = len(text.encode("utf-8")) * args.iterations
    print(
        f"raw_bytes={raw_bytes} seconds={elapsed:.6f} "
        f"raw_mib_s={raw_bytes / (1 << 20) / elapsed:.6f} sha256={digest.hexdigest()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
