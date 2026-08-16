#!/usr/bin/env python3
"""Memory-mapped PyTorch loader for AGLM little-endian uint32 shards."""

from __future__ import annotations

import argparse
import bisect
import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass


UINT32_LE = np.dtype("<u4")


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


class AGLMShardedDataset(Dataset):
    """Stateless deterministic random sampling without crossing shard boundaries.

    Args:
        manifest: Path to ``metadata/dataset_manifest.json``.
        split: ``train`` or ``val``.
        seq_len: Number of input IDs; each sample reads ``seq_len + 1`` IDs.
        seed: Deterministic sampling seed.
        max_open_shards: LRU limit for open mmap objects/file descriptors.
    """

    def __init__(
        self,
        manifest: Optional[str] = None,
        split: str = "train",
        seq_len: int = 2048,
        seed: int = 42,
        max_open_shards: int = 32,
        manifest_path: Optional[str] = None,
    ):
        if torch is None:
            raise ImportError("PyTorch is required to use AGLMShardedDataset")
        selected_manifest = manifest or manifest_path
        if not selected_manifest:
            raise TypeError("manifest is required")
        self.manifest_path = str(Path(selected_manifest).expanduser().resolve())
        self.split = split
        self.seq_len = int(seq_len)
        self.seed = int(seed) & 0xFFFFFFFFFFFFFFFF
        self.max_open_shards = int(max_open_shards)
        if self.seq_len <= 0 or self.max_open_shards <= 0:
            raise ValueError("seq_len and max_open_shards must be positive")
        with open(self.manifest_path, "r", encoding="utf-8") as handle:
            self.manifest: Dict[str, Any] = json.load(handle)
        if self.manifest.get("numpy_dtype") != "<u4" or self.manifest.get("endian") != "little":
            raise ValueError("manifest is not canonical little-endian uint32 AGLM data")
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        self.root = Path(self.manifest_path).parent.parent
        rows = self.manifest.get("shards", {}).get(split, [])
        self.shards: List[Dict[str, Any]] = []
        self.cumulative_starts: List[int] = []
        total_starts = 0
        for row in rows:
            path = Path(row["path"])
            path = path if path.is_absolute() else self.root / path
            size = path.stat().st_size
            expected = int(row["token_count"]) * 4
            if size != expected or size % 4:
                raise ValueError(f"shard size does not match manifest: {path}")
            valid_starts = max(0, int(row["token_count"]) - self.seq_len)
            if not valid_starts:
                continue
            self.shards.append({**row, "_path": str(path), "_valid_starts": valid_starts})
            total_starts += valid_starts
            self.cumulative_starts.append(total_starts)
        if not self.shards:
            raise ValueError(f"no {split!r} shard is long enough for seq_len={seq_len}")
        self.total_valid_starts = total_starts
        self.total_tokens = sum(int(row["token_count"]) for row in rows)
        self.total_sequences = sum(int(row["token_count"]) // self.seq_len for row in rows)
        self._maps: "OrderedDict[int, np.memmap]" = OrderedDict()

    def __len__(self) -> int:
        return max(1, self.total_sequences)

    def _map(self, shard_index: int) -> np.memmap:
        existing = self._maps.pop(shard_index, None)
        if existing is not None:
            self._maps[shard_index] = existing
            return existing
        row = self.shards[shard_index]
        mapped = np.memmap(row["_path"], dtype=UINT32_LE, mode="r", shape=(int(row["token_count"]),))
        self._maps[shard_index] = mapped
        while len(self._maps) > self.max_open_shards:
            _, stale = self._maps.popitem(last=False)
            stale._mmap.close()  # type: ignore[attr-defined]
        return mapped

    def sample_location(self, index: int) -> Tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        draw = _splitmix64(self.seed ^ int(index)) % self.total_valid_starts
        shard_index = bisect.bisect_right(self.cumulative_starts, draw)
        previous = self.cumulative_starts[shard_index - 1] if shard_index else 0
        return shard_index, int(draw - previous)

    def __getitem__(self, index: int) -> Tuple["torch.Tensor", "torch.Tensor"]:
        shard_index, start = self.sample_location(index)
        mapped = self._map(shard_index)
        # uint32 is unsupported by many torch operations; copy/cast only the sampled window.
        window = np.asarray(mapped[start : start + self.seq_len + 1], dtype=np.int64)
        inputs = torch.from_numpy(window[:-1].copy())
        targets = torch.from_numpy(window[1:].copy())
        return inputs, targets

    def get_batch(self, batch_size: int, device: Optional["torch.device"] = None, start_index: int = 0) -> Tuple["torch.Tensor", "torch.Tensor", int]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        pairs = [self[(start_index + offset) % len(self)] for offset in range(batch_size)]
        inputs = torch.stack([pair[0] for pair in pairs])
        targets = torch.stack([pair[1] for pair in pairs])
        if device is not None:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
        return inputs, targets, batch_size * self.seq_len

    def close(self) -> None:
        while self._maps:
            _, mapped = self._maps.popitem()
            mapped._mmap.close()  # type: ignore[attr-defined]

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_maps"] = OrderedDict()
        return state

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass


def benchmark_sharded_loader(
    manifest: Optional[str] = None,
    split: str = "train",
    seq_len: int = 2048,
    batch_size: int = 8,
    num_batches: int = 100,
    seed: int = 42,
    manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    selected = manifest or manifest_path
    if not selected:
        raise TypeError("manifest is required")
    dataset = AGLMShardedDataset(manifest=selected, split=split, seq_len=seq_len, seed=seed)
    started = time.perf_counter()
    tokens = 0
    try:
        for batch in range(num_batches):
            _, _, count = dataset.get_batch(batch_size, start_index=batch * batch_size)
            tokens += count
    finally:
        dataset.close()
    elapsed = time.perf_counter() - started
    result = {
        "split": split, "seq_len": seq_len, "batch_size": batch_size, "batches": num_batches,
        "samples": batch_size * num_batches, "tokens": tokens, "elapsed_seconds": elapsed,
        "samples_per_second": batch_size * num_batches / max(elapsed, 1e-9),
        "tokens_per_second": tokens / max(elapsed, 1e-9),
        "uint32_mb_per_second": tokens * 4 / (1 << 20) / max(elapsed, 1e-9),
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark AGLM mmap shard sampling; does not train")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    benchmark_sharded_loader(args.manifest, args.split, args.seq_len, args.batch_size, args.num_batches, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
