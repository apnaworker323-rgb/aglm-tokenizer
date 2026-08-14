"""
Binary Memory-Mapped Dataset Loader for High-Throughput AGLM Training.
Streams pre-tokenized binary files (.bin) directly into GPU tensors with zero RAM overhead.
"""

from typing import Tuple
import os
import json
import torch
import numpy as np


class BinaryMmapDataset:
    """Streams token sequences directly from memory-mapped numpy binary files."""

    def __init__(self, bin_path: str, seq_len: int = 256, vocab_size: int = 32768):
        self.bin_path = bin_path
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Binary token file not found: {bin_path}")

        # Memory map the binary file
        self.tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.total_tokens = len(self.tokens)
        self.total_sequences = max(1, (self.total_tokens - 1) // self.seq_len)

    def __len__(self) -> int:
        return self.total_sequences

    def get_batch(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Extracts a random causal next-token batch from the memory-mapped tokens.
        Returns:
            input_ids: (B, T)
            target_ids: (B, T)
            total_tokens: int
            total_bytes: int
        """
        max_start = self.total_tokens - self.seq_len - 1
        starts = np.random.randint(0, max_start, size=batch_size)

        inps = []
        tgts = []
        for s in starts:
            chunk = self.tokens[s : s + self.seq_len + 1].astype(np.int64)
            # Clamp to vocab_size
            chunk = chunk % self.vocab_size
            inps.append(chunk[:-1])
            tgts.append(chunk[1:])

        inp_tensor = torch.tensor(np.array(inps), dtype=torch.long, device=device)
        tgt_tensor = torch.tensor(np.array(tgts), dtype=torch.long, device=device)
        tot_toks = batch_size * self.seq_len
        tot_bytes = tot_toks  # 1 byte per token for byte-level representation

        return inp_tensor, tgt_tensor, tot_toks, tot_bytes
