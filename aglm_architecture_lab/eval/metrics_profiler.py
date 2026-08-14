"""
Comprehensive Metrics Profiler and Evaluation Utilities.
Computes both token-oriented and raw-data-oriented metrics:
- Validation BPB (Bits Per Byte)
- Loss per token
- Training throughput (tok/s, MB/s, bytes/s)
- Peak VRAM (allocated & reserved)
- Prefill Latency & Decode Speed (ms/token)
- KV-Cache & Recurrent State Memory Footprint
"""

from typing import Dict, Any, List, Optional
import time
import math
import torch
import torch.nn as nn


class ArchitectureProfiler:
    def __init__(self, device: torch.device):
        self.device = device

    def measure_memory(self) -> Dict[str, float]:
        """Returns peak allocated and reserved VRAM in MB."""
        if self.device.type == "cuda":
            return {
                "peak_allocated_mb": torch.cuda.max_memory_allocated(self.device) / (1024 * 1024),
                "peak_reserved_mb": torch.cuda.max_memory_reserved(self.device) / (1024 * 1024),
                "current_allocated_mb": torch.cuda.memory_allocated(self.device) / (1024 * 1024),
            }
        return {"peak_allocated_mb": 0.0, "peak_reserved_mb": 0.0, "current_allocated_mb": 0.0}

    def reset_memory_stats(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def measure_inference_latencies(
        self,
        model: nn.Module,
        prompt_len: int = 256,
        gen_tokens: int = 32,
        vocab_size: int = 32768,
        warmup_runs: int = 3,
        test_runs: int = 10
    ) -> Dict[str, float]:
        """Measures Prefill Latency (ms) and Per-Token Decode Latency (ms/token)."""
        model.eval()
        dummy_prompt = torch.randint(0, vocab_size, (1, prompt_len), device=self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(dummy_prompt)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # 1. Measure Prefill Latency
        prefill_times = []
        with torch.no_grad():
            for _ in range(test_runs):
                t0 = time.perf_counter()
                out = model(dummy_prompt)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                prefill_times.append((time.perf_counter() - t0) * 1000.0)

        # 2. Measure Autoregressive Decode Latency
        decode_times = []
        curr_input = dummy_prompt
        with torch.no_grad():
            for _ in range(gen_tokens):
                t0 = time.perf_counter()
                out = model(curr_input)
                logits = out[0] if isinstance(out, tuple) else out
                next_tok = logits[:, -1:].argmax(dim=-1)
                curr_input = torch.cat([curr_input, next_tok], dim=1)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                decode_times.append((time.perf_counter() - t0) * 1000.0)

        return {
            "prefill_latency_ms": float(torch.tensor(prefill_times).median().item()),
            "decode_latency_ms_per_token": float(torch.tensor(decode_times).median().item()),
            "decode_tokens_per_sec": 1000.0 / float(torch.tensor(decode_times).median().item()),
        }


def compute_validation_bpb(
    loss_nats_per_token: float,
    total_tokens: int,
    total_utf8_bytes: int
) -> float:
    """
    Computes tokenizer-independent Bits Per Byte (BPB):
    BPB = (loss_in_nats / ln(2)) * (total_tokens / total_utf8_bytes)
    """
    if total_utf8_bytes <= 0:
        return 0.0
    bits_per_token = loss_nats_per_token / math.log(2.0)
    tokens_per_byte = total_tokens / total_utf8_bytes
    return bits_per_token * tokens_per_byte
