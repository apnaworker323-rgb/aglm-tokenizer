"""
Unified Controlled Architecture Trainer.
Guarantees identical optimization, precision, data batches, and step conditions.
"""

from typing import Dict, Any, List, Optional
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from aglm_architecture_lab.data.dataloader import MultilingualTextDataset
from aglm_architecture_lab.eval.metrics_profiler import ArchitectureProfiler, compute_validation_bpb


class ArchitectureTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: MultilingualTextDataset,
        val_dataset: MultilingualTextDataset,
        device: torch.device,
        lr: float = 6e-4,
        weight_decay: float = 0.1,
        max_steps: int = 500,
        batch_size: int = 4,
        grad_accum_steps: int = 2,
        clip_grad: float = 1.0,
        warmup_steps: int = 50,
        log_interval: int = 25,
        val_interval: int = 100
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.lr = lr
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.clip_grad = clip_grad
        self.log_interval = log_interval
        self.val_interval = val_interval

        # Optimizer with weight decay exclusion on 1D params/norms
        decay_params = []
        nodecay_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)

        self.optimizer = AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max_steps, eta_min=lr * 0.05)
        self.profiler = ArchitectureProfiler(device)

    def evaluate(self, num_batches: int = 10) -> Dict[str, float]:
        """Evaluates model on deterministic validation set."""
        self.model.eval()
        total_loss = 0.0
        total_toks = 0
        total_bytes = 0

        with torch.no_grad():
            for _ in range(num_batches):
                inp, tgt, toks, raw_b = self.val_dataset.get_batch(self.batch_size, self.device)
                with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32):
                    out = self.model(inp)
                    logits = out[0] if isinstance(out, (tuple, list)) else out
                    loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), tgt.view(-1))
                total_loss += loss.item() * toks
                total_toks += toks
                total_bytes += raw_b

        avg_loss = total_loss / max(1, total_toks)
        val_bpb = compute_validation_bpb(avg_loss, total_toks, total_bytes)
        return {
            "val_loss": avg_loss,
            "val_bpb": val_bpb,
            "val_tokens": total_toks,
            "val_bytes": total_bytes
        }

    def train(self) -> Dict[str, Any]:
        """Runs the training loop and returns execution telemetry."""
        self.model.train()
        self.profiler.reset_memory_stats()

        step_times = []
        tokens_processed = 0
        bytes_processed = 0
        start_time = time.perf_counter()

        history = []
        nan_detected = False

        for step in range(1, self.max_steps + 1):
            t0 = time.perf_counter()
            self.optimizer.zero_grad()

            accum_loss = 0.0
            for _ in range(self.grad_accum_steps):
                inp, tgt, toks, raw_b = self.train_dataset.get_batch(self.batch_size, self.device)
                tokens_processed += toks
                bytes_processed += raw_b

                with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32):
                    out = self.model(inp)
                    logits = out[0] if isinstance(out, (tuple, list)) else out
                    ce_loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), tgt.view(-1))

                    # Include aux loss if present (e.g. AGLM router capacity)
                    aux_loss = torch.tensor(0.0, device=self.device)
                    if isinstance(out, tuple) and len(out) > 1 and isinstance(out[1], dict) and "aux_loss" in out[1]:
                        aux_loss = out[1]["aux_loss"]

                    total_loss = (ce_loss + 0.1 * aux_loss) / self.grad_accum_steps

                total_loss.backward()
                accum_loss += ce_loss.item() / self.grad_accum_steps

            # Check for NaN / Inf
            if math.isnan(accum_loss) or math.isinf(accum_loss):
                nan_detected = True
                print(f"[ERROR] NaN/Inf detected at step {step}!")
                break

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.optimizer.step()
            self.scheduler.step()

            step_time = time.perf_counter() - t0
            step_times.append(step_time)

            if step % self.log_interval == 0 or step == self.max_steps:
                mem = self.profiler.measure_memory()
                tok_per_sec = (self.batch_size * self.grad_accum_steps * self.train_dataset.seq_len) / step_time
                mb_per_sec = (bytes_processed / (time.perf_counter() - start_time)) / (1024 * 1024)
                print(f"Step {step:4d}/{self.max_steps} | Loss: {accum_loss:.4f} | Throughput: {tok_per_sec:.0f} tok/s ({mb_per_sec:.2f} MB/s) | VRAM: {mem['peak_allocated_mb']:.1f} MB")

        elapsed_total = time.perf_counter() - start_time
        val_metrics = self.evaluate()
        mem_final = self.profiler.measure_memory()

        return {
            "final_train_loss": accum_loss,
            "val_loss": val_metrics["val_loss"],
            "val_bpb": val_metrics["val_bpb"],
            "train_tokens_per_sec": tokens_processed / max(1e-5, elapsed_total),
            "train_mb_per_sec": (bytes_processed / (1024 * 1024)) / max(1e-5, elapsed_total),
            "peak_vram_mb": mem_final["peak_allocated_mb"],
            "step_time_ms_p50": float(torch.tensor(step_times).median().item()) * 1000.0 if step_times else 0.0,
            "nan_instability": nan_detected,
            "total_elapsed_sec": elapsed_total,
        }
