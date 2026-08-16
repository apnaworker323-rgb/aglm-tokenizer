"""Lossless, strictly causal hierarchical patching over frozen AGLM token IDs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from benchmark_aglm_gpu_training import CausalTransformerBlock, RMSNorm


VOCAB_SIZE = 1_551_017
MAX_TOKEN_ID = VOCAB_SIZE - 1
MAX_PATCH_TOKENS = 8


@dataclass(frozen=True)
class PatchPolicy:
    name: str
    kind: Literal["fixed", "entropy"]
    fixed_size: int = 1
    entropy_threshold_bits: float = 0.0
    max_patch_tokens: int = MAX_PATCH_TOKENS

    def __post_init__(self) -> None:
        if self.kind == "fixed" and not 1 <= self.fixed_size <= MAX_PATCH_TOKENS:
            raise ValueError("fixed patch size must be in [1, 8]")
        if self.kind == "entropy" and self.entropy_threshold_bits <= 0:
            raise ValueError("entropy threshold must be positive")
        if not 1 <= self.max_patch_tokens <= MAX_PATCH_TOKENS:
            raise ValueError("maximum patch size must be in [1, 8]")


@dataclass
class LosslessPatchBatch:
    """One global position per row plus an exactly reversible local code.

    ``exact_latents[..., :8]`` stores integer token IDs as FP32. Every valid AGLM
    ID is below 2**21 and therefore exactly representable by IEEE FP32 (exact
    integer range through 2**24). Channel 8 stores the patch length. The learned
    global summary is derived from these IDs but the exact code is never rounded
    to BF16, so codec reconstruction does not depend on model weights.
    """

    exact_latents: torch.Tensor  # [B, P, 9], float32
    token_ids: torch.Tensor  # [B, P, 8], int64
    token_mask: torch.Tensor  # [B, P, 8], bool
    patch_mask: torch.Tensor  # [B, P], bool
    lengths: torch.Tensor  # [B, P], int64
    original_length: int
    policy_name: str

    @property
    def global_positions(self) -> int:
        return int(self.patch_mask.sum().item())

    @property
    def token_count(self) -> int:
        return int(self.token_mask.sum().item())


class LosslessLatentPatcher:
    def __init__(self, entropy_bits: Optional[np.ndarray] = None) -> None:
        if entropy_bits is not None:
            entropy_bits = np.asarray(entropy_bits, dtype=np.float32)
            if entropy_bits.shape != (VOCAB_SIZE,) or not np.all(np.isfinite(entropy_bits)):
                raise ValueError("entropy table must be a finite float32 vector indexed by frozen token ID")
        self.entropy_bits = entropy_bits

    @staticmethod
    def _validate(ids: torch.Tensor) -> None:
        if ids.ndim != 2 or ids.dtype != torch.long:
            raise TypeError("token IDs must be int64 [batch, sequence]")
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) > MAX_TOKEN_ID):
            raise ValueError("token ID outside the frozen AGLM vocabulary")

    def encode(self, ids: torch.Tensor, policy: PatchPolicy) -> LosslessPatchBatch:
        self._validate(ids)
        ids_cpu = ids.detach().to("cpu")
        batch, sequence = ids_cpu.shape
        patches: list[list[list[int]]] = []
        max_patches = 0
        for row in ids_cpu.tolist():
            if policy.kind == "fixed":
                row_patches = [row[start : start + policy.fixed_size] for start in range(0, sequence, policy.fixed_size)]
            else:
                if self.entropy_bits is None:
                    raise RuntimeError("entropy policy requires a frozen entropy table")
                row_patches = []
                current: list[int] = []
                information = 0.0
                for token_id in row:
                    current.append(token_id)
                    information += float(self.entropy_bits[token_id])
                    if information >= policy.entropy_threshold_bits or len(current) >= policy.max_patch_tokens:
                        row_patches.append(current)
                        current = []
                        information = 0.0
                if current:
                    row_patches.append(current)
            patches.append(row_patches)
            max_patches = max(max_patches, len(row_patches))

        token_ids = torch.zeros((batch, max_patches, MAX_PATCH_TOKENS), dtype=torch.long)
        token_mask = torch.zeros((batch, max_patches, MAX_PATCH_TOKENS), dtype=torch.bool)
        lengths = torch.zeros((batch, max_patches), dtype=torch.long)
        for batch_index, row_patches in enumerate(patches):
            for patch_index, values in enumerate(row_patches):
                width = len(values)
                token_ids[batch_index, patch_index, :width] = torch.tensor(values, dtype=torch.long)
                token_mask[batch_index, patch_index, :width] = True
                lengths[batch_index, patch_index] = width
        patch_mask = lengths > 0
        exact_latents = torch.zeros((batch, max_patches, MAX_PATCH_TOKENS + 1), dtype=torch.float32)
        exact_latents[..., :MAX_PATCH_TOKENS] = token_ids.to(torch.float32)
        exact_latents[..., MAX_PATCH_TOKENS] = lengths.to(torch.float32)
        encoded = LosslessPatchBatch(
            exact_latents=exact_latents,
            token_ids=token_ids,
            token_mask=token_mask,
            patch_mask=patch_mask,
            lengths=lengths,
            original_length=sequence,
            policy_name=policy.name,
        )
        reconstructed = self.decode(encoded)
        if not torch.equal(reconstructed, ids_cpu):
            raise RuntimeError("lossless patch codec invariant failed")
        return encoded

    @staticmethod
    def decode(batch: LosslessPatchBatch) -> torch.Tensor:
        rounded = batch.exact_latents.round().to(torch.long)
        lengths = rounded[..., MAX_PATCH_TOKENS]
        if not torch.equal(lengths, batch.lengths):
            raise RuntimeError("exact latent length corruption")
        if lengths.numel() and (int(lengths.min()) < 0 or int(lengths.max()) > MAX_PATCH_TOKENS):
            raise RuntimeError("invalid exact latent patch length")
        decoded_ids = rounded[..., :MAX_PATCH_TOKENS]
        if decoded_ids.numel() and (int(decoded_ids.min()) < 0 or int(decoded_ids.max()) > MAX_TOKEN_ID):
            raise RuntimeError("exact latent token corruption")
        rows = []
        for batch_index in range(decoded_ids.shape[0]):
            values = decoded_ids[batch_index][batch.token_mask[batch_index]]
            rows.append(values[: batch.original_length])
        return torch.stack(rows)


class HierarchicalLatentLM(nn.Module):
    """Patch encoder + shifted causal global model + causal local token decoder."""

    def __init__(
        self,
        d_model: int = 160,
        d_lexical: int = 24,
        n_layers: int = 6,
        n_heads: int = 5,
        max_global_positions: int = 512,
        cutoffs: Tuple[int, ...] = (16_384, 131_072, 524_288),
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.lexical = nn.Embedding(VOCAB_SIZE, d_lexical)
        self.lexical_projection = nn.Linear(d_lexical, d_model, bias=False)
        self.slot_embedding = nn.Parameter(torch.empty(MAX_PATCH_TOKENS, d_model))
        self.patch_projection = nn.Linear(d_model, d_model, bias=False)
        self.global_bos = nn.Parameter(torch.empty(d_model))
        self.global_positions = nn.Parameter(torch.empty(max_global_positions, d_model))
        self.blocks = nn.ModuleList(CausalTransformerBlock(d_model, n_heads, 8 / 3) for _ in range(n_layers))
        self.global_norm = RMSNorm(d_model)
        self.local_start = nn.Parameter(torch.empty(d_model))
        self.local_decoder = nn.GRU(d_model, d_model, batch_first=True)
        self.local_norm = RMSNorm(d_model)
        self.output = nn.AdaptiveLogSoftmaxWithLoss(
            d_model, VOCAB_SIZE, list(cutoffs), div_value=4.0, head_bias=False,
        )
        nn.init.normal_(self.lexical.weight, std=0.02)
        nn.init.normal_(self.slot_embedding, std=0.02)
        nn.init.normal_(self.global_bos, std=0.02)
        nn.init.normal_(self.global_positions, std=0.02)
        nn.init.normal_(self.local_start, std=0.02)

    def token_hidden(self, patches: LosslessPatchBatch, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = patches.token_ids.to(device)
        mask = patches.token_mask.to(device)
        patch_mask = patches.patch_mask.to(device)
        batch, patch_count, width = ids.shape
        if patch_count > self.global_positions.shape[0]:
            raise ValueError("patch sequence exceeds configured global position table")

        local = self.lexical_projection(self.lexical(ids))
        local = local + self.slot_embedding[:width].view(1, 1, width, self.d_model)
        weights = mask.unsqueeze(-1).to(local.dtype)
        summaries = (local * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1)
        summaries = self.patch_projection(summaries)

        shifted = torch.empty_like(summaries)
        shifted[:, 0] = self.global_bos
        if patch_count > 1:
            shifted[:, 1:] = summaries[:, :-1]
        global_state = shifted + self.global_positions[:patch_count].unsqueeze(0)
        # Padding patches are suffix-only. They cannot affect an earlier causal state.
        global_state = global_state * patch_mask.unsqueeze(-1).to(global_state.dtype)
        for block in self.blocks:
            global_state = block(global_state)
        conditions = self.global_norm(global_state)

        decoder_inputs = torch.empty_like(local)
        decoder_inputs[:, :, 0] = self.local_start
        if width > 1:
            decoder_inputs[:, :, 1:] = local[:, :, :-1]
        flat_inputs = decoder_inputs.reshape(batch * patch_count, width, self.d_model)
        initial = conditions.reshape(1, batch * patch_count, self.d_model).contiguous()
        decoded, _ = self.local_decoder(flat_inputs, initial)
        decoded = self.local_norm(decoded).reshape(batch, patch_count, width, self.d_model)
        return decoded[mask], ids[mask]

    def loss(self, patches: LosslessPatchBatch, device: torch.device) -> torch.Tensor:
        hidden, targets = self.token_hidden(patches, device)
        return self.output(hidden, targets).loss

    @torch.no_grad()
    def predict_tokens(self, patches: LosslessPatchBatch, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden, targets = self.token_hidden(patches, device)
        return self.output.predict(hidden), targets


def entropy_table_from_counts(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    if counts.shape != (VOCAB_SIZE,) or np.any(counts < 0):
        raise ValueError("frequency counts must index the frozen vocabulary")
    denominator = float(counts.sum() + VOCAB_SIZE)
    return (-np.log2((counts + 1.0) / denominator)).astype(np.float32)


def calibrate_entropy_threshold(ids: torch.Tensor, entropy_bits: np.ndarray, target_mean_tokens: float = 4.0) -> float:
    if not 1 <= target_mean_tokens <= MAX_PATCH_TOKENS:
        raise ValueError("target mean patch size must be in [1, 8]")
    values = entropy_bits[ids.detach().cpu().numpy()]
    return float(values.mean() * target_mean_tokens)


def bits_per_byte(mean_loss_nats: float, token_count: int, raw_bytes: int) -> float:
    if raw_bytes <= 0:
        raise ValueError("raw byte count must be positive")
    return mean_loss_nats * token_count / (math.log(2.0) * raw_bytes)
