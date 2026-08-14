"""
Configuration and parameter budgeting for iso-parameter architecture comparisons.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import math


@dataclass
class ArchConfig:
    name: str
    vocab_size: int = 32768
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 2
    d_head: int = 64
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    ffn_mult: float = 8 / 3
    seq_len: int = 512
    dropout: float = 0.0
    rope_base: float = 10000.0
    # Embedding Factorization
    factorize_embeddings: bool = False
    d_lexical: int = 128
    # Routing / Sparsity for AGLM Hybrid
    tau_threshold: float = 0.5
    target_capacity: float = 0.5
    recursions: int = 2
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def d_ffn(self) -> int:
        return int(self.d_model * self.ffn_mult / 64 + 0.5) * 64

    def copy_with(self, **kwargs) -> "ArchConfig":
        d = self.__dict__.copy()
        d.update(kwargs)
        return ArchConfig(**d)


def count_parameters(model: Any) -> Dict[str, int]:
    """Breakdown parameters into Embedding, Backbone, and Head."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    embed_params = 0
    if hasattr(model, "embed"):
        embed_params = sum(p.numel() for p in model.embed.parameters())

    head_params = 0
    if hasattr(model, "lm_head"):
        head_params = sum(p.numel() for p in model.lm_head.parameters())

    backbone_params = total_params - embed_params - head_params

    return {
        "total": total_params,
        "trainable": trainable_params,
        "embedding": embed_params,
        "backbone": backbone_params,
        "head": head_params,
    }
