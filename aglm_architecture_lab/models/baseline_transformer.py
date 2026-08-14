"""
Architecture A: Iso-Parameter Transformer Baseline with GQA and SwiGLU.
"""

import math
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
from aglm_architecture_lab.models.config import ArchConfig
from aglm_architecture_lab.models.layers import RMSNorm, SwiGLU, GQAttention, get_rope_cache


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ArchConfig):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model)
        self.attn = GQAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            d_head=cfg.d_head,
            rope_base=cfg.rope_base
        )
        self.norm_ffn = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ffn)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        normed = self.norm_attn(x)
        attn_out, new_kv = self.attn(normed, cos=cos, sin=sin, kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x, new_kv


class BaselineTransformerLM(nn.Module):
    """Full Transformer Baseline Language Model."""
    def __init__(self, cfg: ArchConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embed.weight, std=0.02)

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Tie weights optionally or untie
        if cfg.extra.get("tie_embeddings", False):
            self.lm_head.weight = self.embed.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T = input_ids.shape
        x = self.embed(input_ids)
        cos, sin = get_rope_cache(T, self.cfg.d_head, x.device, x.dtype, self.cfg.rope_base)

        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            kv = kv_caches[i] if kv_caches is not None else None
            x, next_kv = block(x, cos=cos, sin=sin, kv_cache=kv)
            new_kv_caches.append(next_kv)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, new_kv_caches
