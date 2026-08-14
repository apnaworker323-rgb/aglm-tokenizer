"""
Architecture C: Alternating Hybrid (1 Mamba Block : 1 Attention Block).
"""

from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
from aglm_architecture_lab.models.config import ArchConfig
from aglm_architecture_lab.models.layers import RMSNorm, SwiGLU, GQAttention, SelectiveSSM, get_rope_cache
from aglm_architecture_lab.models.sequential_hybrid import MambaBlock, AttentionBlock


class AlternatingHybridLM(nn.Module):
    """Alternating Hybrid (1 Mamba : 1 Attention)."""
    def __init__(self, cfg: ArchConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embed.weight, std=0.02)

        self.blocks = nn.ModuleList()
        self.block_types = []

        # 1:1 Alternating layout
        for i in range(cfg.n_layers):
            if i % 2 == 0:
                self.blocks.append(MambaBlock(cfg))
                self.block_types.append("ssm")
            else:
                self.blocks.append(AttentionBlock(cfg))
                self.block_types.append("attn")

        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[List[Any]] = None
    ) -> Tuple[torch.Tensor, List[Any]]:
        B, T = input_ids.shape
        x = self.embed(input_ids)
        cos, sin = get_rope_cache(T, self.cfg.d_head, x.device, x.dtype, self.cfg.rope_base)

        new_states = []
        for i, (b_type, block) in enumerate(zip(self.block_types, self.blocks)):
            st = states[i] if states is not None else None
            if b_type == "ssm":
                x, nxt_st = block(x, state=st)
            else:
                x, nxt_st = block(x, cos=cos, sin=sin, kv_cache=st)
            new_states.append(nxt_st)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, new_states
