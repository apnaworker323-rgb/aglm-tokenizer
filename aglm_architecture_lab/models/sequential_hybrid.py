"""
Architecture B: Sequential Mamba-Heavy Hybrid (3 Mamba Blocks : 1 Attention Block).
"""

from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
from aglm_architecture_lab.models.config import ArchConfig
from aglm_architecture_lab.models.layers import RMSNorm, SwiGLU, GQAttention, SelectiveSSM, get_rope_cache


class MambaBlock(nn.Module):
    def __init__(self, cfg: ArchConfig):
        super().__init__()
        self.norm_ssm = RMSNorm(cfg.d_model)
        self.ssm = SelectiveSSM(
            d_model=cfg.d_model,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.expand
        )
        self.norm_ffn = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ffn)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normed = self.norm_ssm(x)
        ssm_out, next_state = self.ssm(normed, state=state)
        x = x + ssm_out
        x = x + self.ffn(self.norm_ffn(x))
        return x, next_state


class AttentionBlock(nn.Module):
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
        attn_out, next_kv = self.attn(normed, cos=cos, sin=sin, kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x, next_kv


class SequentialHybridLM(nn.Module):
    """Sequential Hybrid Language Model (3 Mamba : 1 Attention)."""
    def __init__(self, cfg: ArchConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embed.weight, std=0.02)

        self.blocks = nn.ModuleList()
        self.block_types = []  # "ssm" or "attn"

        # 3 Mamba : 1 Attention layout across n_layers
        for i in range(cfg.n_layers):
            if (i + 1) % 4 == 0:
                self.blocks.append(AttentionBlock(cfg))
                self.block_types.append("attn")
            else:
                self.blocks.append(MambaBlock(cfg))
                self.block_types.append("ssm")

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
