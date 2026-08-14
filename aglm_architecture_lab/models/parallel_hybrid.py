"""
Architecture D: Parallel Hybrid with Gated Attention-SSM Fusion.
In every block:
    A = Attention(Norm(x))
    M = Mamba(Norm(x))
    g = sigmoid(W_gate(x))
    Y = g * A + (1 - g) * M
"""

from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
from aglm_architecture_lab.models.config import ArchConfig
from aglm_architecture_lab.models.layers import RMSNorm, SwiGLU, GQAttention, SelectiveSSM, get_rope_cache


class ParallelBlock(nn.Module):
    def __init__(self, cfg: ArchConfig):
        super().__init__()
        self.norm_in = RMSNorm(cfg.d_model)
        self.norm_attn = RMSNorm(cfg.d_model)
        self.norm_ssm = RMSNorm(cfg.d_model)

        self.attn = GQAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            d_head=cfg.d_head,
            rope_base=cfg.rope_base
        )
        self.ssm = SelectiveSSM(
            d_model=cfg.d_model,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.expand
        )

        # Dynamic Gating Network
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # FFN
        self.norm_ffn = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ffn)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        ssm_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], torch.Tensor]:
        h_norm = self.norm_in(x)

        # 1. Parallel execution
        attn_out, next_kv = self.attn(self.norm_attn(h_norm), cos=cos, sin=sin, kv_cache=kv_cache)
        ssm_out, next_ssm_st = self.ssm(self.norm_ssm(h_norm), state=ssm_state)

        # 2. Gated Fusion: g in [0, 1] per channel
        g = torch.sigmoid(self.gate_proj(h_norm))
        fused = g * attn_out + (1.0 - g) * ssm_out

        # 3. Residual & FFN
        x = x + fused
        x = x + self.ffn(self.norm_ffn(x))

        return x, (next_kv, next_ssm_st), g


class ParallelHybridLM(nn.Module):
    """Parallel Hybrid Language Model."""
    def __init__(self, cfg: ArchConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embed.weight, std=0.02)

        self.blocks = nn.ModuleList([ParallelBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.last_gate_stats: List[Dict[str, float]] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[List[Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, List[Any]]:
        B, T = input_ids.shape
        x = self.embed(input_ids)
        cos, sin = get_rope_cache(T, self.cfg.d_head, x.device, x.dtype, self.cfg.rope_base)

        new_states = []
        gate_stats = []
        for i, block in enumerate(self.blocks):
            st = states[i] if states is not None else (None, None)
            kv_st, ssm_st = st
            x, next_st, g = block(x, cos=cos, sin=sin, kv_cache=kv_st, ssm_state=ssm_st)
            new_states.append(next_st)

            # Record gate diagnostics
            with torch.no_grad():
                gate_stats.append({
                    "layer": i,
                    "gate_mean": float(g.mean().item()),
                    "gate_std": float(g.std().item()),
                    "gate_min": float(g.min().item()),
                    "gate_max": float(g.max().item()),
                })

        self.last_gate_stats = gate_stats
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, new_states
