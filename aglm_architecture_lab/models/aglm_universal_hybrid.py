"""
Architecture E: AGLM-Specific Universal Hybrid Architecture.
Combines:
1. Large Multilingual Token Representation with Factorized Lexical Embeddings (d_lexical -> d_model).
2. Global Recurrent Mamba-3 Memory (processes ALL tokens, state-safe, exact sequence continuity).
3. Selective Causal Threshold-Routed Attention Refinement (processes informative tokens for exact retrieval).
4. Efficient SwiGLU FFN & Causal Residual Connections.
"""

from typing import Optional, Tuple, Dict, Any, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from aglm_architecture_lab.models.config import ArchConfig
from aglm_architecture_lab.models.layers import RMSNorm, SwiGLU, GQAttention, SelectiveSSM, CausalThresholdRouter, get_rope_cache


class FactorizedLexicalEmbedding(nn.Module):
    """
    Factorized Lexical Input Representation.
    Token ID -> nn.Embedding(V, d_lexical) -> Linear(d_lexical, d_model) -> RMSNorm.
    Prevents parameter explosion when scaling vocab to 256K / 1.55M.
    """
    def __init__(self, vocab_size: int, d_model: int, d_lexical: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_lexical = d_lexical
        self.lexical_table = nn.Embedding(vocab_size, d_lexical)
        self.proj = nn.Linear(d_lexical, d_model, bias=False)
        self.norm = RMSNorm(d_model)
        nn.init.normal_(self.lexical_table.weight, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        lexical = self.lexical_table(token_ids)
        return self.norm(self.proj(lexical))


class AGLMRefinementBlock(nn.Module):
    """Transformer block with gathered causal position handling."""
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
        pos: torch.Tensor
    ) -> torch.Tensor:
        normed = self.norm_attn(x)
        attn_out, _ = self.attn(normed, cos=cos, sin=sin, pos=pos)
        x = x + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x


class AGLMUniversalHybridLM(nn.Module):
    """
    AGLM Universal Production Hybrid LM:
    - Input: Factorized Lexical Embeddings
    - Memory Pass: 2 Full Mamba SSM Blocks (all tokens, exact recurrence)
    - Refinement Pass: Recursively shared Causal-Routed GQA Blocks
    - Output: Scaled LM Head
    """
    def __init__(self, cfg: ArchConfig):
        super().__init__()
        self.cfg = cfg
        self.tau = cfg.tau_threshold
        self.target_cap = cfg.target_capacity
        self.recursions = cfg.recursions

        # 1. Input representation
        if cfg.factorize_embeddings:
            self.embed = FactorizedLexicalEmbedding(cfg.vocab_size, cfg.d_model, cfg.d_lexical)
        else:
            self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
            nn.init.normal_(self.embed.weight, std=0.02)

        # 2. Global Recurrent Memory Pass (Mamba SSM)
        self.n_mem = cfg.extra.get("n_mem_layers", 2)
        self.mem_blocks = nn.ModuleList([
            nn.ModuleDict({
                "norm_ssm": RMSNorm(cfg.d_model),
                "ssm": SelectiveSSM(cfg.d_model, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand),
                "norm_ffn": RMSNorm(cfg.d_model),
                "ffn": SwiGLU(cfg.d_model, cfg.d_ffn),
            })
            for _ in range(self.n_mem)
        ])

        # 3. Causal Threshold Routers
        self.routers = nn.ModuleList([
            CausalThresholdRouter(cfg.d_model, tau=self.tau)
            for _ in range(self.recursions)
        ])

        # 4. Refinement Blocks
        self.n_ref = cfg.extra.get("n_ref_layers", 2)
        self.ref_blocks = nn.ModuleList([
            AGLMRefinementBlock(cfg)
            for _ in range(self.n_ref)
        ])

        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.aux_loss: torch.Tensor = torch.tensor(0.0)
        self.last_routing_stats: List[Dict[str, float]] = []

    def _route_causal(self, gate: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Causal threshold selection."""
        B, T = gate.shape
        mask = gate > self.tau
        n_selected = mask.sum(dim=1)  # (B,)
        K = max(1, int(n_selected.max().item()))

        ar = torch.arange(T, device=gate.device).unsqueeze(0).expand(B, T)
        # Sort so selected tokens appear first
        sort_key = torch.where(mask, ar, ar + T)
        sel_indices = sort_key.argsort(dim=1)[:, :K]
        valid_mask = torch.arange(K, device=gate.device).unsqueeze(0).expand(B, K) < n_selected.unsqueeze(1)
        return sel_indices, valid_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[Any] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        B, T = input_ids.shape
        x = self.embed(input_ids)

        # 1. Global Memory Pass (Full Sequence)
        for block in self.mem_blocks:
            normed = block["norm_ssm"](x)
            ssm_out, _ = block["ssm"](normed)
            x = x + ssm_out
            x = x + block["ffn"](block["norm_ffn"](x))

        cos, sin = get_rope_cache(T, self.cfg.d_head, x.device, x.dtype, self.cfg.rope_base)

        # 2. Causal Selective Refinement Pass
        aux_loss = torch.tensor(0.0, device=x.device)
        routing_stats = []

        for r in range(self.recursions):
            gate, mask = self.routers[r](x)
            sel_idx, valid = self._route_causal(gate)  # (B, K)

            # Gather routed subset
            gidx = sel_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            xs = x.gather(1, gidx)
            gs = gate.gather(1, sel_idx).unsqueeze(-1) * valid.unsqueeze(-1)

            hs = xs
            for ref_block in self.ref_blocks:
                hs = ref_block(hs, cos=cos, sin=sin, pos=sel_idx)

            # Causal Scatter-Add
            x = x.scatter_add(1, gidx, (gs * (hs - xs)).to(x.dtype))

            # Router Diagnostics & Aux Loss
            aux_loss = aux_loss + (gate.mean() - self.target_cap).pow(2)
            routed_pct = float(valid.float().mean().item()) * 100.0
            routing_stats.append({
                "recursion": r,
                "routed_pct": routed_pct,
                "gate_mean": float(gate.mean().item()),
                "gate_std": float(gate.std().item()),
            })

        self.aux_loss = aux_loss / self.recursions
        self.last_routing_stats = routing_stats

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, {"aux_loss": self.aux_loss, "routing_stats": routing_stats}
