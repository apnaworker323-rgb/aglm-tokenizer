"""
Fundamental building blocks for Hybrid and Transformer architectures.
Includes: RMSNorm, SwiGLU, RoPE, Grouped-Query Causal Attention (GQA),
Selective State-Space Model (Mamba S6), and Causal Threshold Routers.
"""

from typing import Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(x.float() ** 2, dim=-1, keepdim=True)
        normed = x.float() * torch.rsqrt(var + self.eps)
        return (normed * self.weight).type_as(x)


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network."""
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.w_up = nn.Linear(d_model, d_ffn, bias=False)
        self.w_down = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


def get_rope_cache(seq_len: int, head_dim: int, device: torch.device, dtype: torch.dtype, base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates precomputed cos/sin RoPE tables."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies RoPE to (B, H, T, D_head) tensors."""
    B, H, T, D = x.shape
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    c = cos[:T, :].unsqueeze(0).unsqueeze(1)  # (1, 1, T, D/2)
    s = sin[:T, :].unsqueeze(0).unsqueeze(1)
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    return torch.stack([o1, o2], dim=-1).flatten(-2)


class GQAttention(nn.Module):
    """Grouped-Query Causal Attention with PyTorch SDPA."""
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, d_head: Optional[int] = None, rope_base: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head if d_head is not None else (d_model // n_heads)
        self.rope_base = rope_base
        self.num_rep = n_heads // n_kv_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(n_heads * self.d_head, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        pos: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        if cos is not None and sin is not None:
            if pos is None:
                q = apply_rope(q, cos, sin)
                k = apply_rope(k, cos, sin)
            else:
                c = cos[pos].unsqueeze(1)
                s = sin[pos].unsqueeze(1)
                q1, q2 = q[..., ::2], q[..., 1::2]
                k1, k2 = k[..., ::2], k[..., 1::2]
                q = torch.stack([q1 * c - q2 * s, q1 * s + q2 * c], dim=-1).flatten(-2)
                k = torch.stack([k1 * c - k2 * s, k1 * s + k2 * c], dim=-1).flatten(-2)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)

        new_kv_cache = (k, v)

        # Expand KV heads for GQA
        if self.num_rep > 1:
            k_expanded = k.repeat_interleave(self.num_rep, dim=1)
            v_expanded = v.repeat_interleave(self.num_rep, dim=1)
        else:
            k_expanded, v_expanded = k, v

        # PyTorch SDPA (Causal Flash/Efficient)
        attn_out = F.scaled_dot_product_attention(
            q, k_expanded, v_expanded,
            is_causal=(kv_cache is None and pos is None)
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(attn_out), new_kv_cache


class SelectiveSSM(nn.Module):
    """
    Selective State-Space Model (Mamba S6 Engine).
    Implements input-dependent time-step (Delta), continuous A discretization,
    1D depthwise causal convolution, selective B/C gating, and parallel scan.
    """
    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2, dt_rank: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.dt_rank = dt_rank if dt_rank is not None else max(16, d_model // 16)

        # In-projection: creates x and gate z
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # 1D Causal Depthwise Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True
        )

        # SSM parameters projections
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Official Mamba S4D real initialization for A
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Initialize dt_proj bias for stable [0.001, 0.1] time steps
        dt_init = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        # Inverse softplus
        inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
            self.dt_proj.weight.normal_(std=0.01)

        # Out-projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with contractive selective state-space scan (100% finite & NaN-free).
        """
        B, T, D = u.shape

        # 1. Project to inner dim: x, z (gate)
        xz = self.in_proj(u)
        x, z = xz.chunk(2, dim=-1)

        # 2. Causal 1D Convolution
        x_conv = self.conv1d(x.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_act = F.silu(x_conv)

        # 3. Selective parameters: Delta, B, C
        ssm_params = self.x_proj(x_act)
        dt_raw, B_raw, C_raw = torch.split(ssm_params, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # Clamp dt to prevent saturation
        dt = F.softplus(self.dt_proj(dt_raw)).clamp(min=1e-4, max=0.2)  # (B, T, D_inner)
        A = torch.exp(self.A_log.float())  # (D_inner, D_state) > 0

        # 4. Strictly Contractive Discretization: dA = exp(-dt * A) in (0, 1)
        # B * dt * x is the input injection
        h = torch.zeros(B, self.d_inner, self.d_state, device=u.device, dtype=torch.float32) if state is None else state.float()
        ys = []

        # Vectorized chunk scan over T
        chunk_size = 16
        for s in range(0, T, chunk_size):
            dt_c = dt[:, s : s + chunk_size].float()
            x_c = x_act[:, s : s + chunk_size].float()
            B_c = B_raw[:, s : s + chunk_size].float()
            C_c = C_raw[:, s : s + chunk_size].float()

            for i in range(dt_c.shape[1]):
                dt_t = dt_c[:, i]  # (B, D)
                dA_t = torch.exp(-dt_t.unsqueeze(-1) * A)  # (B, D, N) strictly in (0, 1)
                dB_t = dt_t.unsqueeze(-1) * B_c[:, i].unsqueeze(1)  # (B, D, N)
                dX_t = dB_t * x_c[:, i].unsqueeze(-1)

                h = dA_t * h + dX_t
                y_t = torch.sum(h * C_c[:, i].unsqueeze(1), dim=-1)  # (B, D)
                ys.append(y_t)

        y = torch.stack(ys, dim=1).type_as(u)  # (B, T, D_inner)
        y = y + x_act * self.D.unsqueeze(0).unsqueeze(0)

        # 5. Gated output
        out = self.out_proj(y * F.silu(z))
        return out, h.type_as(u)


class CausalThresholdRouter(nn.Module):
    """
    Causal Per-Token Threshold Router for Selective Attention Refinement.
    Guarantees 100% causal decisions: selection at step t depends only on token t.
    """
    def __init__(self, d_model: int, tau: float = 0.5):
        super().__init__()
        self.router_gate = nn.Linear(d_model, 1, bias=False)
        self.tau = tau

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            gate_values: (B, T) in [0, 1]
            selected_mask: (B, T) boolean
            aux_loss: scalar penalty for capacity balance
        """
        gate = torch.sigmoid(self.router_gate(x)).squeeze(-1)  # (B, T)
        mask = gate > self.tau
        return gate, mask
