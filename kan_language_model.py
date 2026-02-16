"""
KAN Language Model - Kolmogorov-Arnold Networks for NLP

Based on: "KAN: Kolmogorov-Arnold Networks" (Liu et al., 2024)

Key idea: Replace linear projections (y = Wx + b) with sums of learned
univariate spline functions: y_j = Σᵢ φᵢⱼ(xᵢ)

Each φᵢⱼ is a piecewise-linear B-spline parameterized by `grid_size`
control points, making the per-connection function learnable.

Bug fixes vs mission spec:
  1. Removed unused `scipy` import (not installed)
  2. Vectorized KANLinear.forward — nested Python for-loops over
     out_features × in_features are O(B·D²) Python ops (10–100× slower);
     replaced with pure-tensor gather + einsum ops.
  3. KANAttention / KANTransformerBlock: KANLinear expects 2-D input
     [batch, features]; added reshape around calls so 3-D sequences
     [batch, seq, embed] pass through correctly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core KAN layer
# ---------------------------------------------------------------------------

class KANLinear(nn.Module):
    """
    KAN layer: replaces a linear layer with learnable piecewise-linear splines.

    Instead of: y = Wx + b  (in × out parameters)
    We have:    y_j = Σᵢ φᵢⱼ(xᵢ)   (in × out × grid_size parameters)

    Each φᵢⱼ is evaluated by linear interpolation between grid_size
    learnable control points placed on a uniform grid over [-1, 1].
    """

    def __init__(self, in_features: int, out_features: int, grid_size: int = 5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size

        # Learnable spline control points: one curve per (input, output) pair
        self.spline_weights = nn.Parameter(
            torch.randn(out_features, in_features, grid_size) * 0.1
        )

        # Fixed uniform grid over [-1, 1]
        self.register_buffer('grid', torch.linspace(-1, 1, grid_size))

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., in_features]  (any leading batch dims)
        Returns:
            output: [..., out_features]
        """
        leading = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)   # [N, in]
        out = self._forward_2d(x_flat)              # [N, out]
        return out.reshape(*leading, self.out_features)

    def _forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        """Vectorized spline evaluation for 2-D input [N, in_features]."""
        N = x.size(0)
        D_in = self.in_features
        D_out = self.out_features
        G = self.grid_size

        # --- normalise to grid range ----------------------------------------
        x_norm = torch.tanh(x)                     # [N, in]
        x_c = torch.clamp(x_norm, self.grid[0], self.grid[-1])  # [N, in]

        # --- find interval indices ------------------------------------------
        # Count grid points strictly less than x_c; clamp to [0, G-2]
        # grid: [G], x_c: [N, in]
        idx = (x_c.unsqueeze(-1) >= self.grid.view(1, 1, G)).sum(dim=-1) - 1
        idx = idx.clamp(0, G - 2)                  # [N, in]

        # --- local interpolation weight t ∈ [0,1] --------------------------
        g_left  = self.grid[idx]                   # [N, in]
        g_right = self.grid[(idx + 1).clamp(max=G - 1)]  # [N, in]
        t = (x_c - g_left) / (g_right - g_left + 1e-8)   # [N, in]

        # --- gather control points -----------------------------------------
        # spline_weights: [D_out, D_in, G]
        # idx expanded:   [D_out, N, D_in, 1]
        idx_exp = idx.unsqueeze(0).unsqueeze(-1).expand(D_out, N, D_in, 1)
        sw = self.spline_weights.unsqueeze(1).expand(D_out, N, D_in, G)

        cp_left  = sw.gather(-1, idx_exp).squeeze(-1)              # [D_out, N, D_in]
        cp_right = sw.gather(-1, (idx_exp + 1).clamp(max=G-1)).squeeze(-1)

        # --- interpolate and sum over input dim ----------------------------
        t_exp = t.unsqueeze(0).expand(D_out, N, D_in)             # [D_out, N, D_in]
        values = (1 - t_exp) * cp_left + t_exp * cp_right         # [D_out, N, D_in]
        output = values.sum(dim=-1).transpose(0, 1)                # [N, D_out]

        return output

    def parameter_count(self) -> int:
        return self.in_features * self.out_features * self.grid_size


# ---------------------------------------------------------------------------
# KAN Attention
# ---------------------------------------------------------------------------

class KANAttention(nn.Module):
    """Multi-head attention where all linear projections are KAN layers."""

    def __init__(self, embed_dim: int, num_heads: int = 4, grid_size: int = 5):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj   = KANLinear(embed_dim, embed_dim, grid_size)
        self.k_proj   = KANLinear(embed_dim, embed_dim, grid_size)
        self.v_proj   = KANLinear(embed_dim, embed_dim, grid_size)
        self.out_proj = KANLinear(embed_dim, embed_dim, grid_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim]
        Returns:
            out: [batch, seq_len, embed_dim]
        """
        batch, seq_len, _ = x.shape

        # KANLinear handles arbitrary leading dims via forward()
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn   = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)                                        # [B, heads, seq, head_dim]
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)

        return self.out_proj(out)


# ---------------------------------------------------------------------------
# KAN Transformer Block
# ---------------------------------------------------------------------------

class KANTransformerBlock(nn.Module):
    """Transformer block with KAN layers replacing all linear operations."""

    def __init__(self, embed_dim: int, num_heads: int = 4, grid_size: int = 5):
        super().__init__()
        self.attention = KANAttention(embed_dim, num_heads, grid_size)
        self.norm1 = nn.LayerNorm(embed_dim)

        # Feed-forward: KAN handles [batch, seq, embed] via leading-dim support
        self.ff_kan1 = KANLinear(embed_dim, embed_dim * 4, grid_size)
        self.ff_kan2 = KANLinear(embed_dim * 4, embed_dim, grid_size)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        residual = x
        h = F.gelu(self.ff_kan1(self.norm2(x)))
        x = residual + self.ff_kan2(h)
        return x


# ---------------------------------------------------------------------------
# KAN Language Model (classification)
# ---------------------------------------------------------------------------

class KANLanguageModel(nn.Module):
    """
    Sentiment / classification model using KAN layers throughout.

    Architecture mirrors the baseline Transformer:
      Embedding → positional encoding → N × KANTransformerBlock
      → mean-pool → KANLinear classifier
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        num_classes: int = 2,
        grid_size: int = 5,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.embed_dim  = embed_dim
        self.grid_size  = grid_size

        self.embedding  = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, embed_dim) * 0.02
        )

        self.blocks = nn.ModuleList([
            KANTransformerBlock(embed_dim, num_heads, grid_size)
            for _ in range(num_layers)
        ])

        self.norm       = nn.LayerNorm(embed_dim)
        self.classifier = KANLinear(embed_dim, num_classes, grid_size)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [batch, seq_len] token ids
        Returns:
            logits: [batch, num_classes]
            aux:    {} (empty dict for API compatibility)
        """
        batch, seq_len = x.shape
        h = self.embedding(x) + self.pos_encoding[:, :seq_len, :]

        for block in self.blocks:
            h = block(h)

        h = self.norm(h)
        pooled = h.mean(dim=1)                      # [batch, embed_dim]
        logits = self.classifier(pooled)            # [batch, num_classes]
        return logits, {}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
