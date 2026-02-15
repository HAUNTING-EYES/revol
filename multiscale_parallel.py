"""
Multi-Scale Parallel Processing Network

Inspired by the brain's multi-timescale processing:
  - Fast path: Quick local features via lightweight attention (like neurons)
  - Medium path: Medium-range integration via reaction-diffusion (like cortical columns)
  - Slow path: Global context via slow diffusion (like brain regions)

All three paths run in parallel and are fused with learned weights.

Complexity: O(n^2) for fast path attention, O(n) for medium/slow paths.
At long sequences, the medium/slow paths dominate (much cheaper than pure attention).

Reuses ReactionDiffusionDynamics from wavenet_neuro.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, List
from wavenet_neuro import ReactionDiffusionDynamics, FixedStepEvolution, ContinuousField


class FastPath(nn.Module):
    """
    Fast path: lightweight self-attention for local/global pattern capture.
    Uses standard TransformerEncoder but with fewer layers (speed > depth).
    O(n^2) but with small constant (1-2 layers).
    """

    def __init__(self, dim: int, num_heads: int = 4, num_layers: int = 1, max_seq_len: int = 8192):
        super().__init__()
        self.dim = dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim * 2,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x [batch, seq_len, dim]
        Returns: [batch, dim] (pooled)
        """
        out = self.encoder(x)
        return out.mean(dim=1)


class ReactionDiffusionPath(nn.Module):
    """
    Medium or Slow path: reaction-diffusion dynamics on a 2D field.
    Reuses the proven WaveNetNeuro mechanics.

    - Medium path: 10 steps, diffusion_coeff=0.3 (fast spreading)
    - Slow path: 10 steps, diffusion_coeff=0.05 (slow spreading)
    """

    def __init__(
        self,
        dim: int,
        steps: int = 10,
        dt: float = 0.3,
        diffusion_init: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.steps = steps
        self.dt = dt

        # Projection from embed_dim to field_channels
        self.to_field = nn.Linear(dim, dim)

        # Reaction-diffusion dynamics
        self.dynamics = ReactionDiffusionDynamics(dim)
        # Override diffusion coefficient
        self.dynamics.diffusion_coeff = nn.Parameter(torch.ones(1) * diffusion_init)

        # Fixed-step evolution (no adaptive stopping for predictable timing)
        self.evolution = FixedStepEvolution(self.dynamics, fixed_steps=steps)

        # Pool from 2D field to vector
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x [batch, seq_len, dim]
        Returns: [batch, dim] (pooled from 2D field)
        """
        batch, seq_len, dim = x.shape

        # Project
        projected = self.to_field(x)

        # Reshape to 2D field [batch, dim, H, W]
        height = int(math.sqrt(seq_len))
        width = seq_len // height
        usable = height * width
        field = projected[:, :usable, :].reshape(batch, height, width, dim).permute(0, 3, 1, 2)

        # Run reaction-diffusion
        evolved, _, _ = self.evolution.evolve(field, dt=self.dt)

        # Pool to vector
        return self.pool(evolved)


class MultiScaleNetwork(nn.Module):
    """
    Multi-Scale Parallel Processing Network.

    Three parallel paths process the same input at different timescales:
      1. Fast: 1-layer attention (captures global patterns quickly)
      2. Medium: 10-step reaction-diffusion with fast diffusion
      3. Slow: 10-step reaction-diffusion with slow diffusion

    Outputs are fused with learned weights.

    The key insight: at short sequences (<512), fast path dominates.
    At long sequences (>2048), medium/slow paths are much cheaper than attention.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        num_classes: int = 2,
        max_seq_len: int = 8192,
        fast_layers: int = 1,
        medium_steps: int = 10,
        slow_steps: int = 10,
        medium_diffusion: float = 0.3,
        slow_diffusion: float = 0.05,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Shared embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, embed_dim) * 0.02
        )

        # Three parallel paths
        self.fast_path = FastPath(embed_dim, num_heads=4, num_layers=fast_layers,
                                  max_seq_len=max_seq_len)
        self.medium_path = ReactionDiffusionPath(embed_dim, steps=medium_steps,
                                                  dt=0.3, diffusion_init=medium_diffusion)
        self.slow_path = ReactionDiffusionPath(embed_dim, steps=slow_steps,
                                                dt=0.3, diffusion_init=slow_diffusion)

        # Learnable fusion weights (softmax normalized)
        self.fusion_logits = nn.Parameter(torch.zeros(3))

        # Fuse to classification
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict]:
        batch, seq_len = x.shape

        # Embed
        embeds = self.embedding(x)
        embeds = embeds + self.pos_encoding[:, :seq_len, :]

        # Run all three paths
        fast_out = self.fast_path(embeds)          # [batch, dim]
        medium_out = self.medium_path(embeds)      # [batch, dim]
        slow_out = self.slow_path(embeds)          # [batch, dim]

        # Weighted fusion
        weights = F.softmax(self.fusion_logits, dim=0)
        fused = weights[0] * fast_out + weights[1] * medium_out + weights[2] * slow_out

        # Classify
        output = self.classifier(fused)

        info = {
            'steps_taken': float(self.medium_path.steps + self.slow_path.steps),
            'per_example_steps': torch.full((batch,), float(self.medium_path.steps)),
            'changes_history': [],
            'fusion_weights': weights.detach().cpu().tolist(),
        }

        return output, info

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiScaleNoFast(MultiScaleNetwork):
    """Ablation: no fast (attention) path."""
    def forward(self, x, **kwargs):
        batch, seq_len = x.shape
        embeds = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        medium_out = self.medium_path(embeds)
        slow_out = self.slow_path(embeds)
        fused = 0.5 * medium_out + 0.5 * slow_out
        output = self.classifier(fused)
        info = {'steps_taken': float(self.medium_path.steps + self.slow_path.steps),
                'per_example_steps': torch.full((batch,), float(self.medium_path.steps)),
                'changes_history': []}
        return output, info


class MultiScaleNoMedium(MultiScaleNetwork):
    """Ablation: no medium path."""
    def forward(self, x, **kwargs):
        batch, seq_len = x.shape
        embeds = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        fast_out = self.fast_path(embeds)
        slow_out = self.slow_path(embeds)
        fused = 0.5 * fast_out + 0.5 * slow_out
        output = self.classifier(fused)
        info = {'steps_taken': float(self.slow_path.steps),
                'per_example_steps': torch.full((batch,), float(self.slow_path.steps)),
                'changes_history': []}
        return output, info


class MultiScaleNoSlow(MultiScaleNetwork):
    """Ablation: no slow path."""
    def forward(self, x, **kwargs):
        batch, seq_len = x.shape
        embeds = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        fast_out = self.fast_path(embeds)
        medium_out = self.medium_path(embeds)
        fused = 0.5 * fast_out + 0.5 * medium_out
        output = self.classifier(fused)
        info = {'steps_taken': float(self.medium_path.steps),
                'per_example_steps': torch.full((batch,), float(self.medium_path.steps)),
                'changes_history': []}
        return output, info


def make_multiscale_variant(
    variant: str,
    vocab_size: int,
    embed_dim: int = 64,
    num_classes: int = 2,
    max_seq_len: int = 8192,
) -> nn.Module:
    """Create variant for ablation."""
    kwargs = dict(vocab_size=vocab_size, embed_dim=embed_dim,
                  num_classes=num_classes, max_seq_len=max_seq_len)

    if variant == 'full':
        return MultiScaleNetwork(**kwargs)
    elif variant == 'no_fast':
        return MultiScaleNoFast(**kwargs)
    elif variant == 'no_medium':
        return MultiScaleNoMedium(**kwargs)
    elif variant == 'no_slow':
        return MultiScaleNoSlow(**kwargs)
    elif variant == 'medium_5':
        return MultiScaleNetwork(**kwargs, medium_steps=5, slow_steps=5)
    elif variant == 'medium_20':
        return MultiScaleNetwork(**kwargs, medium_steps=20, slow_steps=20)
    elif variant == 'fast_2_layers':
        return MultiScaleNetwork(**kwargs, fast_layers=2)
    else:
        raise ValueError(f"Unknown variant: {variant}")


if __name__ == "__main__":
    print("MultiScaleNetwork - Quick Test")
    print("=" * 60)

    vocab_size = 10000
    model = MultiScaleNetwork(
        vocab_size=vocab_size, embed_dim=64, num_classes=2,
    )

    for seq_len in [64, 256, 512]:
        x = torch.randint(0, vocab_size, (4, seq_len))
        output, info = model(x)
        print(f"  seq_len={seq_len}: output={output.shape}, "
              f"fusion_weights={[f'{w:.3f}' for w in info.get('fusion_weights', [])]}, "
              f"params={model.count_parameters():,}")
