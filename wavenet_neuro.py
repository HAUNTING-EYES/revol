"""
WaveNetNeuro - Nature-Inspired Neural Architecture

Core Ideas:
1. Continuous field dynamics (not discrete tokens)
2. Local computation O(n) (not global attention O(n^2))
3. Adaptive computation (stops when converged)
4. Reaction-diffusion inspired propagation

Mathematical Foundation:
  dphi/dt = D * laplacian(phi) + F(phi)
  - phi: information field
  - D * laplacian(phi): diffusion (spreading)
  - F(phi): reaction (transformation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List


class ContinuousField(nn.Module):
    """Convert discrete token embeddings to continuous 2D field."""

    def __init__(self, channels: int, spatial_dim: int):
        super().__init__()
        self.channels = channels
        self.spatial_dim = spatial_dim

    def initialize_from_sequence(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [batch, seq_len, embed_dim]
        Returns:
            field: [batch, channels, height, width]
        """
        batch, seq_len, embed_dim = embeddings.shape

        height = int(math.sqrt(seq_len))
        width = seq_len // height

        field = embeddings[:, :height * width, :].reshape(
            batch, height, width, embed_dim
        ).permute(0, 3, 1, 2)

        return field


class ReactionDiffusionDynamics(nn.Module):
    """
    Field dynamics as fixed-point iteration.

    Instead of computing a derivative, computes a TARGET field state.
    The evolution rule is:
      field_new = field + dt * (target(field) - field)

    This naturally converges because as field -> target, the update -> 0.
    Diffusion (local 3x3) spreads information; reaction (1x1) transforms it.
    """

    def __init__(self, channels: int):
        super().__init__()

        # Diffusion: local 3x3 depthwise conv
        self.diffusion = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )

        # Reaction: pointwise nonlinear transform
        self.reaction = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, kernel_size=1),
        )

        self.diffusion_coeff = nn.Parameter(torch.ones(1) * 0.1)

        # Norm to stabilize the target field
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """
        Compute target field state.
        Returns the TARGET (not derivative). Evolution subtracts current field.
        """
        diffusion_term = self.diffusion(field)
        reaction_term = self.reaction(field)

        target = self.diffusion_coeff * diffusion_term + reaction_term
        target = self.norm(target)

        return target


class AdaptiveFieldEvolution(nn.Module):
    """
    Evolves field until convergence with per-example adaptive stopping.

    Uses RELATIVE change: compares each step's change to the initial change,
    so the threshold is scale-invariant. Also computes a ponder cost
    (mean steps used) that can be added to the training loss to incentivize
    early convergence.
    """

    def __init__(self, dynamics: nn.Module):
        super().__init__()
        self.dynamics = dynamics

    def evolve(
        self,
        field: torch.Tensor,
        max_steps: int = 50,
        convergence_threshold: float = 0.01,
        dt: float = 0.1,
        track_per_example: bool = False,
    ) -> Tuple[torch.Tensor, float, Dict]:
        """
        Evolve field until stable or max steps.
        Uses relative convergence: stops when change / initial_change < threshold.

        Returns:
            final_field: converged field state
            avg_steps: average steps across batch
            info: dict with per-example data and ponder_cost
        """
        batch_size = field.shape[0]
        current_field = field
        converged = torch.zeros(batch_size, dtype=torch.bool, device=field.device)
        steps_per_example = torch.full(
            (batch_size,), float(max_steps), dtype=torch.float, device=field.device
        )
        final_fields = field.clone()
        changes_history: List[float] = []
        initial_change: Optional[torch.Tensor] = None

        for step in range(max_steps):
            # Fixed-point iteration: move toward target
            target = self.dynamics(current_field)
            new_field = current_field + dt * (target - current_field)

            per_example_change = (
                torch.abs(new_field - current_field)
                .reshape(batch_size, -1)
                .mean(dim=1)
            )

            # Record initial change for relative comparison
            if step == 0:
                initial_change = per_example_change.clone().clamp(min=1e-8)

            changes_history.append(per_example_change.mean().item())

            # Relative change: how much has the rate of change decreased?
            relative_change = per_example_change / initial_change

            # Mark newly converged examples
            newly_converged = (~converged) & (relative_change < convergence_threshold)
            if newly_converged.any():
                steps_per_example[newly_converged] = step + 1
                final_fields[newly_converged] = new_field[newly_converged]
                converged = converged | newly_converged

            current_field = new_field

            if converged.all():
                break

        if not converged.all():
            not_converged = ~converged
            final_fields[not_converged] = current_field[not_converged]

        avg_steps = steps_per_example.mean().item()
        # Ponder cost: normalized mean steps (0-1 range), can be used as regularizer
        ponder_cost = steps_per_example.mean() / max_steps

        info = {
            'per_example_steps': steps_per_example.detach().cpu(),
            'changes_history': changes_history,
            'ponder_cost': ponder_cost,
        }

        return final_fields, avg_steps, info


class WaveNetNeuro(nn.Module):
    """
    WaveNetNeuro Architecture

    Properties:
    1. O(n) complexity via local convolutions
    2. Adaptive computation - stops per-example when converged
    3. Continuous reaction-diffusion dynamics
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        field_channels: int = 256,
        spatial_dim: int = 16,
        num_classes: int = 2,
        max_evolution_steps: int = 30,
        convergence_threshold: float = 0.1,
        dt: float = 0.3,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.field_channels = field_channels
        self.spatial_dim = spatial_dim
        self.max_evolution_steps = max_evolution_steps
        self.convergence_threshold = convergence_threshold
        self.dt = dt

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, spatial_dim * spatial_dim, embed_dim) * 0.02
        )
        self.to_field = nn.Linear(embed_dim, field_channels)
        self.field = ContinuousField(field_channels, spatial_dim)
        self.dynamics = ReactionDiffusionDynamics(field_channels)
        self.evolution = AdaptiveFieldEvolution(self.dynamics)

        self.from_field = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(field_channels, field_channels // 2),
            nn.GELU(),
            nn.Linear(field_channels // 2, num_classes),
        )

    def forward(
        self, x: torch.Tensor, track_per_example: bool = False
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            x: [batch, seq_len] token indices
            track_per_example: if True, return per-example step counts
        Returns:
            output: [batch, num_classes]
            info: dict with 'steps_taken', 'field_energy', optionally per-example data
        """
        batch_size, seq_len = x.shape

        token_embeds = self.embedding(x)
        pos_embeds = self.pos_encoding[:, :seq_len, :]
        embeddings = token_embeds + pos_embeds

        embeddings = self.to_field(embeddings)
        field = self.field.initialize_from_sequence(embeddings)

        final_field, avg_steps, evo_info = self.evolution.evolve(
            field,
            max_steps=self.max_evolution_steps,
            convergence_threshold=self.convergence_threshold,
            dt=self.dt,
            track_per_example=track_per_example,
        )

        output = self.from_field(final_field)

        info = {
            'steps_taken': avg_steps,
            'field_energy': torch.abs(final_field).mean().item(),
            'per_example_steps': evo_info['per_example_steps'],
            'changes_history': evo_info['changes_history'],
            'ponder_cost': evo_info.get('ponder_cost', None),
        }

        return output, info

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BaselineTransformer(nn.Module):
    """Transformer baseline for comparison (O(n^2) attention)."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        num_classes: int = 2,
        max_seq_len: int = 4096,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, embed_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, dict]:
        batch_size, seq_len = x.shape

        embeds = self.embedding(x)
        embeds = embeds + self.pos_encoding[:, :seq_len, :]

        transformed = self.transformer(embeds)

        pooled = transformed.mean(dim=1)
        output = self.classifier(pooled)

        return output, {
            'steps_taken': self.num_layers,
            'per_example_steps': torch.full((batch_size,), float(self.num_layers)),
            'changes_history': [],
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DiffusionOnlyDynamics(nn.Module):
    """Ablation: only diffusion, no reaction term."""

    def __init__(self, channels: int):
        super().__init__()
        self.diffusion = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )
        self.diffusion_coeff = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        return self.diffusion_coeff * self.diffusion(field)


class ReactionOnlyDynamics(nn.Module):
    """Ablation: only reaction, no diffusion term."""

    def __init__(self, channels: int):
        super().__init__()
        self.reaction = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, kernel_size=1),
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        return self.reaction(field)


class FixedStepEvolution(nn.Module):
    """Ablation: fixed number of steps, no adaptive stopping."""

    def __init__(self, dynamics: nn.Module, fixed_steps: int = 10):
        super().__init__()
        self.dynamics = dynamics
        self.fixed_steps = fixed_steps

    def evolve(
        self,
        field: torch.Tensor,
        max_steps: int = 50,
        convergence_threshold: float = 0.01,
        dt: float = 0.1,
        track_per_example: bool = False,
    ) -> Tuple[torch.Tensor, float, Dict]:
        batch_size = field.shape[0]
        current_field = field
        changes_history: List[float] = []

        for step in range(self.fixed_steps):
            target = self.dynamics(current_field)
            new_field = current_field + dt * (target - current_field)
            change = torch.abs(new_field - current_field).reshape(batch_size, -1).mean(1)
            changes_history.append(change.mean().item())
            current_field = new_field

        info = {
            'per_example_steps': torch.full((batch_size,), float(self.fixed_steps)),
            'changes_history': changes_history,
        }
        return current_field, float(self.fixed_steps), info


def make_wavenet_variant(
    variant: str,
    vocab_size: int,
    embed_dim: int = 128,
    field_channels: int = 128,
    num_classes: int = 2,
    max_evolution_steps: int = 30,
    fixed_steps: int = 10,
) -> WaveNetNeuro:
    """
    Create WaveNetNeuro variant for ablation study.

    Variants:
      'full'            - full model (diffusion + reaction + adaptive)
      'diffusion_only'  - only diffusion term
      'reaction_only'   - only reaction term
      'fixed_steps'     - adaptive stopping disabled, fixed N steps
    """
    model = WaveNetNeuro(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        field_channels=field_channels,
        num_classes=num_classes,
        max_evolution_steps=max_evolution_steps,
    )

    if variant == 'diffusion_only':
        model.dynamics = DiffusionOnlyDynamics(field_channels)
        model.evolution = AdaptiveFieldEvolution(model.dynamics)
    elif variant == 'reaction_only':
        model.dynamics = ReactionOnlyDynamics(field_channels)
        model.evolution = AdaptiveFieldEvolution(model.dynamics)
    elif variant == 'fixed_steps':
        model.evolution = FixedStepEvolution(model.dynamics, fixed_steps=fixed_steps)

    return model


if __name__ == "__main__":
    print("WaveNetNeuro - Quick Test")
    print("=" * 60)

    vocab_size = 10000
    wavenet = WaveNetNeuro(
        vocab_size=vocab_size, embed_dim=256, field_channels=256, num_classes=2
    )
    transformer = BaselineTransformer(
        vocab_size=vocab_size, embed_dim=256, num_classes=2
    )

    batch_size = 4
    seq_len = 64
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    print(f"\nModel Statistics:")
    print(f"  WaveNetNeuro parameters: {wavenet.count_parameters():,}")
    print(f"  Transformer parameters:  {transformer.count_parameters():,}")

    output, info = wavenet(x)
    print(f"\nWaveNetNeuro:")
    print(f"  Output shape: {output.shape}")
    print(f"  Avg adaptive steps: {info['steps_taken']:.1f}")
    print(f"  Per-example steps:  {info['per_example_steps'].tolist()}")
    print(f"  Field energy: {info['field_energy']:.4f}")

    output_t, info_t = transformer(x)
    print(f"\nTransformer:")
    print(f"  Output shape: {output_t.shape}")
    print(f"  Fixed steps: {info_t['steps_taken']}")
