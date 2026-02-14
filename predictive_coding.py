"""
Predictive Coding Architecture

Key differences from WaveNetNeuro:
- Bidirectional: predictions down, errors up
- Error-based: only errors propagate (sparse)
- Hierarchical: each layer predicts layer below
- Iterative: refines until errors small

vs WaveNetNeuro:
- WaveNetNeuro: lateral diffusion (sideways spreading via 2D convolutions)
- PredCoding: vertical prediction (up/down hierarchy via linear layers)

Complexity: O(n) for embedding + O(iterations * embed_dim^2) for inference
The PC iterations operate on pooled representations, not full sequences.

Architecture:
1. Embedding + mean pooling -> fixed-size vector
2. Feedforward initialization (residual) -> layered representations
3. Iterative PC refinement (error-driven bidirectional updates)
4. Classification from top-layer representation + skip connection
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, List, Optional


class PredictiveCodingLayer(nn.Module):
    """
    Single predictive coding layer.

    Each layer:
    - Has a feedforward path (for initialization, with residual)
    - Can predict the layer below (top-down)
    - Processes prediction errors (bottom-up)
    - Updates its representation via gated error correction
    """

    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        self.dim = dim
        hidden_dim = hidden_dim or dim * 2

        # Feedforward: bottom-up transform with residual connection
        self.ff_transform = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.ff_norm = nn.LayerNorm(dim)

        # Top-down: predict what layer below should be
        self.predict_down = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

        # Bottom-up: process prediction error into update signal
        self.process_error = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.update_norm = nn.LayerNorm(dim)

        # Learnable step size for error-driven updates
        self.step_size = nn.Parameter(torch.tensor(0.3))

    def init_representation(self, input_below):
        """Feedforward initialization with residual connection."""
        return self.ff_norm(input_below + self.ff_transform(input_below))

    def predict(self, representation):
        """Predict what input should be (top-down prediction)."""
        return self.predict_down(representation)

    def compute_error(self, actual, prediction):
        """Compute prediction error."""
        return actual - prediction

    def update(self, representation, error):
        """Update representation via error-driven correction with residual."""
        correction = self.process_error(error)
        return self.update_norm(representation + self.step_size * correction)


class PredictiveCodingNetwork(nn.Module):
    """
    Multi-layer predictive coding network.

    Architecture:
    Input -> [FF Init] -> Layer1 <-> Layer2 <-> Layer3 <-> Layer4 -> Output
                           errors^    errors^    errors^
                           predict v  predict v  predict v

    Key design decisions:
    - Residual feedforward init: gradient highway through all layers
    - Skip connection: ff_output added to PC output for stable training
    - Detached convergence check: prevents compute graph explosion
    - Per-example adaptive stopping: each example converges independently
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_classes: int = 2,
        max_iterations: int = 20,
        convergence_threshold: float = 0.01,
        max_seq_len: int = 8192,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.bidirectional = bidirectional

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, embed_dim) * 0.02
        )

        # Predictive coding layers
        self.layers = nn.ModuleList([
            PredictiveCodingLayer(embed_dim)
            for _ in range(num_layers)
        ])

        # Classifier (takes both skip and refined representation)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def inference(self, x: torch.Tensor) -> Tuple[torch.Tensor, float, Dict]:
        """
        Iterative inference with bidirectional passes.

        Phase 1: Fast feedforward sweep (residual) for initialization
        Phase 2: Iterative PC refinement with error-driven updates

        Args:
            x: pooled input embeddings [batch, embed_dim]
        Returns:
            final_rep: top layer representation [batch, embed_dim]
            avg_steps: average iterations across batch
            info: dict with per-example data and ponder_cost
        """
        batch_size = x.size(0)
        device = x.device

        # Phase 1: Feedforward initialization with residual connections
        reps = []
        current = x
        for layer in self.layers:
            current = layer.init_representation(current)
            reps.append(current)

        # Save feedforward output for skip connection
        ff_output = reps[-1]

        # Phase 2: Iterative PC refinement
        if self.max_iterations == 0:
            # Pure feedforward mode (no iterations)
            steps_per_example = torch.zeros(batch_size, device=device)
            return ff_output, 0.0, {
                'per_example_steps': steps_per_example.cpu(),
                'changes_history': [],
                'ponder_cost': torch.tensor(0.0),
            }

        converged = torch.zeros(batch_size, dtype=torch.bool, device=device)
        steps_per_example = torch.full(
            (batch_size,), float(self.max_iterations), dtype=torch.float, device=device
        )
        final_reps = [torch.zeros_like(r) for r in reps]
        changes_history: List[float] = []
        initial_change: Optional[torch.Tensor] = None

        for iteration in range(self.max_iterations):
            # Detach old reps for convergence check (not for computation)
            old_reps_detached = [r.detach() for r in reps]

            # BOTTOM-UP: Compute prediction errors
            errors = []

            # Layer 0: compare input to what layer 0 predicts the input is
            pred_0 = self.layers[0].predict(reps[0])
            err_0 = self.layers[0].compute_error(x, pred_0)
            errors.append(err_0)

            # Higher layers: each layer predicts the layer below
            for i in range(1, self.num_layers):
                pred_i = self.layers[i].predict(reps[i])
                err_i = self.layers[i].compute_error(reps[i - 1].detach(), pred_i)
                errors.append(err_i)

            # TOP-DOWN + BOTTOM-UP: Update representations
            new_reps = []
            for i in range(self.num_layers):
                error_below = errors[i]

                if self.bidirectional and i < self.num_layers - 1:
                    pred_from_above = self.layers[i + 1].predict(reps[i + 1])
                    error_above = reps[i] - pred_from_above
                else:
                    error_above = torch.zeros_like(reps[i])

                total_error = error_below + 0.5 * error_above
                new_reps.append(self.layers[i].update(reps[i], total_error))

            reps = new_reps

            # Per-example convergence check (detached to save memory)
            with torch.no_grad():
                per_example_change = torch.stack([
                    (new - old).abs().mean(dim=-1)
                    for new, old in zip(reps, old_reps_detached)
                ], dim=0).max(dim=0).values

                if iteration == 0:
                    initial_change = per_example_change.clamp(min=1e-8)

                changes_history.append(per_example_change.mean().item())
                relative_change = per_example_change / initial_change

                newly_converged = (~converged) & (relative_change < self.convergence_threshold)
                if newly_converged.any():
                    steps_per_example[newly_converged] = iteration + 1
                    for li in range(self.num_layers):
                        final_reps[li][newly_converged] = reps[li][newly_converged]
                    converged = converged | newly_converged

                if converged.all():
                    break

        # Fill in non-converged examples
        not_converged = ~converged
        if not_converged.any():
            for li in range(self.num_layers):
                final_reps[li][not_converged] = reps[li][not_converged]

        # Combine refined output with skip connection from feedforward
        output_rep = final_reps[-1] + ff_output

        avg_steps = steps_per_example.mean().item()
        ponder_cost = steps_per_example.mean() / max(self.max_iterations, 1)

        info = {
            'per_example_steps': steps_per_example.detach().cpu(),
            'changes_history': changes_history,
            'ponder_cost': ponder_cost,
        }

        return output_rep, avg_steps, info

    def forward(
        self, x: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: token indices [batch, seq_len]
        Returns:
            output: predictions [batch, num_classes]
            info: dict with steps_taken, per_example_steps, etc.
        """
        batch, seq_len = x.shape

        # Embed and pool to fixed-size representation
        embeds = self.embedding(x)
        embeds = embeds + self.pos_encoding[:, :seq_len, :]
        pooled = embeds.mean(dim=1)  # [batch, embed_dim]

        # Iterative inference
        final_rep, avg_steps, evo_info = self.inference(pooled)

        # Classify
        output = self.classifier(final_rep)

        info = {
            'steps_taken': avg_steps,
            'per_example_steps': evo_info['per_example_steps'],
            'changes_history': evo_info['changes_history'],
            'ponder_cost': evo_info.get('ponder_cost', None),
        }

        return output, info

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def make_pc_variant(
    variant: str,
    vocab_size: int,
    embed_dim: int = 64,
    num_layers: int = 4,
    num_classes: int = 2,
    max_iterations: int = 20,
    convergence_threshold: float = 0.01,
    max_seq_len: int = 8192,
) -> PredictiveCodingNetwork:
    """
    Create a PredictiveCodingNetwork variant for ablation.

    Variants:
      'full'           - bidirectional (errors up + predictions down)
      'bottom_up_only' - no top-down predictions
      'iterations_5'   - fixed 5 iterations
      'iterations_10'  - fixed 10 iterations
      'iterations_0'   - pure feedforward (no PC iterations)
      'depth_2'        - 2 layers
      'depth_6'        - 6 layers
      'depth_8'        - 8 layers
    """
    kwargs = dict(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        max_seq_len=max_seq_len,
    )

    if variant == 'full':
        return PredictiveCodingNetwork(**kwargs, bidirectional=True)
    elif variant == 'bottom_up_only':
        return PredictiveCodingNetwork(**kwargs, bidirectional=False)
    elif variant.startswith('iterations_'):
        n = int(variant.split('_')[1])
        kwargs['max_iterations'] = n
        if n > 0:
            kwargs['convergence_threshold'] = 1e-10  # never converge early
        return PredictiveCodingNetwork(**kwargs, bidirectional=True)
    elif variant.startswith('depth_'):
        depth = int(variant.split('_')[1])
        kwargs['num_layers'] = depth
        return PredictiveCodingNetwork(**kwargs, bidirectional=True)
    else:
        raise ValueError(f"Unknown variant: {variant}")


if __name__ == "__main__":
    print("PredictiveCodingNetwork - Quick Test")
    print("=" * 60)

    vocab_size = 10000
    model = PredictiveCodingNetwork(
        vocab_size=vocab_size, embed_dim=64, num_layers=4,
        num_classes=2, max_iterations=20, convergence_threshold=0.01,
    )

    batch_size = 4
    seq_len = 64
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    print(f"\nParameters: {model.count_parameters():,}")

    output, info = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Avg iterations: {info['steps_taken']:.1f}")
    print(f"Per-example steps: {info['per_example_steps'].tolist()}")
    print(f"Convergence history: {[f'{c:.4f}' for c in info['changes_history'][:5]]}...")
