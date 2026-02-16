"""
Hierarchical Predictive Coding Network

Unlike the pooled PredCoding (which loses positional info by mean-pooling early),
this operates on FULL SEQUENCES with multi-scale hierarchy:

  Layer 1: [batch, seq_len, dim]      Full resolution
  Layer 2: [batch, seq_len/2, dim]    Downsample 2x
  Layer 3: [batch, seq_len/4, dim]    Downsample 4x
  Layer 4: [batch, seq_len/8, dim]    Downsample 8x

Bidirectional processing:
  - Top-down: each layer predicts the layer below (via ConvTranspose1d upsample)
  - Bottom-up: prediction errors refine representations (via Conv1d stride downsample)
  - Pool ONLY at the very end

Complexity: O(n) per iteration (conv1d), O(n * iterations) total.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional


class HierarchicalPCLayer(nn.Module):
    """
    Single layer in the hierarchical predictive coding network.

    Each layer operates at a specific spatial resolution.
    - feedforward: initialize this layer from the layer below (with downsampling)
    - predict_down: predict what the layer below should look like (with upsampling)
    - process_error: refine representation based on prediction error
    """

    def __init__(self, dim: int, is_bottom: bool = False):
        super().__init__()
        self.dim = dim
        self.is_bottom = is_bottom
        hidden = dim * 2

        # Feedforward init: downsample from layer below (stride 2 conv)
        if not is_bottom:
            self.downsample = nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv1d(dim, dim, kernel_size=1),
            )
        else:
            # Bottom layer: just a local transform
            self.downsample = nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(dim, dim, kernel_size=1),
            )
        self.ff_norm = nn.LayerNorm(dim)

        # Top-down prediction: upsample to predict layer below
        if not is_bottom:
            self.predict_up = nn.Sequential(
                nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                nn.Conv1d(dim, dim, kernel_size=1),
            )

        # Error processing: takes error signal, produces update
        self.process_error = nn.Sequential(
            nn.Conv1d(dim, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, dim, kernel_size=1),
        )
        self.update_norm = nn.LayerNorm(dim)

        # Learnable step size
        self.step_size = nn.Parameter(torch.tensor(0.3))

    def init_representation(self, input_below):
        """
        Feedforward init with residual.
        input_below: [batch, dim, seq_len] (conv format)
        Returns: [batch, dim, seq_len/2] for non-bottom, [batch, dim, seq_len] for bottom
        """
        out = self.downsample(input_below)
        # Residual only if dimensions match
        if out.shape == input_below.shape:
            out = out + input_below
        # LayerNorm expects [batch, seq_len, dim]
        out = self.ff_norm(out.transpose(1, 2)).transpose(1, 2)
        return out

    def predict(self, representation):
        """
        Top-down: predict what the layer below should be.
        representation: [batch, dim, seq_len_this]
        Returns: [batch, dim, seq_len_below] (upsampled)
        """
        if self.is_bottom:
            return representation  # Bottom layer predicts itself
        return self.predict_up(representation)

    def update(self, representation, error):
        """
        Update representation based on error signal.
        representation: [batch, dim, seq_len]
        error: [batch, dim, seq_len] (same size, downsampled if needed)
        """
        correction = self.process_error(error)
        updated = representation + self.step_size * correction
        return self.update_norm(updated.transpose(1, 2)).transpose(1, 2)


class HierarchicalPredCodingNetwork(nn.Module):
    """
    Multi-scale hierarchical predictive coding network.

    Architecture:
      Input [batch, seq_len, dim]
        -> Layer 1 [batch, dim, seq_len]      (full resolution)
        -> Layer 2 [batch, dim, seq_len/2]    (downsample 2x)
        -> Layer 3 [batch, dim, seq_len/4]    (downsample 4x)
        -> Layer 4 [batch, dim, seq_len/8]    (downsample 8x)
        -> Pool -> Classify

    Iterative refinement:
      1. Bottom-up errors: each layer predicts the one below, errors propagate up
      2. Top-down updates: representations refined to minimize prediction errors
      3. Converge or reach max iterations
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        num_layers: int = 4,
        num_classes: int = 2,
        max_iterations: int = 10,
        convergence_threshold: float = 0.1,
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

        # Project to conv dimension
        self.input_proj = nn.Linear(embed_dim, embed_dim)

        # Hierarchical PC layers
        self.layers = nn.ModuleList()
        self.layers.append(HierarchicalPCLayer(embed_dim, is_bottom=True))
        for _ in range(1, num_layers):
            self.layers.append(HierarchicalPCLayer(embed_dim, is_bottom=False))

        # Classifier from top layer
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def _downsample_error(self, error, target_len):
        """Downsample an error signal to match a higher layer's resolution."""
        if error.shape[2] == target_len:
            return error
        return F.adaptive_avg_pool1d(error, target_len)

    def _match_len(self, pred, target_len):
        """Trim or pad prediction to match target length."""
        if pred.shape[2] == target_len:
            return pred
        if pred.shape[2] > target_len:
            return pred[:, :, :target_len]
        # Pad
        pad_size = target_len - pred.shape[2]
        return F.pad(pred, (0, pad_size))

    def inference(self, x_conv: torch.Tensor) -> Tuple[torch.Tensor, float, Dict]:
        """
        Iterative hierarchical inference.

        Args:
            x_conv: [batch, dim, seq_len] input in conv format
        Returns:
            output: [batch, dim] pooled top-layer representation
            avg_steps: average iterations
            info: dict with convergence data
        """
        batch_size = x_conv.shape[0]
        device = x_conv.device

        # Phase 1: Feedforward initialization
        reps = []
        current = x_conv
        for layer in self.layers:
            current = layer.init_representation(current)
            reps.append(current)

        # Save feedforward top-layer output for skip connection
        ff_pooled = reps[-1].mean(dim=2)  # [batch, dim]

        # Phase 2: Iterative PC refinement
        if self.max_iterations == 0:
            steps_per_example = torch.zeros(batch_size, device=device)
            return ff_pooled, 0.0, {
                'per_example_steps': steps_per_example.cpu(),
                'changes_history': [],
                'ponder_cost': torch.tensor(0.0),
            }

        converged = torch.zeros(batch_size, dtype=torch.bool, device=device)
        steps_per_example = torch.full(
            (batch_size,), float(self.max_iterations), dtype=torch.float, device=device
        )
        final_top_pooled = torch.zeros(batch_size, self.embed_dim, device=device)
        changes_history: List[float] = []
        initial_change: Optional[torch.Tensor] = None

        for iteration in range(self.max_iterations):
            old_reps_detached = [r.detach() for r in reps]

            # BOTTOM-UP: compute prediction errors
            errors = []

            # Layer 0 error: input vs what layer 0 predicts
            pred_0 = self.layers[0].predict(reps[0])
            pred_0 = self._match_len(pred_0, x_conv.shape[2])
            err_0 = x_conv - pred_0
            # Downsample error to layer 0's resolution
            err_0_ds = self._downsample_error(err_0, reps[0].shape[2])
            errors.append(err_0_ds)

            # Higher layer errors: each layer predicts the layer below
            for i in range(1, self.num_layers):
                pred_i = self.layers[i].predict(reps[i])
                pred_i = self._match_len(pred_i, reps[i - 1].shape[2])
                err_i = reps[i - 1].detach() - pred_i
                # Downsample error to this layer's resolution
                err_i_ds = self._downsample_error(err_i, reps[i].shape[2])
                errors.append(err_i_ds)

            # TOP-DOWN + BOTTOM-UP: update representations
            new_reps = []
            for i in range(self.num_layers):
                error_below = errors[i]

                if self.bidirectional and i < self.num_layers - 1:
                    # Error from layer above predicting us
                    pred_from_above = self.layers[i + 1].predict(reps[i + 1])
                    pred_from_above = self._match_len(pred_from_above, reps[i].shape[2])
                    error_above = reps[i] - pred_from_above
                else:
                    error_above = torch.zeros_like(reps[i])

                total_error = error_below + 0.5 * error_above
                new_reps.append(self.layers[i].update(reps[i], total_error))

            reps = new_reps

            # Convergence check
            with torch.no_grad():
                # Use top-layer change as convergence metric
                top_change = (reps[-1] - old_reps_detached[-1]).abs()
                per_example_change = top_change.reshape(batch_size, -1).mean(dim=1)

                if iteration == 0:
                    initial_change = per_example_change.clamp(min=1e-8)

                changes_history.append(per_example_change.mean().item())
                relative_change = per_example_change / initial_change

                newly_converged = (~converged) & (relative_change < self.convergence_threshold)
                if newly_converged.any():
                    steps_per_example[newly_converged] = iteration + 1
                    final_top_pooled[newly_converged] = reps[-1].mean(dim=2)[newly_converged]
                    converged = converged | newly_converged

                if converged.all():
                    break

        # Fill non-converged
        not_converged = ~converged
        if not_converged.any():
            final_top_pooled[not_converged] = reps[-1].mean(dim=2)[not_converged]

        # Skip connection from feedforward
        output = final_top_pooled + ff_pooled

        avg_steps = steps_per_example.mean().item()
        ponder_cost = steps_per_example.mean() / max(self.max_iterations, 1)

        return output, avg_steps, {
            'per_example_steps': steps_per_example.detach().cpu(),
            'changes_history': changes_history,
            'ponder_cost': ponder_cost,
        }

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict]:
        batch, seq_len = x.shape

        # Ensure seq_len is divisible by 2^(num_layers-1) for clean downsampling
        divisor = 2 ** (self.num_layers - 1)
        padded_len = ((seq_len + divisor - 1) // divisor) * divisor
        if padded_len != seq_len:
            x = F.pad(x, (0, padded_len - seq_len), value=0)

        embeds = self.embedding(x)
        embeds = embeds + self.pos_encoding[:, :padded_len, :]
        projected = self.input_proj(embeds)

        # Convert to conv format: [batch, dim, seq_len]
        x_conv = projected.transpose(1, 2)

        final_rep, avg_steps, evo_info = self.inference(x_conv)

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


def make_hier_pc_variant(
    variant: str,
    vocab_size: int,
    embed_dim: int = 64,
    num_layers: int = 4,
    num_classes: int = 2,
    max_iterations: int = 10,
    convergence_threshold: float = 0.1,
    max_seq_len: int = 8192,
) -> HierarchicalPredCodingNetwork:
    """Create variant for ablation."""
    kwargs = dict(
        vocab_size=vocab_size, embed_dim=embed_dim, num_layers=num_layers,
        num_classes=num_classes, max_iterations=max_iterations,
        convergence_threshold=convergence_threshold, max_seq_len=max_seq_len,
    )
    if variant == 'full':
        return HierarchicalPredCodingNetwork(**kwargs, bidirectional=True)
    elif variant == 'bottom_up_only':
        return HierarchicalPredCodingNetwork(**kwargs, bidirectional=False)
    elif variant.startswith('iterations_'):
        n = int(variant.split('_')[1])
        kwargs['max_iterations'] = n
        if n > 0:
            kwargs['convergence_threshold'] = 1e-10
        return HierarchicalPredCodingNetwork(**kwargs, bidirectional=True)
    elif variant.startswith('depth_'):
        d = int(variant.split('_')[1])
        kwargs['num_layers'] = d
        return HierarchicalPredCodingNetwork(**kwargs, bidirectional=True)
    else:
        raise ValueError(f"Unknown variant: {variant}")


if __name__ == "__main__":
    print("HierarchicalPredCodingNetwork - Quick Test")
    print("=" * 60)

    vocab_size = 10000
    model = HierarchicalPredCodingNetwork(
        vocab_size=vocab_size, embed_dim=64, num_layers=4,
        num_classes=2, max_iterations=10, convergence_threshold=0.1,
    )

    for seq_len in [64, 256, 512]:
        x = torch.randint(0, vocab_size, (4, seq_len))
        output, info = model(x)
        print(f"  seq_len={seq_len}: output={output.shape}, "
              f"steps={info['steps_taken']:.1f}, "
              f"params={model.count_parameters():,}")
