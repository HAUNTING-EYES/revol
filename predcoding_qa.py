"""
Predictive Coding for Question Answering

Tests whether PredCoding (which pools sequences early via mean pooling)
can handle tasks requiring positional understanding:

1. Multiple Choice (RACE): Classify which of 4 answers is correct.
   - Still fundamentally classification -> pooling should partially work.
   - Encode: [passage | SEP | question | SEP | answer_i] for each candidate.

2. Span Extraction (SQuAD 2.0): Predict start/end token positions.
   - Requires positional info that mean pooling destroys.
   - Two approaches:
     a) PC-Pointer: Use refined pooled vector to attend back to token embeddings
        and predict start/end positions.
     b) Baseline: Just predict start/end from token embeddings directly (no PC).

Key hypothesis: PredCoding excels at "vibes" (overall sentiment/topic) but
struggles with positional extraction. Multiple Choice should partially work;
Span Extraction will likely fail.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
from predictive_coding import PredictiveCodingLayer


# ============================================================================
# Multiple Choice QA (for RACE)
# ============================================================================

class PredCodingMultipleChoice(nn.Module):
    """
    PredCoding for multiple choice QA.

    For each answer option, encode [passage SEP question SEP answer] as a
    single sequence, pool, run PC iterations, and score.
    Final answer = argmax over 4 scores.

    This is essentially 4x classification, so PredCoding's pooling approach
    should work reasonably well.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        num_layers: int = 4,
        num_choices: int = 4,
        max_iterations: int = 10,
        convergence_threshold: float = 0.1,
        max_seq_len: int = 512,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_choices = num_choices
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.bidirectional = bidirectional

        # Shared embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, embed_dim) * 0.02
        )

        # PC layers (shared across all answer options)
        self.layers = nn.ModuleList([
            PredictiveCodingLayer(embed_dim) for _ in range(num_layers)
        ])

        # Score head: pooled representation -> scalar score
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def encode_and_infer(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a single sequence and run PC inference.
        Args: x [batch, seq_len] token IDs
        Returns: [batch, embed_dim] refined representation
        """
        batch, seq_len = x.shape
        device = x.device

        embeds = self.embedding(x)
        embeds = embeds + self.pos_encoding[:, :seq_len, :]
        pooled = embeds.mean(dim=1)  # [batch, embed_dim]

        # Feedforward init
        reps = []
        current = pooled
        for layer in self.layers:
            current = layer.init_representation(current)
            reps.append(current)
        ff_output = reps[-1]

        if self.max_iterations == 0:
            return ff_output

        # Iterative PC refinement
        for iteration in range(self.max_iterations):
            errors = []
            pred_0 = self.layers[0].predict(reps[0])
            err_0 = pooled - pred_0
            errors.append(err_0)

            for i in range(1, self.num_layers):
                pred_i = self.layers[i].predict(reps[i])
                err_i = reps[i - 1].detach() - pred_i
                errors.append(err_i)

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

        return reps[-1] + ff_output

    def forward(self, input_ids: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            input_ids: [batch, num_choices, seq_len] token IDs for each option
        Returns:
            logits: [batch, num_choices] scores
            info: dict
        """
        batch, num_choices, seq_len = input_ids.shape

        # Process each choice
        all_scores = []
        for c in range(num_choices):
            rep = self.encode_and_infer(input_ids[:, c, :])  # [batch, embed_dim]
            score = self.scorer(rep)  # [batch, 1]
            all_scores.append(score)

        logits = torch.cat(all_scores, dim=1)  # [batch, num_choices]

        info = {
            'steps_taken': float(self.max_iterations),
            'per_example_steps': torch.full((batch,), float(self.max_iterations)),
            'changes_history': [],
        }
        return logits, info

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Span Extraction QA (for SQuAD)
# ============================================================================

class PredCodingSpanExtraction(nn.Module):
    """
    PredCoding for span extraction QA.

    The core challenge: PredCoding pools the sequence to a fixed-size vector,
    losing positional information. To predict span start/end, we need to
    "project back" to token-level predictions.

    Approach (PC-Pointer):
      1. Embed [context SEP question] -> token embeddings [batch, seq_len, dim]
      2. Pool -> [batch, dim], run PC iterations -> refined vector
      3. Use refined vector as a query to attend back to token embeddings
      4. Predict start/end logits via cross-attention

    This tests whether PC iterations can refine a "search query" that
    points to the right span in the original sequence.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        num_layers: int = 4,
        max_iterations: int = 10,
        convergence_threshold: float = 0.1,
        max_seq_len: int = 512,
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

        # PC layers
        self.layers = nn.ModuleList([
            PredictiveCodingLayer(embed_dim) for _ in range(num_layers)
        ])

        # Start/end query projections (from refined pooled vector)
        self.start_query = nn.Linear(embed_dim, embed_dim)
        self.end_query = nn.Linear(embed_dim, embed_dim)

        # Key projection for token embeddings
        self.token_key = nn.Linear(embed_dim, embed_dim)

        # Answerable classifier (for SQuAD 2.0 unanswerable questions)
        self.answerable_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 2),  # answerable vs unanswerable
        )

    def forward(self, input_ids: torch.Tensor, **kwargs) -> Tuple[Dict, Dict]:
        """
        Args:
            input_ids: [batch, seq_len] token IDs for [context SEP question]
        Returns:
            outputs: dict with 'start_logits', 'end_logits', 'answerable_logits'
            info: dict
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device

        # Step 1: Embed
        embeds = self.embedding(input_ids)
        embeds = embeds + self.pos_encoding[:, :seq_len, :]

        # Step 2: Pool and run PC
        pooled = embeds.mean(dim=1)  # [batch, embed_dim]

        reps = []
        current = pooled
        for layer in self.layers:
            current = layer.init_representation(current)
            reps.append(current)
        ff_output = reps[-1]

        if self.max_iterations > 0:
            for iteration in range(self.max_iterations):
                errors = []
                pred_0 = self.layers[0].predict(reps[0])
                err_0 = pooled - pred_0
                errors.append(err_0)

                for i in range(1, self.num_layers):
                    pred_i = self.layers[i].predict(reps[i])
                    err_i = reps[i - 1].detach() - pred_i
                    errors.append(err_i)

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

        refined = reps[-1] + ff_output  # [batch, embed_dim]

        # Step 3: Project back to token-level via cross-attention
        token_keys = self.token_key(embeds)  # [batch, seq_len, dim]

        start_q = self.start_query(refined)  # [batch, dim]
        end_q = self.end_query(refined)      # [batch, dim]

        # Dot product attention: query [batch, dim] x keys [batch, seq_len, dim]
        start_logits = torch.einsum('bd,bsd->bs', start_q, token_keys)  # [batch, seq_len]
        end_logits = torch.einsum('bd,bsd->bs', end_q, token_keys)      # [batch, seq_len]

        # Answerable prediction
        answerable_logits = self.answerable_head(refined)  # [batch, 2]

        outputs = {
            'start_logits': start_logits,
            'end_logits': end_logits,
            'answerable_logits': answerable_logits,
        }

        info = {
            'steps_taken': float(self.max_iterations),
            'per_example_steps': torch.full((batch,), float(self.max_iterations)),
            'changes_history': [],
        }
        return outputs, info

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Transformer Baselines for QA
# ============================================================================

class TransformerMultipleChoice(nn.Module):
    """Transformer baseline for multiple choice QA."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        num_choices: int = 4,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_choices = num_choices

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, embed_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 2,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, input_ids: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict]:
        """
        Args: input_ids [batch, num_choices, seq_len]
        Returns: logits [batch, num_choices], info dict
        """
        batch, num_choices, seq_len = input_ids.shape
        all_scores = []

        for c in range(num_choices):
            x = input_ids[:, c, :]
            embeds = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
            encoded = self.encoder(embeds)
            pooled = encoded.mean(dim=1)
            score = self.scorer(pooled)
            all_scores.append(score)

        logits = torch.cat(all_scores, dim=1)
        info = {
            'steps_taken': 0.0,
            'per_example_steps': torch.zeros(batch),
            'changes_history': [],
        }
        return logits, info

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransformerSpanExtraction(nn.Module):
    """Transformer baseline for span extraction QA."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, embed_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 2,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Direct token-level start/end predictions
        self.start_head = nn.Linear(embed_dim, 1)
        self.end_head = nn.Linear(embed_dim, 1)

        # Answerable head from CLS-like pooling
        self.answerable_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 2),
        )

    def forward(self, input_ids: torch.Tensor, **kwargs) -> Tuple[Dict, Dict]:
        batch, seq_len = input_ids.shape

        embeds = self.embedding(input_ids) + self.pos_encoding[:, :seq_len, :]
        encoded = self.encoder(embeds)  # [batch, seq_len, dim]

        start_logits = self.start_head(encoded).squeeze(-1)  # [batch, seq_len]
        end_logits = self.end_head(encoded).squeeze(-1)       # [batch, seq_len]

        pooled = encoded.mean(dim=1)
        answerable_logits = self.answerable_head(pooled)

        outputs = {
            'start_logits': start_logits,
            'end_logits': end_logits,
            'answerable_logits': answerable_logits,
        }
        info = {
            'steps_taken': 0.0,
            'per_example_steps': torch.zeros(batch),
            'changes_history': [],
        }
        return outputs, info

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Quick test
# ============================================================================

if __name__ == "__main__":
    print("PredCoding QA Models - Quick Test")
    print("=" * 60)

    vocab_size = 10000
    batch = 4
    seq_len = 128
    num_choices = 4

    # Multiple Choice
    print("\n--- Multiple Choice ---")
    mc_pc = PredCodingMultipleChoice(vocab_size=vocab_size, embed_dim=64, num_layers=4)
    mc_tf = TransformerMultipleChoice(vocab_size=vocab_size, embed_dim=64)

    x_mc = torch.randint(0, vocab_size, (batch, num_choices, seq_len))
    out_pc, _ = mc_pc(x_mc)
    out_tf, _ = mc_tf(x_mc)
    print(f"  PredCoding MC: output={out_pc.shape}, params={mc_pc.count_parameters():,}")
    print(f"  Transformer MC: output={out_tf.shape}, params={mc_tf.count_parameters():,}")

    # Span Extraction
    print("\n--- Span Extraction ---")
    span_pc = PredCodingSpanExtraction(vocab_size=vocab_size, embed_dim=64, num_layers=4)
    span_tf = TransformerSpanExtraction(vocab_size=vocab_size, embed_dim=64)

    x_span = torch.randint(0, vocab_size, (batch, seq_len))
    out_pc_s, _ = span_pc(x_span)
    out_tf_s, _ = span_tf(x_span)
    print(f"  PredCoding Span: start={out_pc_s['start_logits'].shape}, "
          f"end={out_pc_s['end_logits'].shape}, params={span_pc.count_parameters():,}")
    print(f"  Transformer Span: start={out_tf_s['start_logits'].shape}, "
          f"end={out_tf_s['end_logits'].shape}, params={span_tf.count_parameters():,}")
