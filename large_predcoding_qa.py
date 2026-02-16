"""
Large-Scale Predictive Coding for Question Answering

Definitive test: Does mean pooling kill QA at scale?

Phase 5 showed both PredCoding (1.5M) and Transformer (1.4M) failed equally
on SQuAD. This was inconclusive - models were too small.

This experiment scales up to test the REAL question:
  At BERT-scale, does pooling become a bottleneck?

Architecture: PC-Pointer
  1. BERT embeddings (subword tokens + position + segment)
  2. Mean pool to single vector [batch, dim]
  3. Iterative PC refinement (bidirectional error-driven updates)
  4. Cross-attention from refined vector back to token embeddings
  5. Predict start/end logits via dot-product attention

The PC-Pointer approach is the fairest test: the pooled vector must learn
to encode enough positional information to "point" at answer spans.

Scales:
  - Medium (CPU-feasible): 256-dim, 6 layers, ~15M params
  - Large (GPU required):  768-dim, 12 layers, ~110M params
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import math


class LargePredCodingLayer(nn.Module):
    """
    Scaled predictive coding layer with dropout and pre-norm.

    Same principles as Phase 3's PredictiveCodingLayer but with:
    - Dropout for regularization at scale
    - Pre-norm for stable deep networks
    - Larger hidden dimensions
    """

    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        hidden_dim = hidden_dim or dim * 4

        # Feedforward init (with residual, from Phase 3 fix)
        self.ff_transform = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.ff_norm = nn.LayerNorm(dim)

        # Top-down prediction
        self.predict_down = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

        # Bottom-up error processing
        self.process_error = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.update_norm = nn.LayerNorm(dim)

        # Learnable step size
        self.step_size = nn.Parameter(torch.tensor(0.3))

    def init_representation(self, input_below):
        """Feedforward initialization with residual."""
        return self.ff_norm(input_below + self.ff_transform(input_below))

    def predict(self, representation):
        """Top-down prediction."""
        return self.predict_down(representation)

    def compute_error(self, actual, prediction):
        """Prediction error."""
        return actual - prediction

    def update(self, representation, error):
        """Error-driven update with residual."""
        correction = self.process_error(error)
        return self.update_norm(representation + self.step_size * correction)


class LargePredCodingQA(nn.Module):
    """
    Large-scale PredCoding for span extraction QA.

    Architecture: PC-Pointer
      1. Subword embeddings (BERT's WordPiece) + position + segment
      2. Mean pool sequence to single vector
      3. Iterative PC refinement (20 iterations)
      4. Cross-attend from refined vector back to token embeddings
      5. Start/end logits via dot-product attention

    This tests the CORE question: can a pooled vector learn to
    encode positional information sufficient for span extraction?
    """

    def __init__(
        self,
        vocab_size: int = 30522,  # BERT's WordPiece vocab
        embed_dim: int = 768,
        hidden_dim: int = 3072,
        num_layers: int = 12,
        max_iterations: int = 20,
        convergence_threshold: float = 0.01,
        max_seq_len: int = 384,
        dropout: float = 0.1,
        bidirectional: bool = True,
        use_bert_embeddings: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.bidirectional = bidirectional
        self.use_bert_embeddings = use_bert_embeddings

        if use_bert_embeddings:
            # Use BERT's embedding layer (position + token + segment)
            from transformers import BertModel
            bert = BertModel.from_pretrained('bert-base-uncased')
            self.embeddings = bert.embeddings
            # Freeze or fine-tune based on strategy
        else:
            # Custom embeddings (for medium scale or standalone)
            self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
            self.position_embeddings = nn.Embedding(max_seq_len, embed_dim)
            self.token_type_embeddings = nn.Embedding(2, embed_dim)
            self.embed_norm = nn.LayerNorm(embed_dim)
            self.embed_dropout = nn.Dropout(dropout)

        # Predictive coding layers
        self.pc_layers = nn.ModuleList([
            LargePredCodingLayer(embed_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # PC-Pointer: cross-attention from pooled vector back to tokens
        self.start_query_proj = nn.Linear(embed_dim, embed_dim)
        self.end_query_proj = nn.Linear(embed_dim, embed_dim)
        self.token_key_proj = nn.Linear(embed_dim, embed_dim)
        self.token_value_proj = nn.Linear(embed_dim, embed_dim)

        # Multi-head cross-attention for richer pointing
        self.num_heads = max(embed_dim // 64, 1)
        self.head_dim = embed_dim // self.num_heads

        # Start/end output projections
        self.start_output = nn.Linear(embed_dim, 1)
        self.end_output = nn.Linear(embed_dim, 1)

        # Answerable classifier (for SQuAD 2.0)
        self.answerable_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 2),
        )

    def get_embeddings(self, input_ids, token_type_ids=None, attention_mask=None):
        """Get token embeddings."""
        if self.use_bert_embeddings:
            return self.embeddings(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
            )
        else:
            seq_len = input_ids.size(1)
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

            word_embeds = self.word_embeddings(input_ids)
            pos_embeds = self.position_embeddings(position_ids)

            if token_type_ids is not None:
                type_embeds = self.token_type_embeddings(token_type_ids)
            else:
                type_embeds = self.token_type_embeddings(
                    torch.zeros_like(input_ids)
                )

            embeds = word_embeds + pos_embeds + type_embeds
            embeds = self.embed_norm(embeds)
            embeds = self.embed_dropout(embeds)
            return embeds

    def pc_inference(self, pooled):
        """
        Iterative predictive coding on pooled vector.

        Args: pooled [batch, embed_dim]
        Returns: refined [batch, embed_dim], ff_output [batch, embed_dim]
        """
        # Phase 1: Feedforward init (residual connections)
        reps = []
        current = pooled
        for layer in self.pc_layers:
            current = layer.init_representation(current)
            reps.append(current)
        ff_output = reps[-1]

        if self.max_iterations == 0:
            return ff_output, ff_output

        # Phase 2: Iterative PC refinement
        for iteration in range(self.max_iterations):
            old_reps_detached = [r.detach() for r in reps]

            # Bottom-up errors
            errors = []
            pred_0 = self.pc_layers[0].predict(reps[0])
            errors.append(pooled - pred_0)

            for i in range(1, self.num_layers):
                pred_i = self.pc_layers[i].predict(reps[i])
                errors.append(reps[i - 1].detach() - pred_i)

            # Update all layers
            new_reps = []
            for i in range(self.num_layers):
                error_below = errors[i]
                if self.bidirectional and i < self.num_layers - 1:
                    pred_from_above = self.pc_layers[i + 1].predict(reps[i + 1])
                    error_above = reps[i] - pred_from_above
                else:
                    error_above = torch.zeros_like(reps[i])
                total_error = error_below + 0.5 * error_above
                new_reps.append(self.pc_layers[i].update(reps[i], total_error))
            reps = new_reps

            # Check convergence
            with torch.no_grad():
                max_change = max(
                    (new - old).abs().mean().item()
                    for new, old in zip(reps, old_reps_detached)
                )
                if max_change < self.convergence_threshold:
                    break

        # Skip connection
        refined = reps[-1] + ff_output
        return refined, ff_output

    def pointer_attention(self, query, token_keys, token_values, attention_mask=None):
        """
        Multi-head cross-attention from pooled query to token keys.

        Args:
            query: [batch, embed_dim] - from refined pooled vector
            token_keys: [batch, seq_len, embed_dim]
            token_values: [batch, seq_len, embed_dim]
            attention_mask: [batch, seq_len] - 1 for real tokens, 0 for padding
        Returns:
            attended: [batch, seq_len] - attention scores (logits)
        """
        batch, seq_len, _ = token_keys.shape

        # Reshape for multi-head attention
        # query: [batch, num_heads, 1, head_dim]
        q = query.view(batch, self.num_heads, self.head_dim).unsqueeze(2)
        # keys: [batch, num_heads, seq_len, head_dim]
        k = token_keys.view(batch, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores: [batch, num_heads, 1, seq_len]

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)

        # Average across heads to get per-position logits
        logits = scores.squeeze(2).mean(dim=1)  # [batch, seq_len]
        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        start_positions: torch.Tensor = None,
        end_positions: torch.Tensor = None,
        **kwargs,
    ) -> Dict:
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len] - 1 for real tokens
            token_type_ids: [batch, seq_len] - 0 for context, 1 for question
            start_positions: [batch] - for training
            end_positions: [batch] - for training
        Returns:
            dict with start_logits, end_logits, answerable_logits, loss (if training)
        """
        batch, seq_len = input_ids.shape

        if attention_mask is None:
            attention_mask = (input_ids != 0).long()

        # Step 1: Get token embeddings
        token_embeds = self.get_embeddings(input_ids, token_type_ids, attention_mask)
        # [batch, seq_len, embed_dim]

        # Step 2: Pool (THE CRITICAL STEP - destroys positions!)
        # Mask out padding for proper mean
        mask_expanded = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
        masked_embeds = token_embeds * mask_expanded
        pooled = masked_embeds.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        # [batch, embed_dim]

        # Step 3: PC refinement
        refined, ff_output = self.pc_inference(pooled)
        # [batch, embed_dim]

        # Step 4: PC-Pointer - cross-attend back to tokens
        start_query = self.start_query_proj(refined)  # [batch, embed_dim]
        end_query = self.end_query_proj(refined)

        token_keys = self.token_key_proj(token_embeds)  # [batch, seq_len, embed_dim]
        token_values = self.token_value_proj(token_embeds)

        start_logits = self.pointer_attention(
            start_query, token_keys, token_values, attention_mask
        )  # [batch, seq_len]
        end_logits = self.pointer_attention(
            end_query, token_keys, token_values, attention_mask
        )  # [batch, seq_len]

        # Step 5: Answerable prediction
        answerable_logits = self.answerable_head(refined)  # [batch, 2]

        outputs = {
            'start_logits': start_logits,
            'end_logits': end_logits,
            'answerable_logits': answerable_logits,
        }

        # Compute loss if labels provided
        if start_positions is not None and end_positions is not None:
            # Clamp positions to valid range
            start_positions = start_positions.clamp(0, seq_len - 1)
            end_positions = end_positions.clamp(0, seq_len - 1)

            start_loss = F.cross_entropy(start_logits, start_positions)
            end_loss = F.cross_entropy(end_logits, end_positions)
            outputs['loss'] = start_loss + end_loss

        info = {
            'steps_taken': float(self.max_iterations),
            'per_example_steps': torch.full((batch,), float(self.max_iterations)),
            'changes_history': [],
        }
        outputs['info'] = info

        return outputs

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BERTBaselineQA(nn.Module):
    """
    BERT-base fine-tuned for QA.

    Standard approach: BERT encodes tokens with full self-attention,
    then linear layer predicts start/end logits per token.

    This is the gold standard that PredCoding must compete against.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        if pretrained:
            from transformers import BertModel
            self.bert = BertModel.from_pretrained('bert-base-uncased')
        else:
            from transformers import BertConfig, BertModel
            config = BertConfig()
            self.bert = BertModel(config)

        self.qa_outputs = nn.Linear(768, 2)
        self.answerable_head = nn.Sequential(
            nn.Linear(768, 384),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(384, 2),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        start_positions: torch.Tensor = None,
        end_positions: torch.Tensor = None,
        **kwargs,
    ) -> Dict:
        batch, seq_len = input_ids.shape

        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = bert_out.last_hidden_state  # [batch, seq_len, 768]
        pooled_output = bert_out.pooler_output  # [batch, 768]

        logits = self.qa_outputs(sequence_output)  # [batch, seq_len, 2]
        start_logits = logits[:, :, 0]  # [batch, seq_len]
        end_logits = logits[:, :, 1]    # [batch, seq_len]

        answerable_logits = self.answerable_head(pooled_output)

        outputs = {
            'start_logits': start_logits,
            'end_logits': end_logits,
            'answerable_logits': answerable_logits,
        }

        if start_positions is not None and end_positions is not None:
            start_positions = start_positions.clamp(0, seq_len - 1)
            end_positions = end_positions.clamp(0, seq_len - 1)
            start_loss = F.cross_entropy(start_logits, start_positions)
            end_loss = F.cross_entropy(end_logits, end_positions)
            outputs['loss'] = start_loss + end_loss

        info = {
            'steps_taken': 0.0,
            'per_example_steps': torch.zeros(batch),
            'changes_history': [],
        }
        outputs['info'] = info

        return outputs

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Scale configurations
# ============================================================================

def build_medium_predcoding(vocab_size=30522):
    """Medium scale: ~15M params, CPU-feasible."""
    return LargePredCodingQA(
        vocab_size=vocab_size,
        embed_dim=256,
        hidden_dim=1024,
        num_layers=6,
        max_iterations=10,
        convergence_threshold=0.01,
        max_seq_len=384,
        dropout=0.1,
        use_bert_embeddings=False,
    )


def build_large_predcoding():
    """Large scale: ~110M params, GPU required."""
    return LargePredCodingQA(
        vocab_size=30522,
        embed_dim=768,
        hidden_dim=3072,
        num_layers=12,
        max_iterations=20,
        convergence_threshold=0.01,
        max_seq_len=384,
        dropout=0.1,
        use_bert_embeddings=True,
    )


def build_medium_transformer(vocab_size=30522):
    """Medium transformer baseline matching medium PredCoding scale."""
    from transformers import BertConfig, BertModel

    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=4,
        intermediate_size=1024,
        max_position_embeddings=384,
    )
    bert = BertModel(config)

    class MediumTransformerQA(nn.Module):
        def __init__(self, bert_model):
            super().__init__()
            self.bert = bert_model
            self.qa_outputs = nn.Linear(256, 2)
            self.answerable_head = nn.Sequential(
                nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 2),
            )

        def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                    start_positions=None, end_positions=None, **kwargs):
            batch, seq_len = input_ids.shape
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
            seq_out = out.last_hidden_state
            pooled = out.pooler_output

            logits = self.qa_outputs(seq_out)
            start_logits = logits[:, :, 0]
            end_logits = logits[:, :, 1]
            answerable_logits = self.answerable_head(pooled)

            outputs = {
                'start_logits': start_logits,
                'end_logits': end_logits,
                'answerable_logits': answerable_logits,
            }
            if start_positions is not None and end_positions is not None:
                start_positions = start_positions.clamp(0, seq_len - 1)
                end_positions = end_positions.clamp(0, seq_len - 1)
                outputs['loss'] = (F.cross_entropy(start_logits, start_positions) +
                                   F.cross_entropy(end_logits, end_positions))
            outputs['info'] = {
                'steps_taken': 0.0,
                'per_example_steps': torch.zeros(batch),
                'changes_history': [],
            }
            return outputs

        def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    return MediumTransformerQA(bert)


# ============================================================================
# Quick test
# ============================================================================

if __name__ == "__main__":
    print("Large PredCoding QA - Architecture Test")
    print("=" * 60)

    batch = 2
    seq_len = 128
    vocab_size = 30522

    # Medium scale (CPU-feasible)
    print("\n--- Medium Scale ---")
    pc_medium = build_medium_predcoding(vocab_size)
    tf_medium = build_medium_transformer(vocab_size)

    x = torch.randint(0, vocab_size, (batch, seq_len))
    mask = torch.ones(batch, seq_len, dtype=torch.long)
    types = torch.zeros(batch, seq_len, dtype=torch.long)
    starts = torch.randint(0, seq_len, (batch,))
    ends = torch.clamp(starts + torch.randint(1, 5, (batch,)), max=seq_len - 1)

    out_pc = pc_medium(x, attention_mask=mask, token_type_ids=types,
                       start_positions=starts, end_positions=ends)
    out_tf = tf_medium(x, attention_mask=mask, token_type_ids=types,
                       start_positions=starts, end_positions=ends)

    print(f"  PredCoding Medium:")
    print(f"    Params: {pc_medium.count_parameters():,}")
    print(f"    start_logits: {out_pc['start_logits'].shape}")
    print(f"    loss: {out_pc['loss'].item():.4f}")

    print(f"  Transformer Medium:")
    print(f"    Params: {tf_medium.count_parameters():,}")
    print(f"    start_logits: {out_tf['start_logits'].shape}")
    print(f"    loss: {out_tf['loss'].item():.4f}")

    print(f"\n  Param ratio: {pc_medium.count_parameters() / tf_medium.count_parameters():.2f}x")
    print("\n  Architecture verified!")
