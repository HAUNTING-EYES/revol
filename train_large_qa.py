"""
Training Script for Large-Scale PredCoding QA on SQuAD 2.0

Uses BERT tokenizer (WordPiece subwords) for fair comparison.
Supports both medium (CPU) and large (GPU) scale experiments.

Usage:
  python train_large_qa.py medium    # ~15M params, CPU, 5K samples
  python train_large_qa.py large     # ~110M params, GPU, full SQuAD
  python train_large_qa.py analysis  # Error analysis on trained models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import json
import os
import sys
import numpy as np
from collections import OrderedDict

from large_predcoding_qa import (
    LargePredCodingQA, BERTBaselineQA,
    build_medium_predcoding, build_medium_transformer,
    build_large_predcoding,
)


# ============================================================================
# SQuAD Dataset with BERT Tokenizer
# ============================================================================

class SQuADv2Dataset(Dataset):
    """SQuAD 2.0 with BERT WordPiece tokenization."""

    def __init__(self, split: str, max_len: int = 384, max_samples: int = None):
        from datasets import load_dataset
        from transformers import BertTokenizerFast

        print(f"  Loading SQuAD 2.0 {split}...")
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

        ds = load_dataset("squad_v2", split=split)
        ds = ds.shuffle(seed=42)
        if max_samples and max_samples < len(ds):
            ds = ds.select(range(max_samples))

        self.max_len = max_len
        self.examples = []
        self._preprocess(ds)

    def _preprocess(self, ds):
        """Tokenize and find answer spans in tokenized sequence."""
        skipped = 0
        for idx in range(len(ds)):
            example = ds[idx]
            context = example["context"]
            question = example["question"]
            answers = example["answers"]

            # Tokenize [CLS] question [SEP] context [SEP]
            encoding = self.tokenizer(
                question, context,
                max_length=self.max_len,
                truncation=True,
                padding="max_length",
                return_offsets_mapping=True,
                return_token_type_ids=True,
            )

            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]
            token_type_ids = encoding["token_type_ids"]
            offset_mapping = encoding["offset_mapping"]

            if len(answers["text"]) == 0:
                # Unanswerable
                self.examples.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                    "start_position": 0,
                    "end_position": 0,
                    "is_answerable": 0,
                    "answer_text": "",
                    "question": question,
                    "context": context,
                })
            else:
                answer_text = answers["text"][0]
                answer_start_char = answers["answer_start"][0]
                answer_end_char = answer_start_char + len(answer_text)

                # Find token positions that correspond to the answer span
                # offset_mapping gives (start_char, end_char) for each token
                start_token = None
                end_token = None

                for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping):
                    if tok_start is None or token_type_ids[tok_idx] == 0:
                        continue  # Skip question tokens and special tokens
                    if tok_start <= answer_start_char < tok_end:
                        start_token = tok_idx
                    if tok_start < answer_end_char <= tok_end:
                        end_token = tok_idx

                if start_token is None or end_token is None:
                    # Answer was truncated
                    skipped += 1
                    continue

                self.examples.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                    "start_position": start_token,
                    "end_position": end_token,
                    "is_answerable": 1,
                    "answer_text": answer_text,
                    "question": question,
                    "context": context,
                })

        if skipped > 0:
            print(f"    Skipped {skipped} truncated answers")
        print(f"    Processed {len(self.examples)} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(ex["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(ex["token_type_ids"], dtype=torch.long),
            "start_positions": torch.tensor(ex["start_position"], dtype=torch.long),
            "end_positions": torch.tensor(ex["end_position"], dtype=torch.long),
            "is_answerable": torch.tensor(ex["is_answerable"], dtype=torch.long),
        }


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_epoch(model, dataloader, optimizer, device, max_grad_norm=1.0,
                scheduler=None):
    """Train one epoch."""
    model.train()
    total_loss = 0
    total = 0

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=start_positions,
            end_positions=end_positions,
        )

        loss = outputs['loss']
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item() * input_ids.size(0)
        total += input_ids.size(0)

        if (batch_idx + 1) % 50 == 0:
            print(f"    Batch {batch_idx+1}: loss={total_loss/total:.4f}")

    return total_loss / total


def evaluate_squad(model, dataloader, device):
    """
    Evaluate span extraction.
    Returns exact match, token-level F1, and per-example details.
    """
    model.eval()
    total_em = 0
    total_f1 = 0.0
    total = 0
    total_answerable = 0
    answerable_em = 0
    answerable_f1 = 0.0

    per_example_results = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            start_positions = batch["start_positions"]
            end_positions = batch["end_positions"]
            is_answerable = batch["is_answerable"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            pred_starts = outputs['start_logits'].argmax(dim=1).cpu()
            pred_ends = outputs['end_logits'].argmax(dim=1).cpu()
            # Fix invalid spans
            pred_ends = torch.max(pred_ends, pred_starts)

            for i in range(input_ids.size(0)):
                total += 1
                ps, pe = pred_starts[i].item(), pred_ends[i].item()
                gs, ge = start_positions[i].item(), end_positions[i].item()

                # Exact match
                em = 1.0 if (ps == gs and pe == ge) else 0.0
                total_em += em

                # Token F1
                pred_set = set(range(ps, pe + 1))
                gold_set = set(range(gs, ge + 1))

                if is_answerable[i] == 0:
                    # Unanswerable: EM if model predicts position 0
                    f1 = 1.0 if ps == 0 and pe == 0 else 0.0
                elif len(pred_set) == 0 or len(gold_set) == 0:
                    f1 = 0.0
                else:
                    overlap = len(pred_set & gold_set)
                    prec = overlap / len(pred_set)
                    rec = overlap / len(gold_set)
                    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

                total_f1 += f1

                if is_answerable[i] == 1:
                    total_answerable += 1
                    answerable_em += em
                    answerable_f1 += f1

                per_example_results.append({
                    'pred_start': ps, 'pred_end': pe,
                    'gold_start': gs, 'gold_end': ge,
                    'is_answerable': is_answerable[i].item(),
                    'em': em, 'f1': f1,
                })

    results = {
        'overall_em': total_em / max(total, 1),
        'overall_f1': total_f1 / max(total, 1),
        'answerable_em': answerable_em / max(total_answerable, 1),
        'answerable_f1': answerable_f1 / max(total_answerable, 1),
        'num_total': total,
        'num_answerable': total_answerable,
    }
    return results, per_example_results


# ============================================================================
# Medium-Scale Experiment (CPU)
# ============================================================================

def run_medium_experiment(
    max_seq_len: int = 384,
    batch_size: int = 8,
    num_epochs: int = 5,
    learning_rate: float = 5e-4,
    max_train_samples: int = 5000,
    max_test_samples: int = 1000,
):
    """
    Medium-scale experiment on CPU.
    ~15M params, BERT tokenizer, 5K train samples.
    """
    print("=" * 80)
    print("MEDIUM-SCALE SQuAD EXPERIMENT")
    print("PredCoding (15M) vs Transformer (13M) with BERT tokenizer")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Config: seq_len={max_seq_len}, batch={batch_size}, epochs={num_epochs}")
    print(f"Train: {max_train_samples}, Test: {max_test_samples}")

    # Load SQuAD with BERT tokenizer
    print("\n--- Loading SQuAD 2.0 ---")
    train_dataset = SQuADv2Dataset("train", max_len=max_seq_len,
                                    max_samples=max_train_samples)
    test_dataset = SQuADv2Dataset("validation", max_len=max_seq_len,
                                   max_samples=max_test_samples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=0)

    # Count answerable/unanswerable
    train_ans = sum(1 for ex in train_dataset.examples if ex['is_answerable'])
    test_ans = sum(1 for ex in test_dataset.examples if ex['is_answerable'])
    print(f"\n  Train: {len(train_dataset)} total, {train_ans} answerable")
    print(f"  Test:  {len(test_dataset)} total, {test_ans} answerable")

    # Build models
    vocab_size = 30522  # BERT's vocab
    models = OrderedDict()

    models["PredCoding_PC10"] = build_medium_predcoding(vocab_size)
    models["PredCoding_PC0"] = LargePredCodingQA(
        vocab_size=vocab_size, embed_dim=256, hidden_dim=1024,
        num_layers=6, max_iterations=0, max_seq_len=max_seq_len, dropout=0.1,
    )
    models["Transformer_6L"] = build_medium_transformer(vocab_size)

    print(f"\n--- Models ---")
    for name, model in models.items():
        print(f"  {name:<25} {model.count_parameters():>12,} params")

    # Train all models
    all_results = {}

    for name, model in models.items():
        model = model.to(device)
        print(f"\n{'='*80}")
        print(f"TRAINING: {name}")
        print(f"{'='*80}")

        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = num_epochs * len(train_loader)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=learning_rate, total_steps=total_steps
        )

        best_f1 = 0
        total_time = 0
        epoch_data = []

        for epoch in range(num_epochs):
            epoch_start = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, device,
                                     scheduler=scheduler)
            epoch_time = time.time() - epoch_start
            total_time += epoch_time

            metrics, _ = evaluate_squad(model, test_loader, device)
            best_f1 = max(best_f1, metrics['answerable_f1'])

            print(f"  Epoch {epoch+1}/{num_epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"EM: {metrics['overall_em']:.4f} | "
                  f"F1: {metrics['overall_f1']:.4f} | "
                  f"Ans_F1: {metrics['answerable_f1']:.4f} | "
                  f"Time: {epoch_time:.1f}s")

            epoch_data.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'overall_em': metrics['overall_em'],
                'overall_f1': metrics['overall_f1'],
                'answerable_f1': metrics['answerable_f1'],
                'time': epoch_time,
            })

        # Final eval
        final_metrics, per_example = evaluate_squad(model, test_loader, device)

        all_results[name] = {
            'final_overall_em': final_metrics['overall_em'],
            'final_overall_f1': final_metrics['overall_f1'],
            'final_answerable_em': final_metrics['answerable_em'],
            'final_answerable_f1': final_metrics['answerable_f1'],
            'best_answerable_f1': best_f1,
            'total_train_time': total_time,
            'parameters': model.count_parameters(),
            'epochs': epoch_data,
            'per_example': per_example,
        }

        # Move model off device to free memory
        model = model.cpu()

    # Summary
    print(f"\n{'='*80}")
    print("MEDIUM-SCALE SQuAD 2.0 RESULTS")
    print(f"{'='*80}")
    print(f"\n{'Model':<25} {'EM':>8} {'F1':>8} {'Ans_EM':>8} {'Ans_F1':>8} {'Best_F1':>8} {'Params':>12}")
    print("-" * 82)
    for name, r in all_results.items():
        print(f"{name:<25} {r['final_overall_em']:>7.4f} {r['final_overall_f1']:>7.4f} "
              f"{r['final_answerable_em']:>7.4f} {r['final_answerable_f1']:>7.4f} "
              f"{r['best_answerable_f1']:>7.4f} {r['parameters']:>12,}")

    # Hypothesis test
    pc_f1 = all_results.get("PredCoding_PC10", {}).get("final_answerable_f1", 0)
    tf_f1 = all_results.get("Transformer_6L", {}).get("final_answerable_f1", 0)
    gap = tf_f1 - pc_f1

    print(f"\n--- Hypothesis Test ---")
    print(f"  Transformer F1: {tf_f1:.4f}")
    print(f"  PredCoding F1:  {pc_f1:.4f}")
    print(f"  Gap:            {gap:+.4f}")

    if gap < 0.05:
        print(f"  HYPOTHESIS B SUPPORTED: Pooling viable for QA (gap < 5%)")
    elif gap < 0.15:
        print(f"  INTERMEDIATE: Pooling hurts but doesn't kill QA (5-15% gap)")
    else:
        print(f"  HYPOTHESIS A SUPPORTED: Pooling kills QA (gap > 15%)")

    # Error analysis by question type
    print(f"\n--- Error Analysis by Question Type ---")
    question_types = {}
    test_examples = test_dataset.examples

    for i, ex in enumerate(test_examples):
        if i >= len(all_results.get("PredCoding_PC10", {}).get("per_example", [])):
            break

        q = ex["question"].lower()
        if q.startswith("what") or "what " in q[:15]:
            qtype = "what"
        elif q.startswith("who") or "who " in q[:15]:
            qtype = "who"
        elif q.startswith("when") or "when " in q[:15]:
            qtype = "when"
        elif q.startswith("where") or "where " in q[:15]:
            qtype = "where"
        elif q.startswith("why") or "why " in q[:15]:
            qtype = "why"
        elif q.startswith("how") or "how " in q[:15]:
            qtype = "how"
        elif q.startswith("which") or "which " in q[:15]:
            qtype = "which"
        else:
            qtype = "other"

        if qtype not in question_types:
            question_types[qtype] = {"pc_f1s": [], "tf_f1s": []}

        pc_per = all_results.get("PredCoding_PC10", {}).get("per_example", [])
        tf_per = all_results.get("Transformer_6L", {}).get("per_example", [])

        if i < len(pc_per):
            question_types[qtype]["pc_f1s"].append(pc_per[i]["f1"])
        if i < len(tf_per):
            question_types[qtype]["tf_f1s"].append(tf_per[i]["f1"])

    print(f"\n  {'Q-Type':<10} {'Count':>6} {'PC F1':>8} {'TF F1':>8} {'Gap':>8}")
    print("  " + "-" * 45)
    analysis_by_type = {}
    for qtype, data in sorted(question_types.items(), key=lambda x: -len(x[1]["pc_f1s"])):
        n = len(data["pc_f1s"])
        pc_f1_avg = np.mean(data["pc_f1s"]) if data["pc_f1s"] else 0
        tf_f1_avg = np.mean(data["tf_f1s"]) if data["tf_f1s"] else 0
        gap_t = tf_f1_avg - pc_f1_avg
        print(f"  {qtype:<10} {n:>6} {pc_f1_avg:>7.3f} {tf_f1_avg:>7.3f} {gap_t:>+7.3f}")
        analysis_by_type[qtype] = {
            "count": n,
            "pc_f1": float(pc_f1_avg),
            "tf_f1": float(tf_f1_avg),
            "gap": float(gap_t),
        }

    # Error analysis by answer length
    print(f"\n--- Error Analysis by Answer Length ---")
    length_bins = {"1-2 tokens": (1, 3), "3-5 tokens": (3, 6), "6+ tokens": (6, 100)}
    length_data = {k: {"pc_f1s": [], "tf_f1s": []} for k in length_bins}

    for i, ex in enumerate(test_examples):
        if not ex['is_answerable'] or i >= len(all_results.get("PredCoding_PC10", {}).get("per_example", [])):
            continue

        answer_len = len(ex['answer_text'].split())
        for bin_name, (lo, hi) in length_bins.items():
            if lo <= answer_len < hi:
                pc_per = all_results.get("PredCoding_PC10", {}).get("per_example", [])
                tf_per = all_results.get("Transformer_6L", {}).get("per_example", [])
                if i < len(pc_per):
                    length_data[bin_name]["pc_f1s"].append(pc_per[i]["f1"])
                if i < len(tf_per):
                    length_data[bin_name]["tf_f1s"].append(tf_per[i]["f1"])
                break

    print(f"  {'Length':<12} {'Count':>6} {'PC F1':>8} {'TF F1':>8} {'Gap':>8}")
    print("  " + "-" * 45)
    analysis_by_length = {}
    for bin_name, data in length_data.items():
        n = len(data["pc_f1s"])
        if n > 0:
            pc_avg = np.mean(data["pc_f1s"])
            tf_avg = np.mean(data["tf_f1s"]) if data["tf_f1s"] else 0
            gap_l = tf_avg - pc_avg
            print(f"  {bin_name:<12} {n:>6} {pc_avg:>7.3f} {tf_avg:>7.3f} {gap_l:>+7.3f}")
            analysis_by_length[bin_name] = {
                "count": n, "pc_f1": float(pc_avg), "tf_f1": float(tf_avg), "gap": float(gap_l)
            }

    # Error analysis by answer position
    print(f"\n--- Error Analysis by Answer Position ---")
    pos_bins = {"beginning (0-25%)": (0, 0.25), "middle (25-75%)": (0.25, 0.75), "end (75-100%)": (0.75, 1.01)}
    pos_data = {k: {"pc_f1s": [], "tf_f1s": []} for k in pos_bins}

    for i, ex in enumerate(test_examples):
        if not ex['is_answerable'] or i >= len(all_results.get("PredCoding_PC10", {}).get("per_example", [])):
            continue

        context_len = len(ex['context'].split())
        if context_len == 0:
            continue
        answer_start_word = ex['context'][:ex['context'].find(ex['answer_text'])].count(' ')
        rel_pos = answer_start_word / max(context_len, 1)

        for bin_name, (lo, hi) in pos_bins.items():
            if lo <= rel_pos < hi:
                pc_per = all_results.get("PredCoding_PC10", {}).get("per_example", [])
                tf_per = all_results.get("Transformer_6L", {}).get("per_example", [])
                if i < len(pc_per):
                    pos_data[bin_name]["pc_f1s"].append(pc_per[i]["f1"])
                if i < len(tf_per):
                    pos_data[bin_name]["tf_f1s"].append(tf_per[i]["f1"])
                break

    print(f"  {'Position':<20} {'Count':>6} {'PC F1':>8} {'TF F1':>8} {'Gap':>8}")
    print("  " + "-" * 55)
    analysis_by_position = {}
    for bin_name, data in pos_data.items():
        n = len(data["pc_f1s"])
        if n > 0:
            pc_avg = np.mean(data["pc_f1s"])
            tf_avg = np.mean(data["tf_f1s"]) if data["tf_f1s"] else 0
            gap_p = tf_avg - pc_avg
            print(f"  {bin_name:<20} {n:>6} {pc_avg:>7.3f} {tf_avg:>7.3f} {gap_p:>+7.3f}")
            analysis_by_position[bin_name] = {
                "count": n, "pc_f1": float(pc_avg), "tf_f1": float(tf_avg), "gap": float(gap_p)
            }

    # Save results
    save_data = {}
    for name, r in all_results.items():
        save_data[name] = {k: v for k, v in r.items() if k != 'per_example'}

    save_data["analysis"] = {
        "by_question_type": analysis_by_type,
        "by_answer_length": analysis_by_length,
        "by_answer_position": analysis_by_position,
    }

    save_path = os.path.join(os.path.dirname(__file__), "large_qa_results.json")
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")

    return all_results


# ============================================================================
# Large-Scale Experiment (GPU)
# ============================================================================

def run_large_experiment(
    max_seq_len: int = 384,
    batch_size: int = 16,
    num_epochs: int = 3,
    learning_rate: float = 3e-5,
    max_train_samples: int = None,
    max_test_samples: int = None,
):
    """
    Large-scale experiment on GPU.
    110M params, full SQuAD 2.0 dataset.
    """
    print("=" * 80)
    print("LARGE-SCALE SQuAD EXPERIMENT (GPU REQUIRED)")
    print("PredCoding (110M) vs BERT-base (110M)")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("\n  WARNING: No GPU detected!")
        print("  Large-scale experiment requires GPU (A100/V100)")
        print("  Falling back to CPU with reduced samples...")
        max_train_samples = max_train_samples or 1000
        max_test_samples = max_test_samples or 500
        batch_size = 4

    print(f"\nDevice: {device}")

    # Load data
    print("\n--- Loading SQuAD 2.0 ---")
    train_dataset = SQuADv2Dataset("train", max_len=max_seq_len,
                                    max_samples=max_train_samples)
    test_dataset = SQuADv2Dataset("validation", max_len=max_seq_len,
                                   max_samples=max_test_samples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=0)

    # Build models
    models = OrderedDict()
    models["BERT_base"] = BERTBaselineQA(pretrained=True).to(device)
    models["LargePredCoding"] = build_large_predcoding().to(device)

    print(f"\n--- Models ---")
    for name, model in models.items():
        print(f"  {name:<25} {model.count_parameters():>12,} params")

    all_results = {}

    for name, model in models.items():
        print(f"\n{'='*80}")
        print(f"TRAINING: {name}")
        print(f"{'='*80}")

        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = num_epochs * len(train_loader)
        warmup_steps = min(500, total_steps // 10)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            return max(0, 1 - (step - warmup_steps) / max(total_steps - warmup_steps, 1))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        best_f1 = 0
        total_time = 0

        for epoch in range(num_epochs):
            epoch_start = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, device,
                                     scheduler=scheduler)
            epoch_time = time.time() - epoch_start
            total_time += epoch_time

            metrics, per_example = evaluate_squad(model, test_loader, device)
            best_f1 = max(best_f1, metrics['answerable_f1'])

            print(f"  Epoch {epoch+1}/{num_epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"EM: {metrics['overall_em']:.4f} | "
                  f"Ans_F1: {metrics['answerable_f1']:.4f} | "
                  f"Time: {epoch_time:.1f}s")

        final_metrics, _ = evaluate_squad(model, test_loader, device)
        all_results[name] = {
            'final_answerable_f1': final_metrics['answerable_f1'],
            'final_answerable_em': final_metrics['answerable_em'],
            'best_f1': best_f1,
            'parameters': model.count_parameters(),
            'total_time': total_time,
        }

    # Summary
    print(f"\n{'='*80}")
    print("LARGE-SCALE RESULTS")
    print(f"{'='*80}")
    for name, r in all_results.items():
        print(f"  {name}: Ans_F1={r['final_answerable_f1']:.4f}, "
              f"Ans_EM={r['final_answerable_em']:.4f}, "
              f"Params={r['parameters']:,}")

    save_path = os.path.join(os.path.dirname(__file__), "large_scale_qa_results.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")

    return all_results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "medium"

    if task == "medium":
        run_medium_experiment()
    elif task == "large":
        run_large_experiment()
    else:
        print(f"Usage: python train_large_qa.py [medium|large]")
