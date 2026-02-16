"""
QA Training and Evaluation Infrastructure

Tests PredCoding on two QA tasks:
1. Multiple Choice (RACE) - classify which answer option is correct
2. Span Extraction (SQuAD 2.0) - predict start/end token positions

Compares PredCoding against Transformer baseline on both tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import json
import os
import numpy as np
from collections import Counter, OrderedDict

from train_wavenet import SimpleTokenizer
from predcoding_qa import (
    PredCodingMultipleChoice, PredCodingSpanExtraction,
    TransformerMultipleChoice, TransformerSpanExtraction,
)


# ============================================================================
# RACE Dataset (Multiple Choice)
# ============================================================================

class RACEDataset(Dataset):
    """RACE reading comprehension dataset for multiple choice QA."""

    SEP_TOKEN = "<sep>"

    def __init__(self, split: str, tokenizer: SimpleTokenizer, max_len: int = 256,
                 max_samples: int = None, difficulty: str = "middle"):
        from datasets import load_dataset

        print(f"  Loading RACE {difficulty}/{split}...")
        ds = load_dataset("race", difficulty, split=split)
        ds = ds.shuffle(seed=42)

        if max_samples and max_samples < len(ds):
            ds = ds.select(range(max_samples))

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.articles = ds["article"]
        self.questions = ds["question"]
        self.options = ds["options"]
        self.answers = ds["answer"]  # 'A', 'B', 'C', 'D'

        # Map answer letters to indices
        self.answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, idx):
        article = self.articles[idx]
        question = self.questions[idx]
        options = self.options[idx]
        answer = self.answer_map[self.answers[idx]]

        # Encode each option as: [article SEP question SEP option]
        all_ids = []
        for opt in options:
            text = f"{article} {self.SEP_TOKEN} {question} {self.SEP_TOKEN} {opt}"
            ids = self.tokenizer.encode(text, self.max_len)
            all_ids.append(ids)

        input_ids = torch.tensor(all_ids, dtype=torch.long)  # [4, max_len]
        label = torch.tensor(answer, dtype=torch.long)
        return input_ids, label


# ============================================================================
# SQuAD 2.0 Dataset (Span Extraction)
# ============================================================================

class SQuADDataset(Dataset):
    """SQuAD 2.0 dataset for span extraction QA."""

    SEP_TOKEN = "<sep>"

    def __init__(self, split: str, tokenizer: SimpleTokenizer, max_len: int = 256,
                 max_samples: int = None):
        from datasets import load_dataset

        print(f"  Loading SQuAD 2.0 {split}...")
        ds = load_dataset("squad_v2", split=split)
        ds = ds.shuffle(seed=42)

        if max_samples and max_samples < len(ds):
            ds = ds.select(range(max_samples))

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.contexts = ds["context"]
        self.questions = ds["question"]
        self.answers = ds["answers"]
        self.ids = ds["id"]

        # Precompute tokenized data for span finding
        self._preprocess()

    def _preprocess(self):
        """Precompute token-level answer spans."""
        self.input_texts = []
        self.start_positions = []
        self.end_positions = []
        self.is_answerable = []
        self.answer_texts = []

        for idx in range(len(self.contexts)):
            context = self.contexts[idx]
            question = self.questions[idx]
            answers = self.answers[idx]

            # Combine: [context SEP question]
            text = f"{context} {self.SEP_TOKEN} {question}"
            self.input_texts.append(text)

            if len(answers["text"]) == 0:
                # Unanswerable
                self.start_positions.append(0)
                self.end_positions.append(0)
                self.is_answerable.append(0)
                self.answer_texts.append("")
            else:
                answer_text = answers["text"][0]
                self.answer_texts.append(answer_text)
                self.is_answerable.append(1)

                # Find answer span in tokenized sequence
                # Tokenize context words to find the answer position
                context_words = context.lower().split()
                answer_words = answer_text.lower().split()

                # Search for answer span in context words
                start_pos = 0
                found = False
                for i in range(len(context_words) - len(answer_words) + 1):
                    if context_words[i:i + len(answer_words)] == answer_words:
                        start_pos = i
                        found = True
                        break

                if not found:
                    # Fuzzy match: find best overlap
                    best_overlap = 0
                    for i in range(len(context_words)):
                        overlap = 0
                        for j, aw in enumerate(answer_words):
                            if i + j < len(context_words) and context_words[i + j] == aw:
                                overlap += 1
                        if overlap > best_overlap:
                            best_overlap = overlap
                            start_pos = i

                end_pos = min(start_pos + len(answer_words) - 1, self.max_len - 1)
                start_pos = min(start_pos, self.max_len - 1)

                self.start_positions.append(start_pos)
                self.end_positions.append(end_pos)

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        ids = self.tokenizer.encode(self.input_texts[idx], self.max_len)
        input_ids = torch.tensor(ids, dtype=torch.long)
        start_pos = torch.tensor(self.start_positions[idx], dtype=torch.long)
        end_pos = torch.tensor(self.end_positions[idx], dtype=torch.long)
        is_answerable = torch.tensor(self.is_answerable[idx], dtype=torch.long)

        return input_ids, start_pos, end_pos, is_answerable


# ============================================================================
# Training Functions
# ============================================================================

def train_mc_epoch(model, dataloader, optimizer, criterion, device, max_grad_norm=1.0):
    """Train one epoch for multiple choice."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for input_ids, labels in dataloader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, info = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate_mc(model, dataloader, criterion, device):
    """Evaluate multiple choice model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits, info = model(input_ids)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


def train_span_epoch(model, dataloader, optimizer, device, max_grad_norm=1.0):
    """Train one epoch for span extraction."""
    model.train()
    total_loss = 0
    total = 0

    for input_ids, start_pos, end_pos, is_answerable in dataloader:
        input_ids = input_ids.to(device)
        start_pos = start_pos.to(device)
        end_pos = end_pos.to(device)
        is_answerable = is_answerable.to(device)

        optimizer.zero_grad()
        outputs, info = model(input_ids)

        # Span loss
        start_loss = nn.CrossEntropyLoss()(outputs['start_logits'], start_pos)
        end_loss = nn.CrossEntropyLoss()(outputs['end_logits'], end_pos)

        # Answerable loss
        ans_loss = nn.CrossEntropyLoss()(outputs['answerable_logits'], is_answerable)

        loss = start_loss + end_loss + 0.5 * ans_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
        total += input_ids.size(0)

    return total_loss / total


def evaluate_span(model, dataloader, device):
    """Evaluate span extraction model. Returns exact match and F1."""
    model.eval()
    total_em = 0
    total_f1 = 0.0
    total_ans_correct = 0
    total = 0
    total_answerable = 0
    total_answerable_correct = 0

    with torch.no_grad():
        for input_ids, start_pos, end_pos, is_answerable in dataloader:
            input_ids = input_ids.to(device)
            start_pos = start_pos.to(device)
            end_pos = end_pos.to(device)
            is_answerable = is_answerable.to(device)

            outputs, info = model(input_ids)

            # Predicted spans
            pred_start = outputs['start_logits'].argmax(dim=1)
            pred_end = outputs['end_logits'].argmax(dim=1)

            # Fix invalid spans (end < start)
            pred_end = torch.max(pred_end, pred_start)

            # Answerable prediction
            pred_ans = outputs['answerable_logits'].argmax(dim=1)
            total_ans_correct += (pred_ans == is_answerable).sum().item()

            # Exact match
            em = ((pred_start == start_pos) & (pred_end == end_pos)).float()
            total_em += em.sum().item()

            # Token-level F1 (for answerable questions)
            for i in range(input_ids.size(0)):
                total += 1
                if is_answerable[i] == 1:
                    total_answerable += 1
                    ps = pred_start[i].item()
                    pe = pred_end[i].item()
                    gs = start_pos[i].item()
                    ge = end_pos[i].item()

                    pred_set = set(range(ps, pe + 1))
                    gold_set = set(range(gs, ge + 1))

                    if len(pred_set) == 0 or len(gold_set) == 0:
                        f1 = 0.0
                    else:
                        overlap = len(pred_set & gold_set)
                        precision = overlap / len(pred_set) if pred_set else 0
                        recall = overlap / len(gold_set) if gold_set else 0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                    total_f1 += f1
                    if ps == gs and pe == ge:
                        total_answerable_correct += 1

    results = {
        'exact_match': total_em / max(total, 1),
        'f1': total_f1 / max(total_answerable, 1),
        'answerable_acc': total_ans_correct / max(total, 1),
        'answerable_em': total_answerable_correct / max(total_answerable, 1),
    }
    return results


# ============================================================================
# RACE Benchmark
# ============================================================================

def run_race_benchmark(
    max_seq_len: int = 256,
    batch_size: int = 16,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    vocab_size: int = 20000,
    max_train_samples: int = 5000,
    max_test_samples: int = 1000,
    embed_dim: int = 64,
):
    print("=" * 80)
    print("RACE MULTIPLE CHOICE BENCHMARK")
    print("PredCoding vs Transformer")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Config: seq_len={max_seq_len}, batch={batch_size}, epochs={num_epochs}")

    # Load RACE
    print("\n--- Loading RACE ---")
    from datasets import load_dataset
    raw_train = load_dataset("race", "middle", split="train")

    tokenizer = SimpleTokenizer(vocab_size=vocab_size)
    # Build vocab from RACE articles
    tokenizer.build_vocab(raw_train["article"])
    actual_vocab = len(tokenizer.word2idx)
    print(f"  Vocab: {actual_vocab}")

    train_dataset = RACEDataset("train", tokenizer, max_len=max_seq_len,
                                 max_samples=max_train_samples, difficulty="middle")
    test_dataset = RACEDataset("test", tokenizer, max_len=max_seq_len,
                                max_samples=max_test_samples, difficulty="middle")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # Models
    models = OrderedDict()
    models["PredCoding_MC"] = PredCodingMultipleChoice(
        vocab_size=actual_vocab, embed_dim=embed_dim, num_layers=4,
        max_iterations=10, convergence_threshold=0.1, max_seq_len=max_seq_len,
    ).to(device)

    models["PredCoding_MC_0iter"] = PredCodingMultipleChoice(
        vocab_size=actual_vocab, embed_dim=embed_dim, num_layers=4,
        max_iterations=0, max_seq_len=max_seq_len,
    ).to(device)

    models["Transformer_MC"] = TransformerMultipleChoice(
        vocab_size=actual_vocab, embed_dim=embed_dim, num_heads=4,
        num_layers=2, max_seq_len=max_seq_len,
    ).to(device)

    print(f"\n--- Parameter Counts ---")
    for name, model in models.items():
        print(f"  {name:<25} {model.count_parameters():>10,}")

    # Train
    all_results = {}
    criterion = nn.CrossEntropyLoss()

    for name, model in models.items():
        print(f"\n{'='*80}")
        print(f"TRAINING: {name}")
        print(f"{'='*80}")

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        best_test_acc = 0
        total_time = 0

        for epoch in range(num_epochs):
            epoch_start = time.time()
            train_loss, train_acc = train_mc_epoch(
                model, train_loader, optimizer, criterion, device
            )
            epoch_time = time.time() - epoch_start
            total_time += epoch_time

            test_loss, test_acc, _, _ = evaluate_mc(model, test_loader, criterion, device)
            scheduler.step()
            best_test_acc = max(best_test_acc, test_acc)

            print(f"  Epoch {epoch+1}/{num_epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Train: {train_acc:.4f} | "
                  f"Test: {test_acc:.4f} | "
                  f"Time: {epoch_time:.1f}s")

        final_loss, final_acc, preds, labels = evaluate_mc(
            model, test_loader, criterion, device
        )

        all_results[name] = {
            'final_test_acc': final_acc,
            'best_test_acc': best_test_acc,
            'total_train_time': total_time,
            'parameters': model.count_parameters(),
            'predictions': preds,
            'labels': labels,
        }

    # Summary
    print(f"\n{'='*80}")
    print("RACE RESULTS")
    print(f"{'='*80}")
    print(f"  Random baseline: 25.0%")
    print(f"{'Model':<25} {'Test Acc':>10} {'Best':>8} {'Params':>10} {'Time(s)':>10}")
    print("-" * 65)
    for name, r in all_results.items():
        print(f"{name:<25} {r['final_test_acc']:>9.4f} {r['best_test_acc']:>7.4f} "
              f"{r['parameters']:>10,} {r['total_train_time']:>9.1f}")

    # Verdict
    for name in ['PredCoding_MC', 'PredCoding_MC_0iter']:
        if name in all_results:
            acc = all_results[name]['final_test_acc']
            if acc >= 0.55:
                print(f"  {name}: STRONG SUCCESS (>= 55%)")
            elif acc >= 0.35:
                print(f"  {name}: PARTIAL SUCCESS (> random 25%, < 55%)")
            else:
                print(f"  {name}: NEAR RANDOM ({acc:.1%})")

    save_path = os.path.join(os.path.dirname(__file__), "race_results.json")
    serializable = {k: {kk: vv for kk, vv in v.items() if kk not in ('predictions', 'labels')}
                    for k, v in all_results.items()}
    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")

    return all_results


# ============================================================================
# SQuAD Benchmark
# ============================================================================

def run_squad_benchmark(
    max_seq_len: int = 256,
    batch_size: int = 16,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    vocab_size: int = 20000,
    max_train_samples: int = 5000,
    max_test_samples: int = 1000,
    embed_dim: int = 64,
):
    print("=" * 80)
    print("SQuAD 2.0 SPAN EXTRACTION BENCHMARK")
    print("PredCoding vs Transformer")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Config: seq_len={max_seq_len}, batch={batch_size}, epochs={num_epochs}")

    # Load SQuAD
    print("\n--- Loading SQuAD 2.0 ---")
    from datasets import load_dataset
    raw_train = load_dataset("squad_v2", split="train")

    tokenizer = SimpleTokenizer(vocab_size=vocab_size)
    tokenizer.build_vocab(raw_train["context"])
    actual_vocab = len(tokenizer.word2idx)
    print(f"  Vocab: {actual_vocab}")

    train_dataset = SQuADDataset("train", tokenizer, max_len=max_seq_len,
                                  max_samples=max_train_samples)
    test_dataset = SQuADDataset("validation", tokenizer, max_len=max_seq_len,
                                 max_samples=max_test_samples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # Count answerable
    train_ans = sum(train_dataset.is_answerable)
    test_ans = sum(test_dataset.is_answerable)
    print(f"  Train answerable: {train_ans}/{len(train_dataset)}")
    print(f"  Test answerable: {test_ans}/{len(test_dataset)}")

    # Models
    models = OrderedDict()
    models["PredCoding_Span"] = PredCodingSpanExtraction(
        vocab_size=actual_vocab, embed_dim=embed_dim, num_layers=4,
        max_iterations=10, convergence_threshold=0.1, max_seq_len=max_seq_len,
    ).to(device)

    models["PredCoding_Span_0iter"] = PredCodingSpanExtraction(
        vocab_size=actual_vocab, embed_dim=embed_dim, num_layers=4,
        max_iterations=0, max_seq_len=max_seq_len,
    ).to(device)

    models["Transformer_Span"] = TransformerSpanExtraction(
        vocab_size=actual_vocab, embed_dim=embed_dim, num_heads=4,
        num_layers=2, max_seq_len=max_seq_len,
    ).to(device)

    print(f"\n--- Parameter Counts ---")
    for name, model in models.items():
        print(f"  {name:<25} {model.count_parameters():>10,}")

    # Train
    all_results = {}

    for name, model in models.items():
        print(f"\n{'='*80}")
        print(f"TRAINING: {name}")
        print(f"{'='*80}")

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        best_f1 = 0
        total_time = 0

        for epoch in range(num_epochs):
            epoch_start = time.time()
            train_loss = train_span_epoch(model, train_loader, optimizer, device)
            epoch_time = time.time() - epoch_start
            total_time += epoch_time

            metrics = evaluate_span(model, test_loader, device)
            scheduler.step()
            best_f1 = max(best_f1, metrics['f1'])

            print(f"  Epoch {epoch+1}/{num_epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"EM: {metrics['exact_match']:.4f} | "
                  f"F1: {metrics['f1']:.4f} | "
                  f"Ans: {metrics['answerable_acc']:.4f} | "
                  f"Time: {epoch_time:.1f}s")

        final_metrics = evaluate_span(model, test_loader, device)

        all_results[name] = {
            'final_em': final_metrics['exact_match'],
            'final_f1': final_metrics['f1'],
            'best_f1': best_f1,
            'answerable_acc': final_metrics['answerable_acc'],
            'answerable_em': final_metrics['answerable_em'],
            'total_train_time': total_time,
            'parameters': model.count_parameters(),
        }

    # Summary
    print(f"\n{'='*80}")
    print("SQuAD 2.0 RESULTS")
    print(f"{'='*80}")
    print(f"{'Model':<25} {'EM':>8} {'F1':>8} {'Best F1':>8} {'Ans Acc':>8} {'Params':>10}")
    print("-" * 70)
    for name, r in all_results.items():
        print(f"{name:<25} {r['final_em']:>7.4f} {r['final_f1']:>7.4f} "
              f"{r['best_f1']:>7.4f} {r['answerable_acc']:>7.4f} "
              f"{r['parameters']:>10,}")

    # Verdict
    for name in ['PredCoding_Span', 'PredCoding_Span_0iter']:
        if name in all_results:
            f1 = all_results[name]['final_f1']
            if f1 >= 0.50:
                print(f"  {name}: STRONG SUCCESS (F1 >= 50%)")
            elif f1 >= 0.20:
                print(f"  {name}: PARTIAL SUCCESS (F1 >= 20%)")
            else:
                print(f"  {name}: WEAK (F1 < 20%) - pooling likely too lossy")

    save_path = os.path.join(os.path.dirname(__file__), "squad_results.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")

    return all_results


# ============================================================================
# Error Analysis
# ============================================================================

def run_error_analysis(
    max_seq_len: int = 256,
    vocab_size: int = 20000,
    embed_dim: int = 64,
    max_samples: int = 500,
):
    """
    Diagnostic error analysis for PredCoding QA.

    Categorizes errors by:
    - Question type (who/what/when/where/why/how)
    - Context length (short/medium/long)
    - Answer position (beginning/middle/end of context)
    """
    print("=" * 80)
    print("QA ERROR ANALYSIS")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- RACE Error Analysis ---
    print("\n--- RACE Error Analysis ---")
    from datasets import load_dataset
    raw_race = load_dataset("race", "middle", split="train")

    tokenizer = SimpleTokenizer(vocab_size=vocab_size)
    tokenizer.build_vocab(raw_race["article"])
    actual_vocab = len(tokenizer.word2idx)

    test_dataset = RACEDataset("test", tokenizer, max_len=max_seq_len,
                                max_samples=max_samples, difficulty="middle")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    # Load trained models (or train quickly)
    pc_model = PredCodingMultipleChoice(
        vocab_size=actual_vocab, embed_dim=embed_dim, num_layers=4,
        max_iterations=10, max_seq_len=max_seq_len,
    ).to(device)
    tf_model = TransformerMultipleChoice(
        vocab_size=actual_vocab, embed_dim=embed_dim, num_heads=4,
        num_layers=2, max_seq_len=max_seq_len,
    ).to(device)

    # Quick training (5 epochs)
    train_dataset = RACEDataset("train", tokenizer, max_len=max_seq_len,
                                 max_samples=2000, difficulty="middle")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    criterion = nn.CrossEntropyLoss()

    for name, model in [("PC", pc_model), ("TF", tf_model)]:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(5):
            train_mc_epoch(model, train_loader, optimizer, criterion, device)
        _, acc, _, _ = evaluate_mc(model, test_loader, criterion, device)
        print(f"  {name} accuracy: {acc:.4f}")

    # Categorize by question type
    question_types = {}
    raw_test = load_dataset("race", "middle", split="test")
    if max_samples and max_samples < len(raw_test):
        raw_test = raw_test.shuffle(seed=42).select(range(max_samples))

    for idx in range(len(raw_test)):
        q = raw_test[idx]["question"].lower()
        if q.startswith("who") or "who " in q:
            qtype = "who"
        elif q.startswith("what") or "what " in q:
            qtype = "what"
        elif q.startswith("when") or "when " in q:
            qtype = "when"
        elif q.startswith("where") or "where " in q:
            qtype = "where"
        elif q.startswith("why") or "why " in q:
            qtype = "why"
        elif q.startswith("how") or "how " in q:
            qtype = "how"
        else:
            qtype = "other"

        if qtype not in question_types:
            question_types[qtype] = {"total": 0, "pc_correct": 0, "tf_correct": 0}
        question_types[qtype]["total"] += 1

    # Get per-example predictions
    pc_model.eval()
    tf_model.eval()
    all_pc_preds = []
    all_tf_preds = []
    all_labels_list = []

    with torch.no_grad():
        for input_ids, labels in test_loader:
            input_ids = input_ids.to(device)
            pc_out, _ = pc_model(input_ids)
            tf_out, _ = tf_model(input_ids)
            all_pc_preds.extend(pc_out.argmax(dim=1).cpu().tolist())
            all_tf_preds.extend(tf_out.argmax(dim=1).cpu().tolist())
            all_labels_list.extend(labels.tolist())

    # Map predictions to question types
    idx = 0
    for q_idx in range(min(len(raw_test), len(all_labels_list))):
        q = raw_test[q_idx]["question"].lower()
        if q.startswith("who") or "who " in q:
            qtype = "who"
        elif q.startswith("what") or "what " in q:
            qtype = "what"
        elif q.startswith("when") or "when " in q:
            qtype = "when"
        elif q.startswith("where") or "where " in q:
            qtype = "where"
        elif q.startswith("why") or "why " in q:
            qtype = "why"
        elif q.startswith("how") or "how " in q:
            qtype = "how"
        else:
            qtype = "other"

        if q_idx < len(all_pc_preds):
            if all_pc_preds[q_idx] == all_labels_list[q_idx]:
                question_types[qtype]["pc_correct"] += 1
            if all_tf_preds[q_idx] == all_labels_list[q_idx]:
                question_types[qtype]["tf_correct"] += 1

    print(f"\n  {'Q-Type':<10} {'Count':>6} {'PC Acc':>8} {'TF Acc':>8} {'Delta':>8}")
    print("  " + "-" * 45)
    analysis_results = {"race_by_question_type": {}}
    for qtype, data in sorted(question_types.items(), key=lambda x: -x[1]["total"]):
        n = data["total"]
        pc_acc = data["pc_correct"] / max(n, 1)
        tf_acc = data["tf_correct"] / max(n, 1)
        delta = pc_acc - tf_acc
        print(f"  {qtype:<10} {n:>6} {pc_acc:>7.3f} {tf_acc:>7.3f} {delta:>+7.3f}")
        analysis_results["race_by_question_type"][qtype] = {
            "count": n, "pc_acc": pc_acc, "tf_acc": tf_acc
        }

    # Context length analysis
    print(f"\n  --- By Context Length ---")
    length_bins = {"short(<100)": (0, 100), "medium(100-200)": (100, 200), "long(>200)": (200, 10000)}
    length_data = {k: {"total": 0, "pc_correct": 0, "tf_correct": 0} for k in length_bins}

    for q_idx in range(min(len(raw_test), len(all_labels_list))):
        article_len = len(raw_test[q_idx]["article"].split())
        for bin_name, (lo, hi) in length_bins.items():
            if lo <= article_len < hi:
                length_data[bin_name]["total"] += 1
                if q_idx < len(all_pc_preds) and all_pc_preds[q_idx] == all_labels_list[q_idx]:
                    length_data[bin_name]["pc_correct"] += 1
                if q_idx < len(all_tf_preds) and all_tf_preds[q_idx] == all_labels_list[q_idx]:
                    length_data[bin_name]["tf_correct"] += 1
                break

    print(f"  {'Length':<15} {'Count':>6} {'PC Acc':>8} {'TF Acc':>8}")
    print("  " + "-" * 40)
    analysis_results["race_by_length"] = {}
    for bin_name, data in length_data.items():
        n = data["total"]
        if n > 0:
            pc_acc = data["pc_correct"] / n
            tf_acc = data["tf_correct"] / n
            print(f"  {bin_name:<15} {n:>6} {pc_acc:>7.3f} {tf_acc:>7.3f}")
            analysis_results["race_by_length"][bin_name] = {
                "count": n, "pc_acc": pc_acc, "tf_acc": tf_acc
            }

    # --- SQuAD Error Analysis ---
    print(f"\n--- SQuAD Error Analysis ---")
    raw_squad = load_dataset("squad_v2", split="validation")

    sq_tokenizer = SimpleTokenizer(vocab_size=vocab_size)
    raw_sq_train = load_dataset("squad_v2", split="train")
    sq_tokenizer.build_vocab(raw_sq_train["context"])
    sq_vocab = len(sq_tokenizer.word2idx)

    sq_test = SQuADDataset("validation", sq_tokenizer, max_len=max_seq_len,
                            max_samples=max_samples)
    sq_loader = DataLoader(sq_test, batch_size=16, shuffle=False, num_workers=0)

    pc_span = PredCodingSpanExtraction(
        vocab_size=sq_vocab, embed_dim=embed_dim, num_layers=4,
        max_iterations=10, max_seq_len=max_seq_len,
    ).to(device)
    tf_span = TransformerSpanExtraction(
        vocab_size=sq_vocab, embed_dim=embed_dim, num_heads=4,
        num_layers=2, max_seq_len=max_seq_len,
    ).to(device)

    # Quick train
    sq_train = SQuADDataset("train", sq_tokenizer, max_len=max_seq_len, max_samples=2000)
    sq_train_loader = DataLoader(sq_train, batch_size=16, shuffle=True, num_workers=0)

    for name, model in [("PC", pc_span), ("TF", tf_span)]:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(5):
            train_span_epoch(model, sq_train_loader, optimizer, device)
        metrics = evaluate_span(model, sq_loader, device)
        print(f"  {name}: EM={metrics['exact_match']:.4f}, F1={metrics['f1']:.4f}, "
              f"Ans={metrics['answerable_acc']:.4f}")

    # Categorize SQuAD errors by question type
    if max_samples and max_samples < len(raw_squad):
        raw_sq_test = raw_squad.shuffle(seed=42).select(range(max_samples))
    else:
        raw_sq_test = raw_squad

    sq_qtypes = {}
    pc_span.eval()
    tf_span.eval()

    all_pc_f1s = []
    all_tf_f1s = []
    q_type_list = []

    with torch.no_grad():
        batch_idx = 0
        for input_ids, start_pos, end_pos, is_answerable in sq_loader:
            input_ids = input_ids.to(device)
            pc_out, _ = pc_span(input_ids)
            tf_out, _ = tf_span(input_ids)

            pc_starts = pc_out['start_logits'].argmax(dim=1)
            pc_ends = torch.max(pc_out['end_logits'].argmax(dim=1), pc_starts)
            tf_starts = tf_out['start_logits'].argmax(dim=1)
            tf_ends = torch.max(tf_out['end_logits'].argmax(dim=1), tf_starts)

            for i in range(input_ids.size(0)):
                global_idx = batch_idx * sq_loader.batch_size + i
                if global_idx >= len(raw_sq_test):
                    break

                # F1 calculation
                gs, ge = start_pos[i].item(), end_pos[i].item()
                gold_set = set(range(gs, ge + 1))

                for preds, f1_list in [(pc_starts, all_pc_f1s), (tf_starts, all_tf_f1s)]:
                    if preds is pc_starts:
                        ps, pe = pc_starts[i].item(), pc_ends[i].item()
                    else:
                        ps, pe = tf_starts[i].item(), tf_ends[i].item()
                    pred_set = set(range(ps, pe + 1))
                    if len(pred_set) == 0 or len(gold_set) == 0:
                        f1 = 0.0
                    else:
                        overlap = len(pred_set & gold_set)
                        prec = overlap / len(pred_set)
                        rec = overlap / len(gold_set)
                        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                    f1_list.append(f1)

                q = raw_sq_test[global_idx]["question"].lower()
                if q.startswith("who") or "who " in q[:10]:
                    qtype = "who"
                elif q.startswith("what") or "what " in q[:10]:
                    qtype = "what"
                elif q.startswith("when") or "when " in q[:10]:
                    qtype = "when"
                elif q.startswith("where") or "where " in q[:10]:
                    qtype = "where"
                elif q.startswith("why") or "why " in q[:10]:
                    qtype = "why"
                elif q.startswith("how") or "how " in q[:10]:
                    qtype = "how"
                else:
                    qtype = "other"
                q_type_list.append(qtype)

            batch_idx += 1

    # Aggregate by question type
    sq_type_results = {}
    for i, qtype in enumerate(q_type_list):
        if qtype not in sq_type_results:
            sq_type_results[qtype] = {"pc_f1s": [], "tf_f1s": []}
        if i < len(all_pc_f1s):
            sq_type_results[qtype]["pc_f1s"].append(all_pc_f1s[i])
            sq_type_results[qtype]["tf_f1s"].append(all_tf_f1s[i])

    print(f"\n  {'Q-Type':<10} {'Count':>6} {'PC F1':>8} {'TF F1':>8} {'Delta':>8}")
    print("  " + "-" * 45)
    analysis_results["squad_by_question_type"] = {}
    for qtype, data in sorted(sq_type_results.items(), key=lambda x: -len(x[1]["pc_f1s"])):
        n = len(data["pc_f1s"])
        pc_f1 = np.mean(data["pc_f1s"]) if data["pc_f1s"] else 0
        tf_f1 = np.mean(data["tf_f1s"]) if data["tf_f1s"] else 0
        delta = pc_f1 - tf_f1
        print(f"  {qtype:<10} {n:>6} {pc_f1:>7.3f} {tf_f1:>7.3f} {delta:>+7.3f}")
        analysis_results["squad_by_question_type"][qtype] = {
            "count": n, "pc_f1": float(pc_f1), "tf_f1": float(tf_f1)
        }

    # Save
    save_path = os.path.join(os.path.dirname(__file__), "qa_error_analysis.json")
    with open(save_path, "w") as f:
        json.dump(analysis_results, f, indent=2, default=str)
    print(f"\nError analysis saved to {save_path}")

    return analysis_results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import sys
    task = sys.argv[1] if len(sys.argv) > 1 else "race"

    if task == "race":
        run_race_benchmark()
    elif task == "squad":
        run_squad_benchmark()
    elif task == "analysis":
        run_error_analysis()
    elif task == "all":
        run_race_benchmark()
        run_squad_benchmark()
        run_error_analysis()
    else:
        print(f"Usage: python train_qa.py [race|squad|analysis|all]")
