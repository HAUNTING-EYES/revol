"""
Training Script for WaveNetNeuro vs Transformer on IMDB Sentiment Analysis

Replaces the synthetic dataset with real IMDB movie reviews.
Reports: accuracy, training time, inference time, memory, adaptive steps.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import sys
import os
import json
import numpy as np
from collections import Counter

from wavenet_neuro import WaveNetNeuro, BaselineTransformer


# ---------------------------------------------------------------------------
# Simple word-level tokenizer (no dependency on HuggingFace tokenizers)
# ---------------------------------------------------------------------------

class SimpleTokenizer:
    """Word-level tokenizer with fixed vocabulary."""

    PAD = 0
    UNK = 1

    def __init__(self, vocab_size: int = 20000):
        self.vocab_size = vocab_size
        self.word2idx = {"<pad>": self.PAD, "<unk>": self.UNK}
        self.idx2word = {self.PAD: "<pad>", self.UNK: "<unk>"}

    def build_vocab(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(text.lower().split())
        for word, _ in counter.most_common(self.vocab_size - 2):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, text: str, max_len: int) -> list:
        tokens = text.lower().split()
        ids = [self.word2idx.get(w, self.UNK) for w in tokens[:max_len]]
        # Pad
        ids = ids + [self.PAD] * (max_len - len(ids))
        return ids


# ---------------------------------------------------------------------------
# IMDB Dataset wrapper
# ---------------------------------------------------------------------------

class IMDBDataset(Dataset):
    """IMDB sentiment dataset loaded from HuggingFace datasets."""

    def __init__(self, split: str, tokenizer: SimpleTokenizer, max_len: int = 256,
                 max_samples: int = None):
        from datasets import load_dataset

        print(f"  Loading IMDB {split} split...")
        ds = load_dataset("imdb", split=split)

        # IMPORTANT: shuffle before subsampling - IMDB is sorted by label
        ds = ds.shuffle(seed=42)

        if max_samples and max_samples < len(ds):
            ds = ds.select(range(max_samples))

        self.texts = ds["text"]
        self.labels = ds["label"]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ids = self.tokenizer.encode(self.texts[idx], self.max_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


# ---------------------------------------------------------------------------
# Efficiency tracking
# ---------------------------------------------------------------------------

class EfficiencyBenchmark:
    def __init__(self):
        self.reset()

    def reset(self):
        self.forward_times = []
        self.adaptive_steps = []
        self.memory_usage = []
        self.all_per_example_steps = []

    def record(self, forward_time: float, steps, memory: float, per_example_steps=None):
        self.forward_times.append(forward_time)
        self.adaptive_steps.append(steps)
        self.memory_usage.append(memory)
        if per_example_steps is not None:
            self.all_per_example_steps.append(per_example_steps)

    def get_stats(self):
        stats = {
            'avg_time_ms': np.mean(self.forward_times) * 1000,
            'std_time_ms': np.std(self.forward_times) * 1000,
            'avg_steps': np.mean(self.adaptive_steps),
            'avg_memory_mb': np.mean(self.memory_usage),
        }
        if self.all_per_example_steps:
            all_steps = np.concatenate(self.all_per_example_steps)
            stats['step_min'] = float(np.min(all_steps))
            stats['step_max'] = float(np.max(all_steps))
            stats['step_median'] = float(np.median(all_steps))
            stats['step_std'] = float(np.std(all_steps))
        return stats


# ---------------------------------------------------------------------------
# Train / Evaluate
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, criterion, device, benchmark=None,
                max_grad_norm=1.0, ponder_cost_weight=0.01):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        start_time = time.time()
        output, info = model(data)
        forward_time = time.time() - start_time

        if benchmark:
            memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            per_ex = info.get('per_example_steps')
            per_ex_np = per_ex.numpy() if per_ex is not None else None
            benchmark.record(forward_time, info.get('steps_taken', 0), memory, per_ex_np)

        loss = criterion(output, target)

        # Add ponder cost to incentivize early convergence
        ponder_cost = info.get('ponder_cost')
        if ponder_cost is not None and isinstance(ponder_cost, torch.Tensor):
            loss = loss + ponder_cost_weight * ponder_cost

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * target.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

    return total_loss / total, correct / total


def evaluate(model, dataloader, criterion, device, benchmark=None):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            start_time = time.time()
            output, info = model(data)
            forward_time = time.time() - start_time

            if benchmark:
                memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                per_ex = info.get('per_example_steps')
                per_ex_np = per_ex.numpy() if per_ex is not None else None
                benchmark.record(forward_time, info.get('steps_taken', 0), memory, per_ex_np)

            loss = criterion(output, target)
            total_loss += loss.item() * target.size(0)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Model builders with ~equal parameter counts
# ---------------------------------------------------------------------------

def build_models(vocab_size: int, device, target_params: int = 500_000):
    """Build WaveNetNeuro and Transformer with approximately equal parameter counts."""

    # WaveNetNeuro: embed_dim and field_channels control most params
    # We'll tune embed_dim to get close to target_params
    wavenet = WaveNetNeuro(
        vocab_size=vocab_size,
        embed_dim=64,
        field_channels=64,
        spatial_dim=16,
        num_classes=2,
        max_evolution_steps=30,
        convergence_threshold=0.1,
        dt=0.3,
    ).to(device)

    # Transformer: tune embed_dim and num_layers
    transformer = BaselineTransformer(
        vocab_size=vocab_size,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        num_classes=2,
        max_seq_len=4096,
    ).to(device)

    w_params = wavenet.count_parameters()
    t_params = transformer.count_parameters()
    print(f"  WaveNetNeuro params:  {w_params:,}")
    print(f"  Transformer params:   {t_params:,}")
    print(f"  Ratio (T/W):          {t_params/w_params:.2f}x")

    return wavenet, transformer


# ---------------------------------------------------------------------------
# Main IMDB benchmark
# ---------------------------------------------------------------------------

def run_imdb_benchmark(
    max_seq_len: int = 256,
    batch_size: int = 32,
    num_epochs: int = 5,
    learning_rate: float = 1e-3,
    vocab_size: int = 20000,
    max_train_samples: int = 5000,
    max_test_samples: int = 2000,
):
    """
    Train and compare WaveNetNeuro vs Transformer on IMDB.
    """
    print("=" * 80)
    print("IMDB Sentiment Analysis Benchmark")
    print("WaveNetNeuro vs Transformer Baseline")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Max seq len: {max_seq_len}, Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}, LR: {learning_rate}")
    print(f"Vocab size: {vocab_size}")
    print(f"Max train samples: {max_train_samples}, Max test samples: {max_test_samples}")

    # --- Load IMDB ---
    print("\n--- Loading IMDB Dataset ---")
    from datasets import load_dataset
    raw_train = load_dataset("imdb", split="train")
    raw_test = load_dataset("imdb", split="test")

    # Build tokenizer from training data
    print("  Building vocabulary...")
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)
    tokenizer.build_vocab(raw_train["text"])
    actual_vocab = len(tokenizer.word2idx)
    print(f"  Actual vocab size: {actual_vocab}")

    # Create datasets
    train_dataset = IMDBDataset("train", tokenizer, max_len=max_seq_len,
                                max_samples=max_train_samples)
    test_dataset = IMDBDataset("test", tokenizer, max_len=max_seq_len,
                               max_samples=max_test_samples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=False)

    print(f"  Train: {len(train_dataset)} samples, Test: {len(test_dataset)} samples")

    # --- Build models ---
    print("\n--- Building Models ---")
    wavenet, transformer = build_models(actual_vocab, device)

    criterion = nn.CrossEntropyLoss()

    results = {}

    # --- Train each model ---
    for model_name, model in [("WaveNetNeuro", wavenet), ("Transformer", transformer)]:
        print(f"\n{'='*80}")
        print(f"TRAINING {model_name.upper()}")
        print(f"{'='*80}")

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        benchmark = EfficiencyBenchmark()

        epoch_results = []
        total_train_time = 0

        for epoch in range(num_epochs):
            benchmark.reset()

            epoch_start = time.time()
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device, benchmark
            )
            epoch_time = time.time() - epoch_start
            total_train_time += epoch_time

            # Evaluate
            eval_benchmark = EfficiencyBenchmark()
            test_loss, test_acc = evaluate(model, test_loader, criterion, device, eval_benchmark)
            eval_stats = eval_benchmark.get_stats()

            scheduler.step()

            train_stats = benchmark.get_stats()
            epoch_data = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'epoch_time': epoch_time,
                'avg_steps': train_stats['avg_steps'],
                'inference_ms': eval_stats['avg_time_ms'],
            }
            epoch_results.append(epoch_data)

            step_info = ""
            if 'step_min' in train_stats:
                step_info = (f" | Steps: {train_stats['avg_steps']:.1f} "
                             f"(min={train_stats['step_min']:.0f}, "
                             f"max={train_stats['step_max']:.0f})")

            print(f"  Epoch {epoch+1}/{num_epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Train: {train_acc:.4f} | "
                  f"Test: {test_acc:.4f} | "
                  f"Time: {epoch_time:.1f}s"
                  f"{step_info}")

        # Final evaluation with detailed benchmark
        final_bench = EfficiencyBenchmark()
        final_loss, final_acc = evaluate(model, test_loader, criterion, device, final_bench)
        final_stats = final_bench.get_stats()

        results[model_name] = {
            'final_test_acc': final_acc,
            'final_test_loss': final_loss,
            'total_train_time': total_train_time,
            'inference_ms_per_batch': final_stats['avg_time_ms'],
            'parameters': model.count_parameters(),
            'epoch_results': epoch_results,
            'final_stats': final_stats,
        }

    # --- Summary ---
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")

    w = results["WaveNetNeuro"]
    t = results["Transformer"]

    print(f"\n{'Metric':<30} {'WaveNetNeuro':>15} {'Transformer':>15} {'Delta':>15}")
    print("-" * 75)
    print(f"{'Test Accuracy':<30} {w['final_test_acc']:>14.4f} {t['final_test_acc']:>14.4f} "
          f"{w['final_test_acc'] - t['final_test_acc']:>+14.4f}")
    print(f"{'Test Loss':<30} {w['final_test_loss']:>14.4f} {t['final_test_loss']:>14.4f} "
          f"{w['final_test_loss'] - t['final_test_loss']:>+14.4f}")
    print(f"{'Parameters':<30} {w['parameters']:>15,} {t['parameters']:>15,}")
    print(f"{'Total Train Time (s)':<30} {w['total_train_time']:>14.1f} {t['total_train_time']:>14.1f} "
          f"{'':>15}")
    print(f"{'Inference (ms/batch)':<30} {w['inference_ms_per_batch']:>14.2f} "
          f"{t['inference_ms_per_batch']:>14.2f} "
          f"{t['inference_ms_per_batch']/w['inference_ms_per_batch']:>14.2f}x")

    ws = w['final_stats']
    if 'step_min' in ws:
        print(f"\nWaveNetNeuro Adaptive Steps:")
        print(f"  Average: {ws['avg_steps']:.1f}")
        print(f"  Median:  {ws['step_median']:.0f}")
        print(f"  Range:   {ws['step_min']:.0f} - {ws['step_max']:.0f}")
        print(f"  Std:     {ws['step_std']:.1f}")

    # Verdict
    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")
    acc_diff = w['final_test_acc'] - t['final_test_acc']
    if acc_diff >= 0:
        print(f"  WaveNetNeuro accuracy MATCHES or BEATS transformer (+{acc_diff:.4f})")
    elif acc_diff >= -0.03:
        print(f"  WaveNetNeuro accuracy within 3% of transformer ({acc_diff:.4f})")
    else:
        print(f"  WaveNetNeuro accuracy FALLS SHORT ({acc_diff:.4f})")

    if w['final_test_acc'] < 0.80:
        print("  WARNING: WaveNetNeuro accuracy < 80% -- FAILURE CRITERION MET")

    # Save results
    save_path = os.path.join(os.path.dirname(__file__), "imdb_results.json")
    serializable = {}
    for name, r in results.items():
        serializable[name] = {k: v for k, v in r.items() if k != 'final_stats'}
        serializable[name]['final_stats'] = {
            k: float(v) for k, v in r['final_stats'].items()
        }
    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")

    return results


if __name__ == "__main__":
    results = run_imdb_benchmark(
        max_seq_len=256,
        batch_size=32,
        num_epochs=5,
        learning_rate=1e-3,
        vocab_size=20000,
        max_train_samples=10000,
        max_test_samples=5000,
    )
