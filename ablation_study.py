"""
Ablation Study for WaveNetNeuro

Tests what components actually matter by comparing:
  1. baseline_transformer - standard transformer
  2. wavenet_full - complete model (diffusion + reaction + adaptive)
  3. wavenet_diffusion_only - no reaction term
  4. wavenet_reaction_only - no diffusion term
  5. wavenet_fixed_steps - adaptive stopping disabled, fixed 10 steps

Uses IMDB sentiment analysis for evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import json
import os
import numpy as np
from collections import OrderedDict

from wavenet_neuro import (
    WaveNetNeuro,
    BaselineTransformer,
    make_wavenet_variant,
)
from train_wavenet import (
    SimpleTokenizer,
    IMDBDataset,
    EfficiencyBenchmark,
    train_epoch,
    evaluate,
)


def run_ablation_study(
    max_seq_len: int = 256,
    batch_size: int = 32,
    num_epochs: int = 5,
    learning_rate: float = 1e-3,
    vocab_size: int = 20000,
    max_train_samples: int = 3000,
    max_test_samples: int = 1000,
    embed_dim: int = 64,
    field_channels: int = 64,
):
    print("=" * 80)
    print("ABLATION STUDY: WaveNetNeuro Components")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Config: seq_len={max_seq_len}, batch={batch_size}, epochs={num_epochs}")

    # --- Load data ---
    print("\n--- Loading IMDB ---")
    from datasets import load_dataset
    raw_train = load_dataset("imdb", split="train")

    tokenizer = SimpleTokenizer(vocab_size=vocab_size)
    tokenizer.build_vocab(raw_train["text"])
    actual_vocab = len(tokenizer.word2idx)
    print(f"  Vocab: {actual_vocab}")

    train_dataset = IMDBDataset("train", tokenizer, max_len=max_seq_len,
                                max_samples=max_train_samples)
    test_dataset = IMDBDataset("test", tokenizer, max_len=max_seq_len,
                               max_samples=max_test_samples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # --- Define variants ---
    variants = OrderedDict()

    # 1. Transformer baseline
    variants["transformer"] = BaselineTransformer(
        vocab_size=actual_vocab,
        embed_dim=embed_dim,
        num_heads=4,
        num_layers=2,
        num_classes=2,
    ).to(device)

    # 2. Full WaveNetNeuro
    variants["wavenet_full"] = make_wavenet_variant(
        'full',
        vocab_size=actual_vocab,
        embed_dim=embed_dim,
        field_channels=field_channels,
        num_classes=2,
        max_evolution_steps=30,
    ).to(device)

    # 3. Diffusion only
    variants["wavenet_diffusion_only"] = make_wavenet_variant(
        'diffusion_only',
        vocab_size=actual_vocab,
        embed_dim=embed_dim,
        field_channels=field_channels,
        num_classes=2,
        max_evolution_steps=30,
    ).to(device)

    # 4. Reaction only
    variants["wavenet_reaction_only"] = make_wavenet_variant(
        'reaction_only',
        vocab_size=actual_vocab,
        embed_dim=embed_dim,
        field_channels=field_channels,
        num_classes=2,
        max_evolution_steps=30,
    ).to(device)

    # 5. Fixed steps (no adaptive)
    variants["wavenet_fixed_steps"] = make_wavenet_variant(
        'fixed_steps',
        vocab_size=actual_vocab,
        embed_dim=embed_dim,
        field_channels=field_channels,
        num_classes=2,
        max_evolution_steps=30,
        fixed_steps=10,
    ).to(device)

    print(f"\n--- Model Parameter Counts ---")
    for name, model in variants.items():
        print(f"  {name:<30} {model.count_parameters():>10,}")

    # --- Train each variant ---
    all_results = {}
    criterion = nn.CrossEntropyLoss()

    for name, model in variants.items():
        print(f"\n{'='*80}")
        print(f"TRAINING: {name}")
        print(f"{'='*80}")

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        best_test_acc = 0
        epoch_data = []
        total_time = 0

        for epoch in range(num_epochs):
            benchmark = EfficiencyBenchmark()

            epoch_start = time.time()
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device, benchmark
            )
            epoch_time = time.time() - epoch_start
            total_time += epoch_time

            eval_bench = EfficiencyBenchmark()
            test_loss, test_acc = evaluate(model, test_loader, criterion, device, eval_bench)
            scheduler.step()

            train_stats = benchmark.get_stats()
            best_test_acc = max(best_test_acc, test_acc)

            step_info = ""
            if 'step_min' in train_stats:
                step_info = (f" | Steps: {train_stats['avg_steps']:.1f}"
                             f" [{train_stats['step_min']:.0f}-{train_stats['step_max']:.0f}]")

            print(f"  Epoch {epoch+1}/{num_epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Train: {train_acc:.4f} | "
                  f"Test: {test_acc:.4f} | "
                  f"Time: {epoch_time:.1f}s"
                  f"{step_info}")

            epoch_data.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'time': epoch_time,
                'avg_steps': train_stats['avg_steps'],
            })

        # Final evaluation
        final_bench = EfficiencyBenchmark()
        final_loss, final_acc = evaluate(model, test_loader, criterion, device, final_bench)
        final_stats = final_bench.get_stats()

        all_results[name] = {
            'final_test_acc': final_acc,
            'best_test_acc': best_test_acc,
            'final_test_loss': final_loss,
            'total_train_time': total_time,
            'inference_ms': final_stats['avg_time_ms'],
            'parameters': model.count_parameters(),
            'avg_steps': final_stats['avg_steps'],
            'epochs': epoch_data,
        }

    # --- Summary ---
    print(f"\n{'='*80}")
    print("ABLATION RESULTS")
    print(f"{'='*80}")

    print(f"\n{'Variant':<30} {'Test Acc':>10} {'Best Acc':>10} "
          f"{'Params':>10} {'Time(s)':>10} {'Inf(ms)':>10} {'Steps':>8}")
    print("-" * 90)

    for name, r in all_results.items():
        print(f"{name:<30} {r['final_test_acc']:>9.4f} {r['best_test_acc']:>9.4f} "
              f"{r['parameters']:>10,} {r['total_train_time']:>9.1f} "
              f"{r['inference_ms']:>9.2f} {r['avg_steps']:>7.1f}")

    # Component contribution analysis
    print(f"\n--- Component Contribution Analysis ---")
    full_acc = all_results.get('wavenet_full', {}).get('final_test_acc', 0)
    for name in ['wavenet_diffusion_only', 'wavenet_reaction_only', 'wavenet_fixed_steps']:
        if name in all_results:
            delta = all_results[name]['final_test_acc'] - full_acc
            component = name.replace('wavenet_', '')
            print(f"  {component:<25} acc delta from full: {delta:+.4f}")

    # Transformer comparison
    t_acc = all_results.get('transformer', {}).get('final_test_acc', 0)
    print(f"\n  Full WaveNetNeuro vs Transformer: {full_acc - t_acc:+.4f}")

    # Adaptive computation analysis
    print(f"\n--- Adaptive Computation ---")
    for name in ['wavenet_full', 'wavenet_diffusion_only', 'wavenet_reaction_only']:
        if name in all_results:
            steps = all_results[name]['avg_steps']
            print(f"  {name:<30} avg steps: {steps:.1f} / 30")

    fixed = all_results.get('wavenet_fixed_steps', {}).get('avg_steps', 0)
    adaptive = all_results.get('wavenet_full', {}).get('avg_steps', 0)
    if fixed > 0:
        reduction = (fixed - adaptive) / fixed * 100
        print(f"\n  Adaptive vs Fixed step reduction: {reduction:.1f}%")

    # Verdict
    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")

    if full_acc > all_results.get('wavenet_diffusion_only', {}).get('final_test_acc', 0) and \
       full_acc > all_results.get('wavenet_reaction_only', {}).get('final_test_acc', 0):
        print("  BOTH diffusion and reaction contribute to performance")
    elif full_acc <= all_results.get('wavenet_reaction_only', {}).get('final_test_acc', 0):
        print("  Diffusion may NOT be contributing; reaction alone is sufficient")
    elif full_acc <= all_results.get('wavenet_diffusion_only', {}).get('final_test_acc', 0):
        print("  Reaction may NOT be contributing; diffusion alone is sufficient")

    if full_acc > all_results.get('wavenet_fixed_steps', {}).get('final_test_acc', 0):
        print("  Adaptive computation HELPS accuracy")
    else:
        print("  Adaptive computation does NOT improve accuracy over fixed steps")

    # Save
    save_path = os.path.join(os.path.dirname(__file__), "ablation_results.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")

    return all_results


if __name__ == "__main__":
    run_ablation_study(
        max_seq_len=256,
        batch_size=32,
        num_epochs=5,
        learning_rate=1e-3,
        vocab_size=20000,
        max_train_samples=3000,
        max_test_samples=1000,
        embed_dim=64,
        field_channels=64,
    )
