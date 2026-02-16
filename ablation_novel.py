"""
Ablation studies for both novel architectures.

Hierarchical PredCoding:
  - depth_2, depth_4 (full), depth_6 layers
  - iterations_5, iterations_10 (full), iterations_0 (feedforward only)
  - bidirectional (full) vs bottom_up_only

MultiScale Parallel:
  - full (all 3 paths)
  - no_fast (drop attention)
  - no_medium (drop medium diffusion)
  - no_slow (drop slow diffusion)
  - medium_5 (fewer steps)
  - medium_20 (more steps)

Compared against WaveNet Fixed-10 and Transformer baselines.
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

from wavenet_neuro import WaveNetNeuro, BaselineTransformer, FixedStepEvolution
from hierarchical_predictive_coding import HierarchicalPredCodingNetwork, make_hier_pc_variant
from multiscale_parallel import make_multiscale_variant
from train_wavenet import (
    SimpleTokenizer, IMDBDataset, EfficiencyBenchmark, train_epoch, evaluate,
)


def run_ablation(
    max_seq_len: int = 256,
    batch_size: int = 32,
    num_epochs: int = 7,
    learning_rate: float = 1e-3,
    vocab_size: int = 20000,
    max_train_samples: int = 3000,
    max_test_samples: int = 1000,
    embed_dim: int = 64,
    field_channels: int = 64,
):
    print("=" * 80)
    print("ABLATION STUDY: Novel Architectures")
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

    train_dataset = IMDBDataset("train", tokenizer, max_len=max_seq_len,
                                max_samples=max_train_samples)
    test_dataset = IMDBDataset("test", tokenizer, max_len=max_seq_len,
                               max_samples=max_test_samples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  Vocab: {actual_vocab}, Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # --- Define variants ---
    variants = OrderedDict()

    # Baselines
    spatial_dim = max(16, int(np.ceil(np.sqrt(max_seq_len))))
    while spatial_dim * spatial_dim < max_seq_len:
        spatial_dim += 1

    variants["transformer"] = BaselineTransformer(
        vocab_size=actual_vocab, embed_dim=embed_dim, num_heads=4,
        num_layers=2, num_classes=2,
    ).to(device)

    wn = WaveNetNeuro(
        vocab_size=actual_vocab, embed_dim=embed_dim, field_channels=field_channels,
        spatial_dim=spatial_dim, num_classes=2, max_evolution_steps=30,
        convergence_threshold=0.1, dt=0.3,
    ).to(device)
    wn.evolution = FixedStepEvolution(wn.dynamics, fixed_steps=10)
    variants["wavenet_f10"] = wn

    # HierPC variants
    for vname in ['full', 'bottom_up_only', 'iterations_0', 'iterations_5', 'depth_2', 'depth_6']:
        variants[f"hierpc_{vname}"] = make_hier_pc_variant(
            vname, vocab_size=actual_vocab, embed_dim=embed_dim,
            num_layers=4, num_classes=2, max_iterations=10,
            convergence_threshold=0.1,
        ).to(device)

    # MultiScale variants
    for vname in ['full', 'no_fast', 'no_medium', 'no_slow', 'medium_5', 'medium_20']:
        variants[f"ms_{vname}"] = make_multiscale_variant(
            vname, vocab_size=actual_vocab, embed_dim=embed_dim, num_classes=2,
        ).to(device)

    print(f"\n--- Model Parameter Counts ---")
    for name, model in variants.items():
        print(f"  {name:<25} {model.count_parameters():>10,}")

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
            })

        final_bench = EfficiencyBenchmark()
        final_loss, final_acc = evaluate(model, test_loader, criterion, device, final_bench)
        final_stats = final_bench.get_stats()

        all_results[name] = {
            'final_test_acc': final_acc,
            'best_test_acc': best_test_acc,
            'total_train_time': total_time,
            'inference_ms': final_stats['avg_time_ms'],
            'parameters': model.count_parameters(),
            'avg_steps': final_stats['avg_steps'],
            'epochs': epoch_data,
        }

    # --- Analysis ---
    print(f"\n{'='*80}")
    print("ABLATION RESULTS")
    print(f"{'='*80}")

    print(f"\n{'Variant':<25} {'Test Acc':>10} {'Best':>8} {'Params':>10} {'Train(s)':>10} {'Inf(ms)':>10}")
    print("-" * 80)
    for name, r in all_results.items():
        print(f"{name:<25} {r['final_test_acc']:>9.4f} {r['best_test_acc']:>7.4f} "
              f"{r['parameters']:>10,} {r['total_train_time']:>9.1f} "
              f"{r['inference_ms']:>9.2f}")

    # HierPC analysis
    print(f"\n--- Hierarchical PredCoding ---")
    print(f"  Bidirectional vs Bottom-up:")
    for v in ['hierpc_full', 'hierpc_bottom_up_only']:
        if v in all_results:
            print(f"    {v}: {all_results[v]['final_test_acc']:.4f}")

    print(f"  Iteration count:")
    for v in ['hierpc_iterations_0', 'hierpc_iterations_5', 'hierpc_full']:
        if v in all_results:
            label = v.replace('hierpc_', '')
            print(f"    {label}: {all_results[v]['final_test_acc']:.4f}  inf={all_results[v]['inference_ms']:.2f}ms")

    print(f"  Layer depth:")
    for v in ['hierpc_depth_2', 'hierpc_full', 'hierpc_depth_6']:
        if v in all_results:
            label = v.replace('hierpc_', '')
            print(f"    {label}: {all_results[v]['final_test_acc']:.4f}  params={all_results[v]['parameters']:,}")

    # MultiScale analysis
    print(f"\n--- MultiScale Parallel ---")
    print(f"  Path ablation:")
    for v in ['ms_full', 'ms_no_fast', 'ms_no_medium', 'ms_no_slow']:
        if v in all_results:
            label = v.replace('ms_', '')
            print(f"    {label}: {all_results[v]['final_test_acc']:.4f}")

    print(f"  Step count:")
    for v in ['ms_medium_5', 'ms_full', 'ms_medium_20']:
        if v in all_results:
            label = v.replace('ms_', '')
            print(f"    {label}: {all_results[v]['final_test_acc']:.4f}  inf={all_results[v]['inference_ms']:.2f}ms")

    # Save
    save_path = os.path.join(os.path.dirname(__file__), "novel_ablation_results.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")

    return all_results


if __name__ == "__main__":
    run_ablation()
