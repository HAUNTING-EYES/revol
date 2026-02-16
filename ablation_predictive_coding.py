"""
Ablation Study for Predictive Coding

Tests what components actually matter:
  1. full - bidirectional (errors up + predictions down)
  2. bottom_up_only - no top-down predictions
  3. iterations_5 - fixed 5 iterations
  4. iterations_10 - fixed 10 iterations
  5. depth_2 - 2 layers
  6. depth_6 - 6 layers
  7. depth_8 - 8 layers

Questions:
- Does bidirectionality matter?
- How many iterations are needed?
- How deep should the hierarchy be?

Compared against WaveNet_Fixed10 and Transformer baselines.
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
from predictive_coding import PredictiveCodingNetwork, make_pc_variant
from train_wavenet import (
    SimpleTokenizer,
    IMDBDataset,
    EfficiencyBenchmark,
    train_epoch,
    evaluate,
)


def run_pc_ablation(
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
    print("ABLATION STUDY: Predictive Coding Components")
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

    # Baselines
    spatial_dim = max(16, int(np.ceil(np.sqrt(max_seq_len))))
    while spatial_dim * spatial_dim < max_seq_len:
        spatial_dim += 1

    variants["transformer"] = BaselineTransformer(
        vocab_size=actual_vocab,
        embed_dim=embed_dim,
        num_heads=4,
        num_layers=2,
        num_classes=2,
    ).to(device)

    wavenet_f10 = WaveNetNeuro(
        vocab_size=actual_vocab,
        embed_dim=embed_dim,
        field_channels=field_channels,
        spatial_dim=spatial_dim,
        num_classes=2,
        max_evolution_steps=30,
        convergence_threshold=0.1,
        dt=0.3,
    ).to(device)
    wavenet_f10.evolution = FixedStepEvolution(wavenet_f10.dynamics, fixed_steps=10)
    variants["wavenet_fixed10"] = wavenet_f10

    # PC variants
    pc_variant_names = [
        'full', 'bottom_up_only',
        'iterations_5', 'iterations_10',
        'depth_2', 'depth_6', 'depth_8',
    ]

    for vname in pc_variant_names:
        variants[f"pc_{vname}"] = make_pc_variant(
            vname,
            vocab_size=actual_vocab,
            embed_dim=embed_dim,
            num_layers=4,
            num_classes=2,
            max_iterations=20,
            convergence_threshold=0.1,
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

    print(f"\n{'Variant':<25} {'Test Acc':>10} {'Best Acc':>10} "
          f"{'Params':>10} {'Time(s)':>10} {'Inf(ms)':>10} {'Steps':>8}")
    print("-" * 90)

    for name, r in all_results.items():
        print(f"{name:<25} {r['final_test_acc']:>9.4f} {r['best_test_acc']:>9.4f} "
              f"{r['parameters']:>10,} {r['total_train_time']:>9.1f} "
              f"{r['inference_ms']:>9.2f} {r['avg_steps']:>7.1f}")

    # --- Bidirectionality analysis ---
    print(f"\n--- Bidirectionality ---")
    full_acc = all_results.get('pc_full', {}).get('final_test_acc', 0)
    bu_acc = all_results.get('pc_bottom_up_only', {}).get('final_test_acc', 0)
    print(f"  Full (bidirectional): {full_acc:.4f}")
    print(f"  Bottom-up only:      {bu_acc:.4f}")
    print(f"  Delta:               {full_acc - bu_acc:+.4f}")
    if full_acc > bu_acc + 0.005:
        print(f"  -> Bidirectionality HELPS")
    elif bu_acc > full_acc + 0.005:
        print(f"  -> Top-down hurts! Bottom-up sufficient")
    else:
        print(f"  -> No significant difference")

    # --- Iteration count analysis ---
    print(f"\n--- Iteration Count ---")
    for vname in ['iterations_5', 'iterations_10', 'full']:
        key = f"pc_{vname}"
        if key in all_results:
            acc = all_results[key]['final_test_acc']
            steps = all_results[key]['avg_steps']
            ms = all_results[key]['inference_ms']
            iters = vname if vname != 'full' else 'adaptive_20'
            print(f"  {iters:<15} acc={acc:.4f}  steps={steps:.1f}  inf={ms:.2f}ms")

    # --- Depth analysis ---
    print(f"\n--- Layer Depth ---")
    for vname in ['depth_2', 'full', 'depth_6', 'depth_8']:
        key = f"pc_{vname}"
        if key in all_results:
            acc = all_results[key]['final_test_acc']
            params = all_results[key]['parameters']
            depth = vname if vname != 'full' else 'depth_4'
            print(f"  {depth:<10} acc={acc:.4f}  params={params:,}")

    # --- vs Baselines ---
    print(f"\n--- vs Baselines ---")
    t_acc = all_results.get('transformer', {}).get('final_test_acc', 0)
    w_acc = all_results.get('wavenet_fixed10', {}).get('final_test_acc', 0)
    print(f"  Transformer:     {t_acc:.4f}")
    print(f"  WaveNet Fixed10: {w_acc:.4f}")
    print(f"  PC Full:         {full_acc:.4f}")
    print(f"  PC vs Transformer: {full_acc - t_acc:+.4f}")
    print(f"  PC vs WaveNet:     {full_acc - w_acc:+.4f}")

    # --- Verdict ---
    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")

    best_pc = max(
        [(n, r) for n, r in all_results.items() if n.startswith('pc_')],
        key=lambda x: x[1]['final_test_acc'],
        default=(None, {'final_test_acc': 0})
    )
    print(f"  Best PC variant: {best_pc[0]} ({best_pc[1]['final_test_acc']:.4f})")

    if best_pc[1]['final_test_acc'] >= w_acc:
        print(f"  Predictive Coding BEATS or TIES WaveNet Fixed-10!")
    elif best_pc[1]['final_test_acc'] >= w_acc - 0.02:
        print(f"  Predictive Coding COMPETITIVE with WaveNet Fixed-10")
    else:
        print(f"  Predictive Coding LOSES to WaveNet Fixed-10")

    # Save
    save_path = os.path.join(os.path.dirname(__file__), "pc_ablation_results.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")

    return all_results


if __name__ == "__main__":
    run_pc_ablation(
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
