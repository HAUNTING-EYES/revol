"""
Predictive Coding vs WaveNetNeuro vs Transformer on IMDB

Head-to-head comparison using the same training infrastructure.
The champion to beat: Fixed-10 WaveNetNeuro.
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
from predictive_coding import PredictiveCodingNetwork
from train_wavenet import (
    SimpleTokenizer,
    IMDBDataset,
    EfficiencyBenchmark,
    train_epoch,
    evaluate,
)


def run_pc_benchmark(
    max_seq_len: int = 256,
    batch_size: int = 32,
    num_epochs: int = 5,
    learning_rate: float = 1e-3,
    vocab_size: int = 20000,
    max_train_samples: int = 5000,
    max_test_samples: int = 2000,
    embed_dim: int = 64,
    field_channels: int = 64,
):
    print("=" * 80)
    print("PREDICTIVE CODING vs WAVENETNEURO vs TRANSFORMER")
    print("IMDB Sentiment Analysis Benchmark")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Config: seq_len={max_seq_len}, batch={batch_size}, epochs={num_epochs}")
    print(f"Samples: train={max_train_samples}, test={max_test_samples}")

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

    # --- Define models ---
    spatial_dim = max(16, int(np.ceil(np.sqrt(max_seq_len))))
    while spatial_dim * spatial_dim < max_seq_len:
        spatial_dim += 1

    models = OrderedDict()

    # 1. Transformer baseline
    models["Transformer"] = BaselineTransformer(
        vocab_size=actual_vocab,
        embed_dim=embed_dim,
        num_heads=4,
        num_layers=2,
        num_classes=2,
    ).to(device)

    # 2. WaveNetNeuro Fixed-10 (the champion)
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
    models["WaveNet_Fixed10"] = wavenet_f10

    # 3. WaveNetNeuro Adaptive
    models["WaveNet_Adaptive"] = WaveNetNeuro(
        vocab_size=actual_vocab,
        embed_dim=embed_dim,
        field_channels=field_channels,
        spatial_dim=spatial_dim,
        num_classes=2,
        max_evolution_steps=30,
        convergence_threshold=0.1,
        dt=0.3,
    ).to(device)

    # 4. Predictive Coding (full bidirectional)
    models["PredCoding"] = PredictiveCodingNetwork(
        vocab_size=actual_vocab,
        embed_dim=embed_dim,
        num_layers=4,
        num_classes=2,
        max_iterations=20,
        convergence_threshold=0.1,
        bidirectional=True,
    ).to(device)

    # 5. Predictive Coding (fixed 10 iterations, to match Fixed-10 WaveNet)
    models["PredCoding_Fixed10"] = PredictiveCodingNetwork(
        vocab_size=actual_vocab,
        embed_dim=embed_dim,
        num_layers=4,
        num_classes=2,
        max_iterations=10,
        convergence_threshold=1e-10,  # never converge early
        bidirectional=True,
    ).to(device)

    print(f"\n--- Model Parameter Counts ---")
    for name, model in models.items():
        print(f"  {name:<25} {model.count_parameters():>10,}")

    # --- Train each model ---
    all_results = {}
    criterion = nn.CrossEntropyLoss()

    for name, model in models.items():
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
    print("RESULTS SUMMARY")
    print(f"{'='*80}")

    print(f"\n{'Model':<25} {'Test Acc':>10} {'Best Acc':>10} "
          f"{'Params':>10} {'Time(s)':>10} {'Inf(ms)':>10} {'Steps':>8}")
    print("-" * 90)

    for name, r in all_results.items():
        print(f"{name:<25} {r['final_test_acc']:>9.4f} {r['best_test_acc']:>9.4f} "
              f"{r['parameters']:>10,} {r['total_train_time']:>9.1f} "
              f"{r['inference_ms']:>9.2f} {r['avg_steps']:>7.1f}")

    # --- Head-to-head comparisons ---
    print(f"\n--- Head-to-Head ---")
    champion = all_results.get('WaveNet_Fixed10', {})
    champion_acc = champion.get('final_test_acc', 0)

    for name in ['PredCoding', 'PredCoding_Fixed10']:
        if name in all_results:
            pc = all_results[name]
            delta = pc['final_test_acc'] - champion_acc
            speed = champion.get('inference_ms', 1) / max(pc['inference_ms'], 0.01)
            print(f"  {name} vs WaveNet_Fixed10:")
            print(f"    Accuracy delta: {delta:+.4f}")
            print(f"    Speed ratio:    {speed:.2f}x")

    t_acc = all_results.get('Transformer', {}).get('final_test_acc', 0)
    for name in ['PredCoding', 'PredCoding_Fixed10']:
        if name in all_results:
            delta = all_results[name]['final_test_acc'] - t_acc
            print(f"  {name} vs Transformer: {delta:+.4f}")

    # --- Verdict ---
    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")

    best_model = max(all_results.items(), key=lambda x: x[1]['final_test_acc'])
    print(f"  Best accuracy: {best_model[0]} ({best_model[1]['final_test_acc']:.4f})")

    fastest_model = min(all_results.items(), key=lambda x: x[1]['inference_ms'])
    print(f"  Fastest inference: {fastest_model[0]} ({fastest_model[1]['inference_ms']:.2f}ms)")

    pc_acc = all_results.get('PredCoding', {}).get('final_test_acc', 0)
    if pc_acc >= champion_acc:
        print(f"  PREDICTIVE CODING WINS or TIES vs Fixed-10 WaveNetNeuro!")
    elif pc_acc >= champion_acc - 0.02:
        print(f"  Predictive coding COMPETITIVE (within 2% of Fixed-10)")
    else:
        print(f"  Predictive coding LOSES to Fixed-10 WaveNetNeuro")
        print(f"  Recommendation: stick with WaveNetNeuro")

    # Save
    save_path = os.path.join(os.path.dirname(__file__), "pc_benchmark_results.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")

    return all_results


if __name__ == "__main__":
    run_pc_benchmark(
        max_seq_len=256,
        batch_size=32,
        num_epochs=5,
        learning_rate=1e-3,
        vocab_size=20000,
        max_train_samples=5000,
        max_test_samples=2000,
        embed_dim=64,
        field_channels=64,
    )
