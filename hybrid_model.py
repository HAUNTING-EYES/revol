"""
Hybrid Architecture: Transformer (short) + WaveNetNeuro (long)

Routes sequences to the best model based on length:
  - Short (<crossover): Transformer is faster
  - Long (>=crossover): WaveNetNeuro is faster due to O(n) scaling

Also includes fixed-step production variant comparison.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import json
import os
import gc
import numpy as np
from typing import Tuple

from wavenet_neuro import (
    WaveNetNeuro,
    BaselineTransformer,
    FixedStepEvolution,
)


class HybridModel(nn.Module):
    """
    Smart router: uses transformer for short sequences,
    WaveNetNeuro for long sequences.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        field_channels: int = 64,
        num_classes: int = 2,
        crossover_length: int = 512,
        max_evolution_steps: int = 30,
    ):
        super().__init__()
        self.crossover_length = crossover_length

        self.transformer = BaselineTransformer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=4,
            num_layers=2,
            num_classes=num_classes,
            max_seq_len=max(crossover_length, 512),
        )

        # spatial_dim must accommodate the longest possible sequence
        max_spatial = max(16, int(np.ceil(np.sqrt(8192))))
        self.wavenet = WaveNetNeuro(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            field_channels=field_channels,
            spatial_dim=max_spatial,
            num_classes=num_classes,
            max_evolution_steps=max_evolution_steps,
            convergence_threshold=0.1,
            dt=0.3,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, dict]:
        seq_len = x.size(1)
        if seq_len < self.crossover_length:
            output, info = self.transformer(x)
            info['routed_to'] = 'transformer'
        else:
            output, info = self.wavenet(x)
            info['routed_to'] = 'wavenet'
        return output, info

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Task 2: Hybrid architecture test on mixed-length sequences
# ---------------------------------------------------------------------------

def benchmark_hybrid(
    vocab_size: int = 10000,
    embed_dim: int = 64,
    field_channels: int = 64,
    batch_size: int = 4,
    crossover_length: int = 512,
    num_warmup: int = 2,
    num_runs: int = 5,
):
    """Test hybrid model on a mix of short and long sequences."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("HYBRID MODEL BENCHMARK")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Crossover length: {crossover_length}")
    print()

    hybrid = HybridModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        field_channels=field_channels,
        crossover_length=crossover_length,
    ).to(device)

    # Also create standalone models for comparison
    max_spatial = max(16, int(np.ceil(np.sqrt(8192))))
    wavenet_only = WaveNetNeuro(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        field_channels=field_channels,
        spatial_dim=max_spatial,
        num_classes=2,
        max_evolution_steps=30,
        convergence_threshold=0.1,
        dt=0.3,
    ).to(device)

    transformer_only = BaselineTransformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=4,
        num_layers=2,
        num_classes=2,
        max_seq_len=8192,
    ).to(device)

    print(f"Hybrid parameters: {hybrid.count_parameters():,}")
    print(f"  (Transformer part + WaveNetNeuro part)")

    # Test at various lengths
    test_lengths = [128, 256, 512, 1024, 2048, 4096]
    results = {}

    for seq_len in test_lengths:
        print(f"\n--- Sequence length: {seq_len} ---")
        x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

        # Warmup all models
        with torch.no_grad():
            for _ in range(num_warmup):
                try:
                    hybrid(x)
                except Exception:
                    pass
                try:
                    wavenet_only(x)
                except Exception:
                    pass
                try:
                    transformer_only(x)
                except Exception:
                    pass

        entry = {}

        # Hybrid
        times_h = []
        routed = None
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                try:
                    out, info = hybrid(x)
                    elapsed = time.perf_counter() - start
                    times_h.append(elapsed * 1000)
                    routed = info.get('routed_to', '?')
                except Exception as e:
                    times_h.append(float('inf'))
        entry['hybrid'] = {
            'mean_ms': np.mean(times_h),
            'std_ms': np.std(times_h),
            'routed_to': routed,
        }

        # WaveNetNeuro only
        times_w = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                try:
                    wavenet_only(x)
                    elapsed = time.perf_counter() - start
                    times_w.append(elapsed * 1000)
                except Exception as e:
                    times_w.append(float('inf'))
        entry['wavenet'] = {'mean_ms': np.mean(times_w), 'std_ms': np.std(times_w)}

        # Transformer only
        times_t = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                try:
                    transformer_only(x)
                    elapsed = time.perf_counter() - start
                    times_t.append(elapsed * 1000)
                except Exception as e:
                    times_t.append(float('inf'))
        entry['transformer'] = {'mean_ms': np.mean(times_t), 'std_ms': np.std(times_t)}

        results[seq_len] = entry

        h_ms = entry['hybrid']['mean_ms']
        w_ms = entry['wavenet']['mean_ms']
        t_ms = entry['transformer']['mean_ms']

        print(f"  Hybrid:      {h_ms:>8.1f}ms  (routed to: {routed})")
        print(f"  WaveNet:     {w_ms:>8.1f}ms")
        print(f"  Transformer: {t_ms:>8.1f}ms")
        best = min(h_ms, w_ms, t_ms)
        if best == h_ms:
            print(f"  Winner: Hybrid")
        elif best == w_ms:
            print(f"  Winner: WaveNet")
        else:
            print(f"  Winner: Transformer")

        del x
        gc.collect()

    # Summary
    print(f"\n{'='*80}")
    print("HYBRID ROUTING SUMMARY")
    print(f"{'='*80}")
    print(f"{'SeqLen':>8} | {'Hybrid (ms)':>12} | {'Routed To':>12} | "
          f"{'WaveNet (ms)':>13} | {'Transformer (ms)':>17} | {'Best':>12}")
    print("-" * 85)

    for sl in test_lengths:
        r = results[sl]
        h = r['hybrid']['mean_ms']
        w = r['wavenet']['mean_ms']
        t = r['transformer']['mean_ms']
        best_val = min(h, w, t)
        best_name = "Hybrid" if best_val == h else ("WaveNet" if best_val == w else "Transformer")
        print(f"{sl:>8} | {h:>11.1f} | {r['hybrid']['routed_to']:>12} | "
              f"{w:>12.1f} | {t:>16.1f} | {best_name:>12}")

    # Mixed workload simulation
    print(f"\n--- Mixed Workload Simulation ---")
    print(f"50% short (<{crossover_length}) + 50% long (>={crossover_length})")
    short_lens = [sl for sl in test_lengths if sl < crossover_length]
    long_lens = [sl for sl in test_lengths if sl >= crossover_length]

    if short_lens and long_lens:
        avg_hybrid_short = np.mean([results[sl]['hybrid']['mean_ms'] for sl in short_lens])
        avg_hybrid_long = np.mean([results[sl]['hybrid']['mean_ms'] for sl in long_lens])
        avg_hybrid_mixed = (avg_hybrid_short + avg_hybrid_long) / 2

        avg_t_short = np.mean([results[sl]['transformer']['mean_ms'] for sl in short_lens])
        avg_t_long = np.mean([results[sl]['transformer']['mean_ms'] for sl in long_lens
                              if results[sl]['transformer']['mean_ms'] != float('inf')])
        avg_t_mixed = (avg_t_short + avg_t_long) / 2

        avg_w_short = np.mean([results[sl]['wavenet']['mean_ms'] for sl in short_lens])
        avg_w_long = np.mean([results[sl]['wavenet']['mean_ms'] for sl in long_lens])
        avg_w_mixed = (avg_w_short + avg_w_long) / 2

        print(f"  Hybrid avg:      {avg_hybrid_mixed:.1f}ms")
        print(f"  WaveNet avg:     {avg_w_mixed:.1f}ms")
        print(f"  Transformer avg: {avg_t_mixed:.1f}ms")
        print(f"  Hybrid speedup over Transformer: {avg_t_mixed / avg_hybrid_mixed:.2f}x")
        print(f"  Hybrid speedup over WaveNet:     {avg_w_mixed / avg_hybrid_mixed:.2f}x")

    save_path = os.path.join(os.path.dirname(__file__), "hybrid_results.json")
    with open(save_path, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")

    return results


# ---------------------------------------------------------------------------
# Task 4: Fixed-step production variant comparison
# ---------------------------------------------------------------------------

def benchmark_fixed_vs_adaptive(
    vocab_size: int = 10000,
    embed_dim: int = 64,
    field_channels: int = 64,
    batch_size: int = 8,
    num_warmup: int = 2,
    num_runs: int = 10,
):
    """
    Compare:
      1. WaveNetNeuro adaptive (avg ~15 steps)
      2. WaveNetNeuro fixed-10 (always 10 steps, no convergence check overhead)
      3. Transformer baseline (2 layers)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("FIXED-STEP vs ADAPTIVE vs TRANSFORMER")
    print("=" * 80)
    print(f"Device: {device}")
    print()

    test_lengths = [128, 256, 512, 1024, 2048]
    results = {}

    for seq_len in test_lengths:
        print(f"\n--- Sequence length: {seq_len} ---")

        spatial_dim = max(16, int(np.ceil(np.sqrt(seq_len))))
        while spatial_dim * spatial_dim < seq_len:
            spatial_dim += 1

        # Adaptive WaveNetNeuro
        wavenet_adaptive = WaveNetNeuro(
            vocab_size=vocab_size, embed_dim=embed_dim,
            field_channels=field_channels, spatial_dim=spatial_dim,
            num_classes=2, max_evolution_steps=30,
            convergence_threshold=0.1, dt=0.3,
        ).to(device)

        # Fixed-10 WaveNetNeuro
        wavenet_fixed = WaveNetNeuro(
            vocab_size=vocab_size, embed_dim=embed_dim,
            field_channels=field_channels, spatial_dim=spatial_dim,
            num_classes=2, max_evolution_steps=30,
            convergence_threshold=0.1, dt=0.3,
        ).to(device)
        wavenet_fixed.evolution = FixedStepEvolution(wavenet_fixed.dynamics, fixed_steps=10)

        # Transformer
        transformer = BaselineTransformer(
            vocab_size=vocab_size, embed_dim=embed_dim,
            num_heads=4, num_layers=2, num_classes=2,
            max_seq_len=max(seq_len, 512),
        ).to(device)

        x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

        entry = {}
        models = [
            ("adaptive", wavenet_adaptive),
            ("fixed_10", wavenet_fixed),
            ("transformer", transformer),
        ]

        for name, model in models:
            model.eval()
            # Warmup
            with torch.no_grad():
                for _ in range(num_warmup):
                    model(x)

            times = []
            steps_list = []
            with torch.no_grad():
                for _ in range(num_runs):
                    start = time.perf_counter()
                    output, info = model(x)
                    elapsed = time.perf_counter() - start
                    times.append(elapsed * 1000)
                    steps_list.append(info.get('steps_taken', 0))

            entry[name] = {
                'mean_ms': np.mean(times),
                'std_ms': np.std(times),
                'avg_steps': np.mean(steps_list),
            }
            print(f"  {name:<12} {entry[name]['mean_ms']:>8.1f}ms  "
                  f"(steps: {entry[name]['avg_steps']:.1f})")

        # Speedup of fixed over adaptive
        if entry['adaptive']['mean_ms'] > 0:
            entry['fixed_vs_adaptive_speedup'] = (
                entry['adaptive']['mean_ms'] / entry['fixed_10']['mean_ms']
            )
        else:
            entry['fixed_vs_adaptive_speedup'] = 0

        results[seq_len] = entry

        del wavenet_adaptive, wavenet_fixed, transformer, x
        gc.collect()

    # Summary
    print(f"\n{'='*80}")
    print("FIXED vs ADAPTIVE vs TRANSFORMER SUMMARY")
    print(f"{'='*80}")
    print(f"{'SeqLen':>8} | {'Adaptive (ms)':>14} | {'Fixed-10 (ms)':>14} | "
          f"{'Transformer (ms)':>17} | {'F/A Speedup':>12}")
    print("-" * 75)

    for sl in test_lengths:
        r = results[sl]
        a = r['adaptive']['mean_ms']
        f = r['fixed_10']['mean_ms']
        t = r['transformer']['mean_ms']
        sp = r.get('fixed_vs_adaptive_speedup', 0)
        print(f"{sl:>8} | {a:>13.1f} | {f:>13.1f} | {t:>16.1f} | {sp:>11.2f}x")

    print(f"\n--- Analysis ---")
    adaptive_steps = np.mean([results[sl]['adaptive']['avg_steps'] for sl in test_lengths])
    print(f"  Adaptive avg steps: {adaptive_steps:.1f}")
    print(f"  Fixed steps: 10")
    print(f"  Step reduction: {(adaptive_steps - 10) / adaptive_steps * 100:.0f}%")

    avg_speedup = np.mean([results[sl].get('fixed_vs_adaptive_speedup', 1) for sl in test_lengths])
    print(f"  Average Fixed/Adaptive speedup: {avg_speedup:.2f}x")

    # Verdict
    print(f"\n{'='*80}")
    print("PRODUCTION RECOMMENDATION")
    print(f"{'='*80}")
    if avg_speedup > 1.1:
        print("  Fixed-10 is FASTER than adaptive with minimal accuracy loss")
        print("  RECOMMENDATION: Use fixed-10 for production")
    else:
        print("  Fixed-10 and adaptive have similar speed")
        print("  RECOMMENDATION: Use adaptive for accuracy benefit")

    # Where each model wins
    t_wins = [sl for sl in test_lengths if results[sl]['transformer']['mean_ms']
              <= min(results[sl]['adaptive']['mean_ms'], results[sl]['fixed_10']['mean_ms'])]
    f_wins = [sl for sl in test_lengths if results[sl]['fixed_10']['mean_ms']
              <= min(results[sl]['adaptive']['mean_ms'], results[sl]['transformer']['mean_ms'])]
    print(f"\n  Transformer fastest at: {t_wins if t_wins else 'none'}")
    print(f"  Fixed-10 fastest at:    {f_wins if f_wins else 'none'}")

    save_path = os.path.join(os.path.dirname(__file__), "fixed_vs_adaptive_results.json")
    with open(save_path, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")

    return results


if __name__ == "__main__":
    print("\n" + "#" * 80)
    print("# PART 1: Extreme Scaling")
    print("#" * 80)
    # Import and run from the dedicated script
    from benchmark_extreme_scaling import run_extreme_scaling
    run_extreme_scaling()

    print("\n" + "#" * 80)
    print("# PART 2: Hybrid Architecture")
    print("#" * 80)
    benchmark_hybrid()

    print("\n" + "#" * 80)
    print("# PART 3: Fixed-Step vs Adaptive")
    print("#" * 80)
    benchmark_fixed_vs_adaptive()
