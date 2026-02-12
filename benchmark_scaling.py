"""
Sequence Length Scaling Benchmark

Tests WaveNetNeuro vs Transformer at increasing sequence lengths.
Hypothesis: WaveNetNeuro O(n) should beat Transformer O(n^2) at long sequences.

Measures:
  - Forward pass time (inference only, no grad)
  - Peak memory usage
  - Time scaling behavior
"""

import torch
import time
import json
import os
import numpy as np
from wavenet_neuro import WaveNetNeuro, BaselineTransformer


def measure_forward_pass(model, x, device, num_warmup=3, num_runs=10):
    """
    Measure average forward pass time.
    Includes warmup runs to stabilize timings.
    """
    model.eval()
    x = x.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            model(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            output, info = model(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000,
    }


def measure_memory(model, x, device):
    """Measure peak memory during forward pass (GPU only)."""
    if not torch.cuda.is_available():
        return 0.0

    model.eval()
    x = x.to(device)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    with torch.no_grad():
        model(x)

    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    return peak_mb


def run_scaling_benchmark(
    sequence_lengths=None,
    batch_size=8,
    vocab_size=10000,
    embed_dim=64,
    field_channels=64,
    num_warmup=3,
    num_runs=10,
):
    """
    Benchmark forward pass time at various sequence lengths.
    """
    if sequence_lengths is None:
        sequence_lengths = [64, 128, 256, 512, 1024, 2048]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("Sequence Length Scaling Benchmark")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence lengths: {sequence_lengths}")
    print(f"Warmup runs: {num_warmup}, Measurement runs: {num_runs}")
    print()

    results = {}

    for seq_len in sequence_lengths:
        print(f"\n--- Sequence Length: {seq_len} ---")

        # spatial_dim must be >= sqrt(seq_len)
        spatial_dim = max(16, int(np.ceil(np.sqrt(seq_len))))
        # Make spatial_dim a power of 2 or a clean number
        # spatial_dim^2 must be >= seq_len
        while spatial_dim * spatial_dim < seq_len:
            spatial_dim += 1

        wavenet = WaveNetNeuro(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            field_channels=field_channels,
            spatial_dim=spatial_dim,
            num_classes=2,
            max_evolution_steps=30,
            convergence_threshold=0.1,
            dt=0.3,
        ).to(device)

        transformer = BaselineTransformer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=4,
            num_layers=2,
            num_classes=2,
            max_seq_len=max(seq_len, 512),
        ).to(device)

        # Random input
        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Measure
        try:
            w_time = measure_forward_pass(wavenet, x, device, num_warmup, num_runs)
        except RuntimeError as e:
            print(f"  WaveNetNeuro FAILED at seq_len={seq_len}: {e}")
            w_time = {'mean_ms': float('inf'), 'std_ms': 0, 'min_ms': float('inf'), 'max_ms': float('inf')}

        try:
            t_time = measure_forward_pass(transformer, x, device, num_warmup, num_runs)
        except RuntimeError as e:
            print(f"  Transformer FAILED at seq_len={seq_len}: {e}")
            t_time = {'mean_ms': float('inf'), 'std_ms': 0, 'min_ms': float('inf'), 'max_ms': float('inf')}

        w_mem = measure_memory(wavenet, x, device)
        t_mem = measure_memory(transformer, x, device)

        speedup = t_time['mean_ms'] / w_time['mean_ms'] if w_time['mean_ms'] > 0 else 0

        results[seq_len] = {
            'wavenet': {**w_time, 'memory_mb': w_mem, 'params': wavenet.count_parameters()},
            'transformer': {**t_time, 'memory_mb': t_mem, 'params': transformer.count_parameters()},
            'speedup': speedup,
        }

        print(f"  WaveNetNeuro:  {w_time['mean_ms']:>8.2f}ms (+/- {w_time['std_ms']:.2f}ms)")
        print(f"  Transformer:   {t_time['mean_ms']:>8.2f}ms (+/- {t_time['std_ms']:.2f}ms)")
        print(f"  Speedup (T/W): {speedup:.2f}x")

        # Free memory
        del wavenet, transformer, x
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Summary Table ---
    print(f"\n{'='*80}")
    print("SCALING SUMMARY")
    print(f"{'='*80}")
    print(f"{'SeqLen':>8} | {'WaveNet (ms)':>14} | {'Transformer (ms)':>17} | {'Speedup':>8}")
    print("-" * 55)
    for sl in sequence_lengths:
        r = results[sl]
        marker = " <--" if r['speedup'] > 1.0 else ""
        print(f"{sl:>8} | {r['wavenet']['mean_ms']:>14.2f} | "
              f"{r['transformer']['mean_ms']:>17.2f} | {r['speedup']:>7.2f}x{marker}")

    # Theoretical analysis
    print(f"\nScaling Analysis:")
    if len(sequence_lengths) >= 2:
        sl_1 = sequence_lengths[0]
        sl_last = sequence_lengths[-1]
        r1 = results[sl_1]
        rl = results[sl_last]

        len_ratio = sl_last / sl_1
        w_time_ratio = rl['wavenet']['mean_ms'] / r1['wavenet']['mean_ms'] if r1['wavenet']['mean_ms'] > 0 else 0
        t_time_ratio = rl['transformer']['mean_ms'] / r1['transformer']['mean_ms'] if r1['transformer']['mean_ms'] > 0 else 0

        print(f"  Length increase: {len_ratio:.0f}x ({sl_1} -> {sl_last})")
        print(f"  WaveNetNeuro time increase: {w_time_ratio:.1f}x (O(n) predicts {len_ratio:.0f}x)")
        print(f"  Transformer time increase:  {t_time_ratio:.1f}x (O(n^2) predicts {len_ratio**2:.0f}x)")

    # Verdict
    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")
    wins = sum(1 for r in results.values() if r['speedup'] > 1.0)
    total = len(results)
    print(f"  WaveNetNeuro faster at {wins}/{total} sequence lengths")

    long_wins = sum(1 for sl, r in results.items() if sl >= 1024 and r['speedup'] > 1.0)
    long_total = sum(1 for sl in results if sl >= 1024)
    if long_total > 0:
        print(f"  At 1024+ tokens: {long_wins}/{long_total} wins")

    # Save
    save_path = os.path.join(os.path.dirname(__file__), "scaling_results.json")
    with open(save_path, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")

    return results


if __name__ == "__main__":
    run_scaling_benchmark(
        sequence_lengths=[64, 128, 256, 512, 1024, 2048],
        batch_size=8,
        vocab_size=10000,
        embed_dim=64,
        field_channels=64,
        num_warmup=3,
        num_runs=10,
    )
