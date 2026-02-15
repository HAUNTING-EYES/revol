"""
Sequence length scaling benchmark for all 5 architectures.

Tests at: 512, 1024, 2048, 4096, 8192
Reports forward pass timing for each model at each length.
"""

import torch
import time
import json
import os
import gc
import math
import numpy as np
from wavenet_neuro import WaveNetNeuro, BaselineTransformer, FixedStepEvolution
from predictive_coding import PredictiveCodingNetwork
from hierarchical_predictive_coding import HierarchicalPredCodingNetwork
from multiscale_parallel import MultiScaleNetwork


def measure_forward(model, x, device, num_warmup=2, num_runs=5):
    model.eval()
    x = x.to(device)
    with torch.no_grad():
        for _ in range(num_warmup):
            model(x)
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            model(x)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
    }


def run_scaling(
    sequence_lengths=None,
    batch_size=4,
    vocab_size=10000,
    embed_dim=64,
    field_channels=64,
):
    if sequence_lengths is None:
        sequence_lengths = [512, 1024, 2048, 4096, 8192]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 90)
    print("SEQUENCE LENGTH SCALING - ALL 5 ARCHITECTURES")
    print("=" * 90)
    print(f"Device: {device}, batch={batch_size}")
    print(f"Lengths: {sequence_lengths}")
    print()

    results = {}

    for seq_len in sequence_lengths:
        print(f"\n{'='*70}")
        print(f"Sequence Length: {seq_len}")
        print(f"{'='*70}")

        spatial_dim = max(16, int(math.ceil(math.sqrt(seq_len))))
        while spatial_dim * spatial_dim < seq_len:
            spatial_dim += 1

        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        entry = {}

        # 1. Transformer
        print(f"  Transformer...")
        try:
            m = BaselineTransformer(vocab_size=vocab_size, embed_dim=embed_dim,
                                    num_heads=4, num_layers=2, num_classes=2,
                                    max_seq_len=max(seq_len, 512)).to(device)
            t = measure_forward(m, x, device)
            entry['Transformer'] = {**t, 'params': m.count_parameters(), 'status': 'OK'}
            print(f"    {t['mean_ms']:.1f}ms")
            del m; gc.collect()
        except Exception as e:
            entry['Transformer'] = {'mean_ms': float('inf'), 'status': f'FAIL: {e}'}
            print(f"    FAILED: {e}")

        # 2. WaveNet Fixed-10
        print(f"  WaveNet Fixed-10...")
        try:
            m = WaveNetNeuro(vocab_size=vocab_size, embed_dim=embed_dim,
                             field_channels=field_channels, spatial_dim=spatial_dim,
                             num_classes=2, max_evolution_steps=30,
                             convergence_threshold=0.1, dt=0.3).to(device)
            m.evolution = FixedStepEvolution(m.dynamics, fixed_steps=10)
            t = measure_forward(m, x, device)
            entry['WaveNet_F10'] = {**t, 'params': m.count_parameters(), 'status': 'OK'}
            print(f"    {t['mean_ms']:.1f}ms")
            del m; gc.collect()
        except Exception as e:
            entry['WaveNet_F10'] = {'mean_ms': float('inf'), 'status': f'FAIL: {e}'}
            print(f"    FAILED: {e}")

        # 3. PredCoding Pooled
        print(f"  PredCoding Pooled...")
        try:
            m = PredictiveCodingNetwork(vocab_size=vocab_size, embed_dim=embed_dim,
                                        num_layers=4, num_classes=2, max_iterations=10,
                                        convergence_threshold=1e-10,
                                        max_seq_len=max(seq_len, 512)).to(device)
            t = measure_forward(m, x, device)
            entry['PredCoding_Pooled'] = {**t, 'params': m.count_parameters(), 'status': 'OK'}
            print(f"    {t['mean_ms']:.1f}ms")
            del m; gc.collect()
        except Exception as e:
            entry['PredCoding_Pooled'] = {'mean_ms': float('inf'), 'status': f'FAIL: {e}'}
            print(f"    FAILED: {e}")

        # 4. Hierarchical PredCoding
        print(f"  Hierarchical PC...")
        try:
            m = HierarchicalPredCodingNetwork(
                vocab_size=vocab_size, embed_dim=embed_dim, num_layers=4,
                num_classes=2, max_iterations=10, convergence_threshold=1e-10,
                max_seq_len=max(seq_len, 512)).to(device)
            t = measure_forward(m, x, device)
            entry['HierPC'] = {**t, 'params': m.count_parameters(), 'status': 'OK'}
            print(f"    {t['mean_ms']:.1f}ms")
            del m; gc.collect()
        except Exception as e:
            entry['HierPC'] = {'mean_ms': float('inf'), 'status': f'FAIL: {e}'}
            print(f"    FAILED: {e}")

        # 5. MultiScale
        print(f"  MultiScale...")
        try:
            m = MultiScaleNetwork(vocab_size=vocab_size, embed_dim=embed_dim,
                                  num_classes=2, max_seq_len=max(seq_len, 512)).to(device)
            t = measure_forward(m, x, device)
            entry['MultiScale'] = {**t, 'params': m.count_parameters(), 'status': 'OK'}
            print(f"    {t['mean_ms']:.1f}ms")
            del m; gc.collect()
        except Exception as e:
            entry['MultiScale'] = {'mean_ms': float('inf'), 'status': f'FAIL: {e}'}
            print(f"    FAILED: {e}")

        results[seq_len] = entry
        del x; gc.collect()

    # --- Summary table ---
    print(f"\n{'='*90}")
    print("SCALING SUMMARY (ms)")
    print(f"{'='*90}")

    model_names = ['Transformer', 'WaveNet_F10', 'PredCoding_Pooled', 'HierPC', 'MultiScale']
    header = f"{'SeqLen':>8}"
    for mn in model_names:
        header += f" | {mn:>14}"
    print(header)
    print("-" * len(header))

    for sl in sequence_lengths:
        r = results[sl]
        row = f"{sl:>8}"
        for mn in model_names:
            ms = r.get(mn, {}).get('mean_ms', float('inf'))
            if ms == float('inf'):
                row += f" | {'FAIL':>14}"
            else:
                row += f" | {ms:>13.1f}ms"
        print(row)

    # Speedup vs transformer
    print(f"\nSpeedup vs Transformer:")
    for sl in sequence_lengths:
        r = results[sl]
        t_ms = r.get('Transformer', {}).get('mean_ms', float('inf'))
        if t_ms == float('inf'):
            continue
        row = f"  {sl:>6}:"
        for mn in model_names[1:]:
            ms = r.get(mn, {}).get('mean_ms', float('inf'))
            if ms != float('inf') and ms > 0:
                row += f"  {mn}={t_ms/ms:.1f}x"
        print(row)

    # Save
    save_path = os.path.join(os.path.dirname(__file__), "novel_scaling_results.json")
    with open(save_path, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")

    return results


if __name__ == "__main__":
    run_scaling()
