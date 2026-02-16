"""
Extreme Sequence Length Scaling Benchmark (2048, 4096, 8192)

Tests where WaveNetNeuro's O(n) should completely dominate Transformer's O(n^2).
Measures forward pass time, peak memory (estimated via tensor sizes on CPU),
and tracks whether transformer can even complete at extreme lengths.
"""

import torch
import traceback
import time
import json
import os
import sys
import gc
import numpy as np
from wavenet_neuro import WaveNetNeuro, BaselineTransformer


def estimate_cpu_memory_mb(model, x):
    """Estimate memory by tracking tensor allocations around a forward pass."""
    gc.collect()
    # Rough estimate: model params + input + intermediate activations
    param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    input_bytes = x.nelement() * x.element_size()
    # Run forward and estimate output size
    model.eval()
    with torch.no_grad():
        output, info = model(x)
    output_bytes = output.nelement() * output.element_size()
    # Activations are hard to measure on CPU; estimate as ~3x param size
    estimated_mb = (param_bytes * 3 + input_bytes + output_bytes) / (1024 ** 2)
    return estimated_mb


def measure_forward_pass(model, x, device, num_warmup=2, num_runs=5):
    """Measure average forward pass time with warmup."""
    model.eval()
    x = x.to(device)

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
        'all_ms': [t * 1000 for t in times],
    }


def run_extreme_scaling(
    sequence_lengths=None,
    batch_size=4,
    vocab_size=10000,
    embed_dim=64,
    field_channels=64,
    num_warmup=2,
    num_runs=5,
):
    if sequence_lengths is None:
        sequence_lengths = [512, 1024, 2048, 4096, 8192]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("EXTREME SEQUENCE LENGTH SCALING (4096+ tokens)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence lengths: {sequence_lengths}")
    print(f"Embed dim: {embed_dim}, Field channels: {field_channels}")
    print()

    results = {}

    for seq_len in sequence_lengths:
        print(f"\n{'='*60}")
        print(f"Sequence Length: {seq_len}")
        print(f"{'='*60}")

        spatial_dim = max(16, int(np.ceil(np.sqrt(seq_len))))
        while spatial_dim * spatial_dim < seq_len:
            spatial_dim += 1

        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        entry = {
            'seq_len': seq_len,
            'spatial_dim': spatial_dim,
            'wavenet': None,
            'transformer': None,
            'speedup': None,
        }

        # --- WaveNetNeuro ---
        print(f"  WaveNetNeuro (spatial_dim={spatial_dim})...")
        try:
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

            w_time = measure_forward_pass(wavenet, x, device, num_warmup, num_runs)
            w_mem = estimate_cpu_memory_mb(wavenet, x.to(device))
            w_params = wavenet.count_parameters()

            entry['wavenet'] = {
                **w_time,
                'memory_est_mb': w_mem,
                'params': w_params,
                'status': 'OK',
            }
            print(f"    Time: {w_time['mean_ms']:.1f}ms (+/- {w_time['std_ms']:.1f}ms)")
            print(f"    Params: {w_params:,}")

            del wavenet
            gc.collect()
        except Exception as e:
            entry['wavenet'] = {
                'mean_ms': float('inf'),
                'status': f'FAILED: {e}',
            }
            print(f"    FAILED: {e}")

        # --- Transformer ---
        print(f"  Transformer...")
        try:
            transformer = BaselineTransformer(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                num_heads=4,
                num_layers=2,
                num_classes=2,
                max_seq_len=max(seq_len, 512),
            ).to(device)

            t_time = measure_forward_pass(transformer, x, device, num_warmup, num_runs)
            t_mem = estimate_cpu_memory_mb(transformer, x.to(device))
            t_params = transformer.count_parameters()

            entry['transformer'] = {
                **t_time,
                'memory_est_mb': t_mem,
                'params': t_params,
                'status': 'OK',
            }
            print(f"    Time: {t_time['mean_ms']:.1f}ms (+/- {t_time['std_ms']:.1f}ms)")
            print(f"    Params: {t_params:,}")

            del transformer
            gc.collect()
        except Exception as e:
            entry['transformer'] = {
                'mean_ms': float('inf'),
                'status': f'FAILED: {e}',
            }
            print(f"    FAILED: {e}")

        # Speedup
        w_ms = entry['wavenet']['mean_ms'] if entry['wavenet'] else float('inf')
        t_ms = entry['transformer']['mean_ms'] if entry['transformer'] else float('inf')
        if w_ms > 0 and w_ms != float('inf'):
            entry['speedup'] = t_ms / w_ms
            print(f"  Speedup (T/W): {entry['speedup']:.2f}x")
        else:
            entry['speedup'] = 0
            print(f"  Speedup: N/A")

        results[seq_len] = entry

        del x
        gc.collect()

    # --- Summary ---
    print(f"\n{'='*80}")
    print("EXTREME SCALING SUMMARY")
    print(f"{'='*80}")
    print(f"{'SeqLen':>8} | {'WaveNet (ms)':>14} | {'Transformer (ms)':>17} | {'Speedup':>10} | {'Status':>12}")
    print("-" * 70)

    for sl in sequence_lengths:
        r = results[sl]
        w_status = r['wavenet']['status'] if r['wavenet'] else 'N/A'
        t_status = r['transformer']['status'] if r['transformer'] else 'N/A'
        w_ms = r['wavenet']['mean_ms'] if r['wavenet'] and r['wavenet']['mean_ms'] != float('inf') else None
        t_ms = r['transformer']['mean_ms'] if r['transformer'] and r['transformer']['mean_ms'] != float('inf') else None

        w_str = f"{w_ms:>14.1f}" if w_ms else f"{'FAIL':>14}"
        t_str = f"{t_ms:>17.1f}" if t_ms else f"{'FAIL':>17}"
        sp_str = f"{r['speedup']:>9.2f}x" if r['speedup'] and r['speedup'] > 0 else f"{'N/A':>10}"

        status = "OK" if w_status == 'OK' and t_status == 'OK' else ""
        if t_status != 'OK' and w_status == 'OK':
            status = "T_FAIL"
        elif w_status != 'OK':
            status = "W_FAIL"

        print(f"{sl:>8} | {w_str} | {t_str} | {sp_str} | {status:>12}")

    # Scaling analysis
    ok_lengths = [sl for sl in sequence_lengths
                  if results[sl]['wavenet'] and results[sl]['wavenet']['mean_ms'] != float('inf')]
    if len(ok_lengths) >= 2:
        sl_1, sl_last = ok_lengths[0], ok_lengths[-1]
        ratio = sl_last / sl_1
        w_ratio = results[sl_last]['wavenet']['mean_ms'] / results[sl_1]['wavenet']['mean_ms']
        print(f"\nWaveNetNeuro scaling: {sl_1} -> {sl_last} ({ratio:.0f}x length)")
        print(f"  Time increase: {w_ratio:.1f}x  (ideal O(n): {ratio:.0f}x)")

    ok_t = [sl for sl in sequence_lengths
            if results[sl]['transformer'] and results[sl]['transformer']['mean_ms'] != float('inf')]
    if len(ok_t) >= 2:
        sl_1, sl_last = ok_t[0], ok_t[-1]
        ratio = sl_last / sl_1
        t_ratio = results[sl_last]['transformer']['mean_ms'] / results[sl_1]['transformer']['mean_ms']
        print(f"\nTransformer scaling: {sl_1} -> {sl_last} ({ratio:.0f}x length)")
        print(f"  Time increase: {t_ratio:.1f}x  (ideal O(n^2): {ratio**2:.0f}x)")

    # Verdict
    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")
    t_fails = [sl for sl in sequence_lengths
               if results[sl]['transformer'] and results[sl]['transformer']['status'] != 'OK']
    w_fails = [sl for sl in sequence_lengths
               if results[sl]['wavenet'] and results[sl]['wavenet']['status'] != 'OK']

    if t_fails and not w_fails:
        print(f"  Transformer FAILED at: {t_fails}")
        print(f"  WaveNetNeuro succeeded at all lengths")
        print(f"  CLEAR WIN for WaveNetNeuro at extreme lengths")
    elif not t_fails and not w_fails:
        big_wins = [sl for sl in sequence_lengths if results[sl]['speedup'] and results[sl]['speedup'] > 2.0]
        if big_wins:
            best = max(big_wins, key=lambda sl: results[sl]['speedup'])
            print(f"  WaveNetNeuro >2x faster at: {big_wins}")
            print(f"  Best speedup: {results[best]['speedup']:.1f}x at {best} tokens")

    # Save
    save_path = os.path.join(os.path.dirname(__file__), "extreme_scaling_results.json")
    with open(save_path, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")

    return results


if __name__ == "__main__":
    run_extreme_scaling(
        sequence_lengths=[512, 1024, 2048, 4096, 8192],
        batch_size=4,
        vocab_size=10000,
        embed_dim=64,
        field_channels=64,
        num_warmup=2,
        num_runs=5,
    )
