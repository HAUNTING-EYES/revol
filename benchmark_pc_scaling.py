"""
Predictive Coding Sequence Length Scaling Benchmark

Tests PredictiveCoding vs WaveNetNeuro (Fixed-10) vs Transformer at 512-8192 tokens.
Hypothesis: PredCoding is O(n) for embedding + O(1) for iterations.
Should scale even better than WaveNetNeuro since iterations don't touch the sequence.
"""

import torch
import time
import json
import os
import gc
import numpy as np
from wavenet_neuro import WaveNetNeuro, BaselineTransformer, FixedStepEvolution
from predictive_coding import PredictiveCodingNetwork


def measure_forward_pass(model, x, device, num_warmup=2, num_runs=5):
    """Measure average forward pass time with warmup."""
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
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000,
    }


def run_pc_scaling(
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
    print("PREDICTIVE CODING SEQUENCE LENGTH SCALING")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence lengths: {sequence_lengths}")
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
            'pred_coding': None,
            'wavenet_f10': None,
            'transformer': None,
        }

        # --- Predictive Coding ---
        print(f"  PredictiveCoding...")
        try:
            pc = PredictiveCodingNetwork(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                num_layers=4,
                num_classes=2,
                max_iterations=10,
                convergence_threshold=1e-10,
                max_seq_len=max(seq_len, 512),
                bidirectional=True,
            ).to(device)

            pc_time = measure_forward_pass(pc, x, device, num_warmup, num_runs)
            entry['pred_coding'] = {**pc_time, 'params': pc.count_parameters(), 'status': 'OK'}
            print(f"    Time: {pc_time['mean_ms']:.1f}ms (+/- {pc_time['std_ms']:.1f}ms)")
            del pc
            gc.collect()
        except Exception as e:
            entry['pred_coding'] = {'mean_ms': float('inf'), 'status': f'FAILED: {e}'}
            print(f"    FAILED: {e}")

        # --- WaveNetNeuro Fixed-10 ---
        print(f"  WaveNet Fixed-10 (spatial_dim={spatial_dim})...")
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
            wavenet.evolution = FixedStepEvolution(wavenet.dynamics, fixed_steps=10)

            w_time = measure_forward_pass(wavenet, x, device, num_warmup, num_runs)
            entry['wavenet_f10'] = {**w_time, 'params': wavenet.count_parameters(), 'status': 'OK'}
            print(f"    Time: {w_time['mean_ms']:.1f}ms (+/- {w_time['std_ms']:.1f}ms)")
            del wavenet
            gc.collect()
        except Exception as e:
            entry['wavenet_f10'] = {'mean_ms': float('inf'), 'status': f'FAILED: {e}'}
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
            entry['transformer'] = {**t_time, 'params': transformer.count_parameters(), 'status': 'OK'}
            print(f"    Time: {t_time['mean_ms']:.1f}ms (+/- {t_time['std_ms']:.1f}ms)")
            del transformer
            gc.collect()
        except Exception as e:
            entry['transformer'] = {'mean_ms': float('inf'), 'status': f'FAILED: {e}'}
            print(f"    FAILED: {e}")

        # Speedups
        pc_ms = entry['pred_coding']['mean_ms'] if entry['pred_coding'] else float('inf')
        w_ms = entry['wavenet_f10']['mean_ms'] if entry['wavenet_f10'] else float('inf')
        t_ms = entry['transformer']['mean_ms'] if entry['transformer'] else float('inf')

        if pc_ms != float('inf'):
            entry['speedup_vs_transformer'] = t_ms / pc_ms if t_ms != float('inf') else float('inf')
            entry['speedup_vs_wavenet'] = w_ms / pc_ms
            print(f"  PC vs Transformer: {entry['speedup_vs_transformer']:.2f}x")
            print(f"  PC vs WaveNet F10: {entry['speedup_vs_wavenet']:.2f}x")

        results[seq_len] = entry

        del x
        gc.collect()

    # --- Summary ---
    print(f"\n{'='*80}")
    print("SCALING SUMMARY")
    print(f"{'='*80}")
    print(f"{'SeqLen':>8} | {'PredCoding':>12} | {'WaveNet F10':>12} | {'Transformer':>12} | {'PC/T':>7} | {'PC/W':>7}")
    print("-" * 75)

    for sl in sequence_lengths:
        r = results[sl]
        pc_ms = r['pred_coding']['mean_ms'] if r['pred_coding'] and r['pred_coding']['mean_ms'] != float('inf') else None
        w_ms = r['wavenet_f10']['mean_ms'] if r['wavenet_f10'] and r['wavenet_f10']['mean_ms'] != float('inf') else None
        t_ms = r['transformer']['mean_ms'] if r['transformer'] and r['transformer']['mean_ms'] != float('inf') else None

        pc_str = f"{pc_ms:>11.1f}ms" if pc_ms else f"{'FAIL':>12}"
        w_str = f"{w_ms:>11.1f}ms" if w_ms else f"{'FAIL':>12}"
        t_str = f"{t_ms:>11.1f}ms" if t_ms else f"{'FAIL':>12}"
        sp_t = f"{t_ms / pc_ms:>6.1f}x" if (pc_ms and t_ms) else f"{'N/A':>7}"
        sp_w = f"{w_ms / pc_ms:>6.1f}x" if (pc_ms and w_ms) else f"{'N/A':>7}"

        print(f"{sl:>8} | {pc_str} | {w_str} | {t_str} | {sp_t} | {sp_w}")

    # Scaling analysis
    ok = [sl for sl in sequence_lengths
          if results[sl]['pred_coding'] and results[sl]['pred_coding']['mean_ms'] != float('inf')]
    if len(ok) >= 2:
        sl1, sl_last = ok[0], ok[-1]
        ratio = sl_last / sl1
        pc_ratio = results[sl_last]['pred_coding']['mean_ms'] / results[sl1]['pred_coding']['mean_ms']
        print(f"\nPredCoding scaling: {sl1} -> {sl_last} ({ratio:.0f}x length)")
        print(f"  Time increase: {pc_ratio:.1f}x  (ideal O(n): {ratio:.0f}x)")

    ok_w = [sl for sl in sequence_lengths
            if results[sl]['wavenet_f10'] and results[sl]['wavenet_f10']['mean_ms'] != float('inf')]
    if len(ok_w) >= 2:
        sl1, sl_last = ok_w[0], ok_w[-1]
        ratio = sl_last / sl1
        w_ratio = results[sl_last]['wavenet_f10']['mean_ms'] / results[sl1]['wavenet_f10']['mean_ms']
        print(f"\nWaveNet F10 scaling: {sl1} -> {sl_last} ({ratio:.0f}x length)")
        print(f"  Time increase: {w_ratio:.1f}x")

    ok_t = [sl for sl in sequence_lengths
            if results[sl]['transformer'] and results[sl]['transformer']['mean_ms'] != float('inf')]
    if len(ok_t) >= 2:
        sl1, sl_last = ok_t[0], ok_t[-1]
        ratio = sl_last / sl1
        t_ratio = results[sl_last]['transformer']['mean_ms'] / results[sl1]['transformer']['mean_ms']
        print(f"\nTransformer scaling: {sl1} -> {sl_last} ({ratio:.0f}x length)")
        print(f"  Time increase: {t_ratio:.1f}x  (ideal O(n^2): {ratio**2:.0f}x)")

    # Verdict
    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")
    big_wins = [sl for sl in sequence_lengths
                if results[sl].get('speedup_vs_wavenet', 0) > 1.0]
    if big_wins:
        print(f"  PredCoding faster than WaveNet F10 at: {big_wins}")
    else:
        print(f"  WaveNet F10 faster at all tested lengths")

    # Save
    save_path = os.path.join(os.path.dirname(__file__), "pc_scaling_results.json")
    with open(save_path, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")

    return results


if __name__ == "__main__":
    run_pc_scaling(
        sequence_lengths=[512, 1024, 2048, 4096, 8192],
        batch_size=4,
        vocab_size=10000,
        embed_dim=64,
        field_channels=64,
        num_warmup=2,
        num_runs=5,
    )
