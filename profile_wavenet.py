"""
Profiling WaveNetNeuro to find bottlenecks.

Uses torch.profiler on CPU to identify:
1. Where most time is spent (per-op breakdown)
2. Which ops could be parallelized
3. Memory allocation patterns
4. Overhead from convergence checking vs actual computation
"""

import torch
import torch.profiler
import time
import json
import os
import gc
import numpy as np
from wavenet_neuro import WaveNetNeuro, BaselineTransformer, FixedStepEvolution


def profile_model(model, x, label="Model", num_warmup=3, num_profile=5):
    """Profile a model's forward pass and return op-level breakdown."""
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            model(x)

    # Profile
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            for _ in range(num_profile):
                model(x)

    # Get summary
    print(f"\n{'='*80}")
    print(f"PROFILE: {label}")
    print(f"{'='*80}")

    # Top ops by CPU time
    table = prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=20,
    )
    print(table)

    # Extract structured data
    events = prof.key_averages()
    op_data = []
    total_cpu_time = sum(e.cpu_time_total for e in events if e.cpu_time_total > 0)

    for event in sorted(events, key=lambda e: e.cpu_time_total, reverse=True)[:15]:
        pct = (event.cpu_time_total / total_cpu_time * 100) if total_cpu_time > 0 else 0
        op_data.append({
            'name': event.key,
            'cpu_time_us': event.cpu_time_total,
            'cpu_time_pct': round(pct, 1),
            'calls': event.count,
            'cpu_time_per_call_us': round(event.cpu_time_total / max(event.count, 1)),
            'cpu_mem_kb': round(event.cpu_memory_usage / 1024) if hasattr(event, 'cpu_memory_usage') else 0,
        })

    return op_data, total_cpu_time


def profile_evolution_breakdown(seq_len=256, embed_dim=64, field_channels=64,
                                vocab_size=10000, batch_size=8):
    """
    Break down time spent in each phase of WaveNetNeuro:
    1. Embedding + field init
    2. Evolution loop (each step)
    3. Output projection
    """
    device = torch.device('cpu')

    spatial_dim = max(16, int(np.ceil(np.sqrt(seq_len))))
    while spatial_dim * spatial_dim < seq_len:
        spatial_dim += 1

    model = WaveNetNeuro(
        vocab_size=vocab_size, embed_dim=embed_dim,
        field_channels=field_channels, spatial_dim=spatial_dim,
        num_classes=2, max_evolution_steps=30,
        convergence_threshold=0.1, dt=0.3,
    ).to(device)
    model.eval()

    x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

    print(f"\n{'='*80}")
    print(f"EVOLUTION BREAKDOWN (seq_len={seq_len})")
    print(f"{'='*80}")

    # Time each phase manually
    num_runs = 10
    embed_times = []
    evolution_times = []
    output_times = []
    step_times = []

    with torch.no_grad():
        for _ in range(num_runs):
            # Phase 1: Embedding
            t0 = time.perf_counter()
            token_embeds = model.embedding(x)
            pos_embeds = model.pos_encoding[:, :seq_len, :]
            embeddings = token_embeds + pos_embeds
            embeddings = model.to_field(embeddings)
            field = model.field.initialize_from_sequence(embeddings)
            t1 = time.perf_counter()
            embed_times.append((t1 - t0) * 1000)

            # Phase 2: Evolution (step by step)
            current_field = field
            run_step_times = []
            initial_change = None

            for step in range(model.max_evolution_steps):
                ts = time.perf_counter()
                target = model.dynamics(current_field)
                new_field = current_field + model.dt * (target - current_field)

                # Convergence check
                per_example_change = (
                    torch.abs(new_field - current_field)
                    .reshape(batch_size, -1)
                    .mean(dim=1)
                )
                if step == 0:
                    initial_change = per_example_change.clone().clamp(min=1e-8)
                relative_change = per_example_change / initial_change
                converged = (relative_change < model.convergence_threshold).all()

                current_field = new_field
                te = time.perf_counter()
                run_step_times.append((te - ts) * 1000)

                if converged:
                    break

            t2 = time.perf_counter()
            evolution_times.append((t2 - t1) * 1000)
            step_times.append(run_step_times)

            # Phase 3: Output
            output = model.from_field(current_field)
            t3 = time.perf_counter()
            output_times.append((t3 - t2) * 1000)

    avg_embed = np.mean(embed_times)
    avg_evolution = np.mean(evolution_times)
    avg_output = np.mean(output_times)
    total = avg_embed + avg_evolution + avg_output

    print(f"\nPhase Breakdown (avg over {num_runs} runs):")
    print(f"  Embedding + Field Init:  {avg_embed:>8.2f}ms  ({avg_embed/total*100:>5.1f}%)")
    print(f"  Evolution Loop:          {avg_evolution:>8.2f}ms  ({avg_evolution/total*100:>5.1f}%)")
    print(f"  Output Projection:       {avg_output:>8.2f}ms  ({avg_output/total*100:>5.1f}%)")
    print(f"  Total:                   {total:>8.2f}ms")

    # Per-step breakdown
    avg_num_steps = np.mean([len(st) for st in step_times])
    all_step_ms = [ms for run in step_times for ms in run]
    avg_step = np.mean(all_step_ms) if all_step_ms else 0

    print(f"\nEvolution Details:")
    print(f"  Average steps taken: {avg_num_steps:.1f}")
    print(f"  Average time per step: {avg_step:.2f}ms")
    print(f"  Evolution = {avg_num_steps:.1f} steps x {avg_step:.2f}ms = {avg_num_steps * avg_step:.1f}ms")

    # Time in dynamics vs convergence check
    # Estimate: dynamics is the conv2d ops, convergence check is the abs/mean/compare
    print(f"\nBottleneck Analysis:")
    print(f"  Evolution takes {avg_evolution/total*100:.0f}% of total time")
    print(f"  Each step: dynamics (conv2d + reaction) + convergence check")
    print(f"  Key ops: depthwise conv2d (3x3), 1x1 conv2d (reaction), GroupNorm")

    return {
        'embed_ms': avg_embed,
        'evolution_ms': avg_evolution,
        'output_ms': avg_output,
        'total_ms': total,
        'avg_steps': avg_num_steps,
        'ms_per_step': avg_step,
        'embed_pct': avg_embed / total * 100,
        'evolution_pct': avg_evolution / total * 100,
        'output_pct': avg_output / total * 100,
    }


def run_profiling(seq_len=256, batch_size=8, vocab_size=10000,
                  embed_dim=64, field_channels=64):
    """Complete profiling pipeline."""
    device = torch.device('cpu')

    print("=" * 80)
    print("WAVENETNEURO PROFILING")
    print("=" * 80)
    print(f"Seq len: {seq_len}, Batch size: {batch_size}")
    print(f"Embed dim: {embed_dim}, Field channels: {field_channels}")

    spatial_dim = max(16, int(np.ceil(np.sqrt(seq_len))))
    while spatial_dim * spatial_dim < seq_len:
        spatial_dim += 1

    x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

    # Profile WaveNetNeuro (adaptive)
    wavenet = WaveNetNeuro(
        vocab_size=vocab_size, embed_dim=embed_dim,
        field_channels=field_channels, spatial_dim=spatial_dim,
        num_classes=2, max_evolution_steps=30,
        convergence_threshold=0.1, dt=0.3,
    ).to(device)
    w_ops, w_total = profile_model(wavenet, x, "WaveNetNeuro (Adaptive)")

    # Profile WaveNetNeuro (fixed-10)
    wavenet_fixed = WaveNetNeuro(
        vocab_size=vocab_size, embed_dim=embed_dim,
        field_channels=field_channels, spatial_dim=spatial_dim,
        num_classes=2, max_evolution_steps=30,
        convergence_threshold=0.1, dt=0.3,
    ).to(device)
    wavenet_fixed.evolution = FixedStepEvolution(wavenet_fixed.dynamics, fixed_steps=10)
    f_ops, f_total = profile_model(wavenet_fixed, x, "WaveNetNeuro (Fixed-10)")

    # Profile Transformer
    transformer = BaselineTransformer(
        vocab_size=vocab_size, embed_dim=embed_dim,
        num_heads=4, num_layers=2, num_classes=2,
        max_seq_len=max(seq_len, 512),
    ).to(device)
    t_ops, t_total = profile_model(transformer, x, "Transformer")

    # Evolution breakdown
    breakdown = profile_evolution_breakdown(
        seq_len=seq_len, embed_dim=embed_dim,
        field_channels=field_channels, vocab_size=vocab_size,
        batch_size=batch_size,
    )

    # Summary
    print(f"\n{'='*80}")
    print("PROFILING SUMMARY")
    print(f"{'='*80}")
    print(f"  WaveNetNeuro (adaptive): {w_total/1000:.1f}ms total CPU time")
    print(f"  WaveNetNeuro (fixed-10): {f_total/1000:.1f}ms total CPU time")
    print(f"  Transformer:             {t_total/1000:.1f}ms total CPU time")

    print(f"\n  WaveNetNeuro top ops:")
    for op in w_ops[:5]:
        print(f"    {op['name']:<40} {op['cpu_time_pct']:>5.1f}%  "
              f"({op['calls']} calls, {op['cpu_time_per_call_us']}us/call)")

    print(f"\n  Transformer top ops:")
    for op in t_ops[:5]:
        print(f"    {op['name']:<40} {op['cpu_time_pct']:>5.1f}%  "
              f"({op['calls']} calls, {op['cpu_time_per_call_us']}us/call)")

    print(f"\n  Optimization Opportunities:")
    if breakdown['evolution_pct'] > 70:
        print(f"    - Evolution loop takes {breakdown['evolution_pct']:.0f}% "
              f"-- reducing steps has high impact")
    print(f"    - Fixed-10 saves {(w_total - f_total)/w_total*100:.0f}% "
          f"CPU time vs adaptive")
    print(f"    - Each evolution step costs {breakdown['ms_per_step']:.2f}ms "
          f"-- {breakdown['avg_steps']:.0f} steps taken")

    # Save
    save_path = os.path.join(os.path.dirname(__file__), "profile_results.json")
    results = {
        'wavenet_adaptive': {'ops': w_ops, 'total_cpu_us': w_total},
        'wavenet_fixed': {'ops': f_ops, 'total_cpu_us': f_total},
        'transformer': {'ops': t_ops, 'total_cpu_us': t_total},
        'breakdown': breakdown,
    }
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")

    return results


if __name__ == "__main__":
    run_profiling(
        seq_len=256,
        batch_size=8,
        vocab_size=10000,
        embed_dim=64,
        field_channels=64,
    )
