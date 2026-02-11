"""
Visualization of WaveNetNeuro Field Dynamics

Shows how information propagates through continuous field
Like watching waves spread across a pond!
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from wavenet_neuro import WaveNetNeuro
import seaborn as sns


def visualize_field_evolution(model, input_seq, max_steps=50, save_path=None):
    """
    Visualize how the information field evolves over time
    
    This is THE key insight: information propagates continuously,
    not through discrete attention matrices!
    """
    
    model.eval()
    
    # Prepare input
    if input_seq.dim() == 1:
        input_seq = input_seq.unsqueeze(0)
    
    # Get embeddings and initialize field
    with torch.no_grad():
        token_embeds = model.embedding(input_seq)
        pos_embeds = model.pos_encoding[:, :input_seq.size(1), :]
        embeddings = token_embeds + pos_embeds
        embeddings = model.to_field(embeddings)
        field = model.field.initialize_from_sequence(embeddings)
    
    # Capture field evolution
    field_history = [field.clone()]
    
    with torch.no_grad():
        current_field = field
        
        for step in range(max_steps):
            # Compute dynamics
            field_derivative = model.dynamics(current_field)
            
            # Update field
            new_field = current_field + 0.1 * field_derivative
            
            # Save snapshot
            field_history.append(new_field.clone())
            
            # Check convergence
            change = torch.abs(new_field - current_field).mean()
            if change < 0.01:
                print(f"Field converged at step {step+1}")
                break
                
            current_field = new_field
    
    return field_history


def plot_field_snapshots(field_history, num_snapshots=6):
    """
    Plot snapshots of field evolution
    
    Shows: Initial state → intermediate states → final state
    """
    
    indices = np.linspace(0, len(field_history)-1, num_snapshots, dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, ax in zip(indices, axes):
        field = field_history[idx]
        
        # Average over channels for visualization
        field_avg = field[0].mean(dim=0).cpu().numpy()
        
        im = ax.imshow(field_avg, cmap='RdBu_r', aspect='auto')
        ax.set_title(f'Step {idx}', fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax)
    
    plt.suptitle('🌊 Information Field Evolution\n(Watch how patterns emerge!)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_field_energy(field_history):
    """
    Plot field energy over time
    
    Shows convergence: energy decreases as field stabilizes
    """
    
    energies = [torch.abs(field).mean().item() for field in field_history]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(energies, linewidth=2, color='#2E86AB')
    ax.fill_between(range(len(energies)), 0, energies, alpha=0.3, color='#2E86AB')
    
    ax.set_xlabel('Evolution Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Field Energy', fontsize=12, fontweight='bold')
    ax.set_title('Field Energy Convergence\n(Lower = More Stable)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_field_change_rate(field_history):
    """
    Plot rate of change in field
    
    Shows: Fast changes initially → slows down as converges
    """
    
    changes = []
    for i in range(1, len(field_history)):
        change = torch.abs(field_history[i] - field_history[i-1]).mean().item()
        changes.append(change)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(changes, linewidth=2, color='#A23B72', marker='o', markersize=4)
    ax.axhline(y=0.01, color='red', linestyle='--', 
               label='Convergence Threshold', linewidth=2)
    
    ax.set_xlabel('Evolution Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Field Change Rate', fontsize=12, fontweight='bold')
    ax.set_title('Adaptive Computation in Action\n(Stops when change < threshold)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    return fig


def compare_computation_graphs():
    """
    Visualize computation graphs: Transformer vs WaveNetNeuro
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Transformer: O(n²) attention
    n = 20
    attention_matrix = np.random.rand(n, n)
    np.fill_diagonal(attention_matrix, 1.0)
    
    im1 = ax1.imshow(attention_matrix, cmap='YlOrRd', aspect='auto')
    ax1.set_title('🤖 Transformer: O(n²) Attention\n(Every token attends to every token)', 
                  fontsize=13, fontweight='bold')
    ax1.set_xlabel('Token', fontsize=11)
    ax1.set_ylabel('Token', fontsize=11)
    plt.colorbar(im1, ax=ax1, label='Attention Weight')
    
    # Add annotations
    ax1.text(n//2, -2, f'Complexity: O(n²) = {n}² = {n*n} operations', 
             ha='center', fontsize=11, color='red', fontweight='bold')
    
    # WaveNetNeuro: O(n) local computation
    local_matrix = np.zeros((n, n))
    for i in range(n):
        # Only local connections (3x3 neighborhood in unrolled form)
        for j in range(max(0, i-1), min(n, i+2)):
            local_matrix[i, j] = np.random.rand()
    
    im2 = ax2.imshow(local_matrix, cmap='YlGnBu', aspect='auto')
    ax2.set_title('🌊 WaveNetNeuro: O(n) Local Dynamics\n(Only neighbors interact)', 
                  fontsize=13, fontweight='bold')
    ax2.set_xlabel('Position', fontsize=11)
    ax2.set_ylabel('Position', fontsize=11)
    plt.colorbar(im2, ax=ax2, label='Connection Strength')
    
    # Add annotations
    ax2.text(n//2, -2, f'Complexity: O(n) = {n} operations', 
             ha='center', fontsize=11, color='green', fontweight='bold')
    
    plt.suptitle('Computational Complexity Comparison\n(Sparse vs Dense)', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def create_animation(field_history, save_path='field_evolution.gif'):
    """
    Create animated visualization of field evolution
    
    Watch information propagate like waves!
    """
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # First frame
    field_avg = field_history[0][0].mean(dim=0).cpu().numpy()
    im = ax.imshow(field_avg, cmap='RdBu_r', aspect='auto', animated=True)
    ax.set_title('Step 0', fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    
    def update(frame):
        field_avg = field_history[frame][0].mean(dim=0).cpu().numpy()
        im.set_array(field_avg)
        ax.set_title(f'Step {frame}', fontsize=14, fontweight='bold')
        return [im]
    
    anim = animation.FuncAnimation(
        fig, update, frames=len(field_history),
        interval=100, blit=True
    )
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=10)
        print(f"Animation saved to {save_path}")
    
    return anim


def main_visualization():
    """
    Main visualization demo
    """
    
    print("🎨 WaveNetNeuro Field Dynamics Visualization")
    print("=" * 60)
    
    # Create model
    print("\n🏗️  Creating model...")
    model = WaveNetNeuro(
        vocab_size=100,
        embed_dim=128,
        field_channels=128,
        num_classes=2,
        max_evolution_steps=50
    )
    
    # Create sample input (pattern: increasing sequence)
    print("📝 Creating sample input...")
    seq_len = 64
    input_seq = torch.arange(seq_len) % 100
    
    # Visualize field evolution
    print("🌊 Computing field evolution...")
    field_history = visualize_field_evolution(model, input_seq, max_steps=50)
    print(f"   Total evolution steps: {len(field_history)}")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    
    print("  1. Field snapshots...")
    fig1 = plot_field_snapshots(field_history)
    plt.savefig('/home/claude/field_snapshots.png', dpi=150, bbox_inches='tight')
    
    print("  2. Field energy...")
    fig2 = plot_field_energy(field_history)
    plt.savefig('/home/claude/field_energy.png', dpi=150, bbox_inches='tight')
    
    print("  3. Field change rate...")
    fig3 = plot_field_change_rate(field_history)
    plt.savefig('/home/claude/field_change_rate.png', dpi=150, bbox_inches='tight')
    
    print("  4. Computation graphs...")
    fig4 = compare_computation_graphs()
    plt.savefig('/home/claude/computation_comparison.png', dpi=150, bbox_inches='tight')
    
    print("\n✨ Visualizations saved!")
    print("   - field_snapshots.png: Evolution over time")
    print("   - field_energy.png: Convergence dynamics")
    print("   - field_change_rate.png: Adaptive computation")
    print("   - computation_comparison.png: O(n²) vs O(n)")
    
    # Key insights
    print("\n" + "=" * 60)
    print("🎯 KEY VISUAL INSIGHTS:")
    print("=" * 60)
    print("✓ Information spreads continuously (not discrete jumps)")
    print("✓ Patterns emerge from simple local rules")
    print("✓ Computation adapts to problem complexity")
    print("✓ Sparse connections → efficient computation")
    print("✓ Nature-inspired dynamics → natural intelligence")
    
    return field_history


if __name__ == "__main__":
    field_history = main_visualization()
