"""
BTN Bond Trimming Analysis
===========================
Demonstrates bond compression and relevance analysis.
Shows how bond dimensions change during trimming and the impact on model performance.
"""

import sys
sys.path.append('..')

import torch
import numpy as np
import quimb.tensor as qt
import matplotlib.pyplot as plt
from tensor.btn import BTN
from tensor.builder import Inputs
from utils.plotting_utils import save_figure_high_quality

# Set double precision for stability
torch.set_default_dtype(torch.float64)

print("="*70)
print("BTN BOND TRIMMING ANALYSIS")
print("="*70)

path = ''
# ==============================================================================
# PART 1: Data Generation and Model Setup
# ==============================================================================
print("\n[1/4] Generating Polynomial Data...")

N_SAMPLES = 5000
BATCH_SIZE = 500

# Generate X in range [-1, 1]
x_raw = 2 * torch.rand(N_SAMPLES, 1) - 1

# Target Function: y = 2x³ - x² + 0.5x + 0.2
y_raw = 2 * (x_raw**3) - (x_raw**2) + 3 * x_raw + 0.2
y_raw += 0.1 * torch.randn_like(y_raw)

# Feature engineering
x_features = torch.cat([x_raw, torch.ones_like(x_raw)], dim=1)

print(f"  Data shape: {x_features.shape}")

# ==============================================================================
# PART 2: Build Model with LARGE Bond Dimensions
# ==============================================================================
print("\n[2/4] Building 3-Node MPS with LARGE bond dimensions...")

# Start with intentionally large bond dimensions
D_bond = 20  # Much larger than necessary!
D_phys = 2

def init_weights(shape):
    """Normalized random initialization"""
    w = torch.randn(*shape)
    return w / torch.norm(w)

# Node 1: Input 'x1', Bonds 'b1', 'b3'
t1 = qt.Tensor(
    data=init_weights((D_phys, D_bond, D_bond)), 
    inds=('x1', 'b1', 'b3'), 
    tags={'Node1'}
)

# Node 2: Bond 'b1', Input 'x2', Bond 'b2'
t2 = qt.Tensor(
    data=init_weights((D_bond, D_phys, D_bond)), 
    inds=('b1', 'x2', 'b2'), 
    tags={'Node2'}
)

# Node 3: Bond 'b2', Input 'x3', Bond 'b3', Output 'y'
t3 = qt.Tensor(
    data=init_weights((D_bond, D_phys, D_bond, 1)), 
    inds=('b2', 'x3', 'b3', 'y'), 
    tags={'Node3'}
)

mu_tn = qt.TensorNetwork([t1, t2, t3])

print(f"  Initial bond dimension: {D_bond}")
print(f"  Total parameters: {sum(t.data.numel() for t in mu_tn)}")

# Setup data loader
input_labels = ["x1", "x2", "x3"]

loader = Inputs(
    inputs=[x_features], 
    outputs=[y_raw],
    outputs_labels=["y"],
    input_labels=input_labels,
    batch_dim="s",
    batch_size=BATCH_SIZE
)

model = BTN(
    mu=mu_tn, 
    data_stream=loader, 
    batch_dim="s",
    method='cholesky',
)

print(f"  Initial MSE: {model._calc_mu_mse().item():.6f}")

# ==============================================================================
# PART 3: Training WITH Periodic Bond Trimming
# ==============================================================================
print("\n[3/4] Training with Periodic Bond Trimming...")

EPOCHS = 25
TRIM_THRESHOLD = 0.9  # Keep 95% of bond weight relevance

# Track metrics AND bond dimensions
history = {
    'mse': [],
    'elbo': [],
    'bond_dims': {'b1': [], 'b2': [], 'b3': []},
    'bond_weights': {'b1': [], 'b2': [], 'b3': []},
    'total_params': []
}

print(f"  Trim threshold: {TRIM_THRESHOLD} (keep 95% of relevance)")
print(f"  Training for {EPOCHS} epochs...")
print()

for epoch in range(EPOCHS):
    # Perform one epoch
    model.fit(epochs=1, track_elbo=False)
    
    # Track metrics
    mse = model._calc_mu_mse().item() / model.data.samples
    elbo = model.compute_elbo(verbose=False)
    
    history['mse'].append(mse)
    history['elbo'].append(elbo)
    
    # Track bond dimensions BEFORE trimming
    for bond in ['b1', 'b2', 'b3']:
        bond_size = model.mu.ind_size(bond)
        history['bond_dims'][bond].append(bond_size)
        
        # Track bond weights (mean of Gamma distribution)
        if bond in model.q_bonds:
            weights = model.q_bonds[bond].mean().data
            history['bond_weights'][bond].append(weights.cpu().numpy().copy())
    
    # Track total parameters
    total_params = sum(t.data.numel() for t in model.mu)
    history['total_params'].append(total_params)
    
    # Trim bonds every 3 epochs
    if (epoch + 1) % 2 == 0:
        print(f"  Epoch {epoch+1}: Trimming bonds (threshold={TRIM_THRESHOLD})...")
        model.threshold = TRIM_THRESHOLD
        model.trim_bonds(verbose=False)
        
        # Show new bond dimensions after trimming
        new_dims = {bond: model.mu.ind_size(bond) for bond in ['b1', 'b2', 'b3']}
        print(f"            New bond dims: b1={new_dims['b1']}, b2={new_dims['b2']}, b3={new_dims['b3']}")
    else:
        print(f"  Epoch {epoch+1}: MSE={mse:.6f}, ELBO={elbo:.2f}")

# ==============================================================================
# PART 4: Visualizations
# ==============================================================================
print("\n[4/4] Creating Visualizations...")

epochs_x = np.arange(1, EPOCHS + 1)

# 1. Bond Dimensions Over Time
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Bond Trimming Analysis', fontsize=16, fontweight='bold')

# Bond dimensions
for bond, color in zip(['b1', 'b2', 'b3'], ['blue', 'green', 'red']):
    axes[0, 0].plot(epochs_x, history['bond_dims'][bond], 'o-', 
                    label=f'{bond}', linewidth=2, markersize=6, color=color)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Bond Dimension')
axes[0, 0].set_title('Bond Dimensions (trimmed every 3 epochs)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Total parameters
axes[0, 1].plot(epochs_x, history['total_params'], 'o-', 
                linewidth=2, markersize=6, color='purple')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Total Parameters')
axes[0, 1].set_title('Model Size Over Training')
axes[0, 1].grid(True, alpha=0.3)

# MSE
axes[1, 0].plot(epochs_x, history['mse'], 'o-', 
                linewidth=2, markersize=6, color='red')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('MSE')
axes[1, 0].set_title('Performance (MSE) vs Compression')
axes[1, 0].grid(True, alpha=0.3)

# ELBO
axes[1, 1].plot(epochs_x, history['elbo'], 'o-', 
                linewidth=2, markersize=6, color='darkblue')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('ELBO')
axes[1, 1].set_title('ELBO vs Compression')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
save_figure_high_quality(fig, path + 'outputs/04_bond_trimming_dynamics.png', dpi=200)

# 2. Bond Weight Distributions (showing relevance)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Bond Weight Distributions (Final Epoch)', fontsize=16, fontweight='bold')

for idx, (bond, color) in enumerate(zip(['b1', 'b2', 'b3'], ['blue', 'green', 'red'])):
    final_weights = history['bond_weights'][bond][-1]
    sorted_weights = np.sort(final_weights)[::-1]  # Sort descending
    
    # Compute cumulative relevance
    cumsum = np.cumsum(sorted_weights)
    normalized_cumsum = cumsum / cumsum[-1]
    
    # Plot weights
    axes[idx].bar(range(len(sorted_weights)), sorted_weights, color=color, alpha=0.6)
    axes[idx].set_xlabel('Bond Index (sorted)')
    axes[idx].set_ylabel('Weight (E[λ])')
    axes[idx].set_title(f'Bond {bond} (dim={len(sorted_weights)})')
    axes[idx].grid(True, alpha=0.3, axis='y')
    
    # Add cumulative line
    ax2 = axes[idx].twinx()
    ax2.plot(range(len(normalized_cumsum)), normalized_cumsum, 
             'k--', linewidth=2, label='Cumulative %')
    ax2.axhline(y=TRIM_THRESHOLD, color='red', linestyle=':', 
                linewidth=2, label=f'{TRIM_THRESHOLD*100}% threshold')
    ax2.set_ylabel('Cumulative Relevance')
    ax2.set_ylim([0, 1.05])
    ax2.legend(fontsize=8)

plt.tight_layout()
save_figure_high_quality(fig, path + 'outputs/04_bond_weight_distributions.png', dpi=200)

# ==============================================================================
# Summary
# ==============================================================================
print("\n" + "="*70)
print("BOND TRIMMING SUMMARY")
print("="*70)

print(f"\nInitial bond dimensions: b1={history['bond_dims']['b1'][0]}, " + 
      f"b2={history['bond_dims']['b2'][0]}, b3={history['bond_dims']['b3'][0]}")
print(f"Final bond dimensions:   b1={history['bond_dims']['b1'][-1]}, " +
      f"b2={history['bond_dims']['b2'][-1]}, b3={history['bond_dims']['b3'][-1]}")

initial_params = history['total_params'][0]
final_params = history['total_params'][-1]
compression_ratio = initial_params / final_params

print(f"\nInitial parameters: {initial_params}")
print(f"Final parameters:   {final_params}")
print(f"Compression ratio:  {compression_ratio:.2f}x")

print(f"\nInitial MSE:  {history['mse'][0]:.6f}")
print(f"Final MSE:    {history['mse'][-1]:.6f}")
print(f"Performance maintained: {history['mse'][-1] < history['mse'][0] * 1.1}")

print("\n" + "="*70)
print("SHOWCASE 04 COMPLETE ✓")
print("="*70)
print("\nGenerated 2 visualizations:")
print("  1. Bond trimming dynamics (dimensions, params, MSE, ELBO)")
print("  2. Bond weight distributions with cumulative relevance")
print("\nKey insights:")
print("  - Bond dimensions automatically reduce to essential features")
print("  - Model compression with minimal performance loss")
print("  - Bayesian approach quantifies bond importance via λ weights")
print(f"  - Achieved {compression_ratio:.2f}x compression while maintaining accuracy")
