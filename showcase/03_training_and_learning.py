"""
BTN Training and Learning Dynamics
===================================
Demonstrates training process with ELBO tracking, convergence visualization,
and prediction uncertainty quantification with epoch-by-epoch evolution.
"""

import sys
sys.path.append('..')

import torch
import numpy as np
import quimb.tensor as qt
import matplotlib.pyplot as plt
import os
from tensor.btn import BTN
from tensor.builder import Inputs
from utils.plotting_utils import (
    plot_elbo_components_stacked,
    save_figure_high_quality
)

# Set double precision for stability
torch.set_default_dtype(torch.float64)

print("="*70)
print("BTN TRAINING AND LEARNING DYNAMICS")
print("="*70)

# ==============================================================================
# PART 1: Data Generation - Oscillating Function
# ==============================================================================
print("\n[1/6] Generating Oscillating Data...")

N_SAMPLES = 5000
BATCH_SIZE = 500

# Generate X in range [-1, 1]
x_raw = 2 * torch.rand(N_SAMPLES, 1) - 1

# Target Function: Gentle oscillation
# y = 0.5*sin(1.5πx) + 0.4x² - 0.2x + 0.3
y_raw = 0.5 * torch.sin(1.5 * np.pi * x_raw) + 0.4 * (x_raw**2) - 0.2 * x_raw + 0.3

# Add noise
y_raw += 0.08 * torch.randn_like(y_raw)

# Feature engineering: [x, 1] allows learning both oscillations and polynomial terms
x_features = torch.cat([x_raw, torch.ones_like(x_raw)], dim=1)

print(f"  Data shape: {x_features.shape} (N, 2) -> [x, 1]")
print(f"  Target: y = 0.5*sin(1.5πx) + 0.4x² - 0.2x + 0.3 + noise")
print(f"  Function has ~0.75 oscillations in [-1, 1]")

# ==============================================================================
# PART 2: Build 5-Node MPS
# ==============================================================================
print("\n[2/6] Building 5-Node MPS Architecture...")

NUM_NODES = 5
D_bond = 6
D_phys = 2

def init_weights(shape):
    """Normalized random initialization"""
    w = torch.randn(*shape)
    return w / torch.norm(w)

# Build 6-node chain
nodes = []
for i in range(NUM_NODES):
    if i == 0:
        # First node: (x, b1, bN)
        t = qt.Tensor(
            data=init_weights((D_phys, D_bond, D_bond)), 
            inds=(f'x{i+1}', 'b1', f'b{NUM_NODES}'), 
            tags={f'Node{i+1}'}
        )
    elif i == NUM_NODES - 1:
        # Last node: (b_{N-1}, x_N, b_N, y)
        t = qt.Tensor(
            data=init_weights((D_bond, D_phys, D_bond, 1)), 
            inds=(f'b{i}', f'x{i+1}', f'b{NUM_NODES}', 'y'), 
            tags={f'Node{i+1}'}
        )
    else:
        # Middle nodes: (b_i, x_{i+1}, b_{i+1})
        t = qt.Tensor(
            data=init_weights((D_bond, D_phys, D_bond)), 
            inds=(f'b{i}', f'x{i+1}', f'b{i+1}'), 
            tags={f'Node{i+1}'}
        )
    nodes.append(t)

mu_tn = qt.TensorNetwork(nodes)

print(f"  Architecture: {NUM_NODES}-node MPS")
print(f"  Bond dimension: {D_bond}")
print(f"  Physical dimension: {D_phys} (features: [x, 1])")

total_params = sum(t.data.numel() for t in mu_tn)
print(f"  Total parameters: {total_params}")

# ==============================================================================
# PART 3: Setup Data Loader and BTN Model
# ==============================================================================
print("\n[3/6] Initializing BTN Model...")

input_labels = [f"x{i+1}" for i in range(NUM_NODES)]

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

# Initial metrics
mse_result = model._calc_mu_mse()
if hasattr(mse_result, 'item'):
    initial_mse = mse_result.item()
elif hasattr(mse_result, 'data'):
    initial_mse = mse_result.data.item() if hasattr(mse_result.data, 'item') else float(mse_result.data)
else:
    initial_mse = float(mse_result)
print(f"  Initial MSE: {initial_mse:.6f}")

initial_elbo = model.compute_elbo(verbose=False)
print(f"  Initial ELBO: {initial_elbo:.2f}")

# ==============================================================================
# PART 4: Training with Block-by-Block Snapshots
# ==============================================================================
print("\n[4/6] Training with Block-by-Block Visualization...")
print("  Creating snapshots after each block update for GIF animation...")

EPOCHS = 5  # Show all block updates across 5 epochs

# Create output directory for snapshots
snapshot_dir = 'outputs/03_snapshots'
os.makedirs(snapshot_dir, exist_ok=True)

# Test grid for predictions
x_test = torch.linspace(-1.2, 1.2, 300).unsqueeze(1)
x_test_features = torch.cat([x_test, torch.ones_like(x_test)], dim=1)
y_test_true = 0.5 * torch.sin(1.5 * np.pi * x_test) + 0.4 * (x_test**2) - 0.2 * x_test + 0.3

test_loader = Inputs(
    inputs=[x_test_features],
    outputs=[y_test_true],
    outputs_labels=["y"],
    input_labels=input_labels,
    batch_dim="s",
    batch_size=300
)

# Select fixed training sample indices for consistent visualization
np.random.seed(42)
fixed_sample_indices = np.random.choice(len(x_raw), 400, replace=False)

# Helper function to create snapshot
def create_snapshot(snapshot_num, epoch, block_idx, description):
    """Create and save prediction snapshot"""
    # Get predictions
    mu_pred = model.forward(model.mu, test_loader.data_mu, sum_over_batch=False, sum_over_output=False)
    if hasattr(mu_pred, 'data'):
        mu_pred_data = torch.as_tensor(mu_pred.data).squeeze()
    else:
        mu_pred_data = torch.as_tensor(mu_pred).squeeze()
    
    # Get uncertainty
    sigma_pred = model.forward(model.sigma, test_loader.data_sigma, sum_over_batch=False, sum_over_output=False)
    if hasattr(sigma_pred, 'data'):
        sigma_pred_data = torch.as_tensor(sigma_pred.data).squeeze()
    else:
        sigma_pred_data = torch.as_tensor(sigma_pred).squeeze()
    
    if sigma_pred_data.dim() == 2:
        epistemic_var = torch.diagonal(sigma_pred_data, dim1=0, dim2=1)
    else:
        epistemic_var = sigma_pred_data
    
    tau_mean = model.get_tau_mean()
    aleatoric_var = 1.0 / tau_mean
    
    epistemic_std = torch.sqrt(torch.abs(epistemic_var))
    total_std = torch.sqrt(torch.abs(epistemic_var + aleatoric_var))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot training data (FIXED samples - same across all frames)
    ax.scatter(x_raw[fixed_sample_indices].squeeze().numpy(), 
              y_raw[fixed_sample_indices].squeeze().numpy(), 
              alpha=0.25, s=12, color='gray', label='Training data', zorder=1)
    
    # Plot true function
    ax.plot(x_test.squeeze().numpy(), y_test_true.squeeze().numpy(), 
            'k-', linewidth=3, label='True function', alpha=0.8, zorder=3)
    
    # Plot predictions
    ax.plot(x_test.squeeze().numpy(), mu_pred_data.numpy(), 
            'b-', linewidth=2.5, label='BTN mean (μ)', zorder=4)
    
    # Total uncertainty
    ax.fill_between(x_test.squeeze().numpy(), 
                    (mu_pred_data - 2*total_std).numpy(),
                    (mu_pred_data + 2*total_std).numpy(),
                    alpha=0.25, color='red', label='±2σ total', zorder=2)
    
    # Epistemic uncertainty
    ax.fill_between(x_test.squeeze().numpy(), 
                    (mu_pred_data - 2*epistemic_std).numpy(),
                    (mu_pred_data + 2*epistemic_std).numpy(),
                    alpha=0.4, color='blue', label='±2σ epistemic (Σ)', zorder=2)
    
    # Add vertical bars at x=-1 and x=1
    ax.axvline(x=-1, color='red', linestyle='--', linewidth=2, alpha=0.6, label='Domain boundary', zorder=5)
    ax.axvline(x=1, color='red', linestyle='--', linewidth=2, alpha=0.6, zorder=5)
    
    ax.set_xlabel('x', fontsize=13)
    ax.set_ylabel('y', fontsize=13)
    ax.set_title(f'Epoch {epoch}/{EPOCHS} | {description}\ny = 0.5*sin(1.5πx) + 0.4x² - 0.2x + 0.3', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-2.5, 2.5)
    
    plt.tight_layout()
    plt.savefig(f'{snapshot_dir}/frame_{snapshot_num:03d}.png', dpi=120, bbox_inches='tight')
    plt.close()

# Get bond and node lists
bonds = [i for i in model.mu.ind_map if i != model.batch_dim]
nodes = list(model.mu.tag_map.keys())

snapshot_count = 0

# Initial snapshot
create_snapshot(snapshot_count, 0, 0, "Initial state (random)")
snapshot_count += 1
print(f"  Snapshot {snapshot_count:3d}: Initial")

# Training loop with block-by-block snapshots
# for epoch in range(EPOCHS):
#     print(f"\n  Epoch {epoch+1}/{EPOCHS}:")
    
#     # Update each node (Sigma then Mu)
#     for node_idx, node_tag in enumerate(nodes):
#         # Update Sigma
#         model.update_sigma_node(node_tag)
#         create_snapshot(snapshot_count, epoch+1, node_idx, f"After Σ update: {node_tag}")
#         snapshot_count += 1
#         print(f"    Snapshot {snapshot_count:3d}: Σ[{node_tag}] updated")
        
#         # Update Mu
#         model.update_mu_node(node_tag)
#         create_snapshot(snapshot_count, epoch+1, node_idx, f"After μ update: {node_tag}")
#         snapshot_count += 1
#         print(f"    Snapshot {snapshot_count:3d}: μ[{node_tag}] updated")
    
#     # Update bonds
#     for bond_tag in bonds:
#         model.update_bond(bond_tag)
    
#     # Update tau
#     model.update_tau()
    
    # Snapshot after full epoch
    # create_snapshot(snapshot_count, epoch+1, len(nodes), f"Epoch {epoch+1} complete (after τ update)")
    # snapshot_count += 1
    # print(f"    Snapshot {snapshot_count:3d}: Epoch {epoch+1} complete")

print(f"\n  Created {snapshot_count} snapshots in {snapshot_dir}/")
print(f"  To create GIF: convert -delay 30 -loop 0 {snapshot_dir}/frame_*.png outputs/03_training_evolution.gif")
print(f"  Shows evolution: {EPOCHS} epochs × {NUM_NODES} nodes × 2 updates/node = {EPOCHS * NUM_NODES * 2} block updates")

# ==============================================================================
# PART 5: Final Training with Full Metrics
# ==============================================================================
print("\n[5/6] Running additional epochs with full metrics tracking...")

history = model.fit(epochs=10, track_elbo=True, track_kl_components=True)

print(f"\n  Final metrics:")
print(f"  Bond KL: {history['bond_kl'][-1]:.4f}")
print(f"  Node KL: {history['node_kl'][-1]:.4f}")
print(f"  Tau KL:  {history['tau_kl'][-1]:.4f}")
print(f"  E[log p(y|x)]: {history['exp_log_lik'][-1]:.4f}")

# ==============================================================================
# PART 6: Final Visualizations
# ==============================================================================
print("\n[6/6] Creating Final Visualizations...")

# 1. ELBO Components Stacked
fig1, ax1 = plt.subplots(1, 1, figsize=(12, 7))
plot_elbo_components_stacked(history, ax=ax1)
plt.tight_layout()
save_figure_high_quality(fig1, 'outputs/03_elbo_components.png', dpi=200)

# 2. Training Metrics Grid
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('Training Dynamics', fontsize=16, fontweight='bold')

epochs_x = np.array(history['epoch'])

# ELBO
axes[0, 0].plot(epochs_x, history['elbo'], 'o-', linewidth=2, markersize=6, color='darkblue')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('ELBO')
axes[0, 0].set_title('ELBO (should increase)')
axes[0, 0].grid(True, alpha=0.3)

# MSE
axes[0, 1].plot(epochs_x, history['mse'], 'o-', linewidth=2, markersize=6, color='red')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('MSE')
axes[0, 1].set_title('Mean Squared Error')
axes[0, 1].grid(True, alpha=0.3)

# KL Components
axes[1, 0].plot(epochs_x, history['bond_kl'], 'o-', label='Bond KL', linewidth=2, markersize=5)
axes[1, 0].plot(epochs_x, history['node_kl'], 's-', label='Node KL', linewidth=2, markersize=5)
axes[1, 0].plot(epochs_x, history['tau_kl'], '^-', label='Tau KL', linewidth=2, markersize=5)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('KL Divergence')
axes[1, 0].set_title('KL Components Over Time')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Expected Log Likelihood
axes[1, 1].plot(epochs_x, history['exp_log_lik'], 'o-', linewidth=2, markersize=6, color='green')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('E[log p(y|x)]')
axes[1, 1].set_title('Expected Log Likelihood')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
save_figure_high_quality(fig2, 'outputs/03_training_metrics.png', dpi=200)

# 3. Final Predictions with Uncertainty
mu_pred = model.forward(model.mu, test_loader.data_mu, sum_over_batch=False, sum_over_output=False)
sigma_pred = model.forward(model.sigma, test_loader.data_sigma, sum_over_batch=False, sum_over_output=False)

if hasattr(mu_pred, 'data'):
    mu_pred_data = torch.as_tensor(mu_pred.data).squeeze()
else:
    mu_pred_data = torch.as_tensor(mu_pred).squeeze()

if hasattr(sigma_pred, 'data'):
    sigma_pred_data = torch.as_tensor(sigma_pred.data).squeeze()
else:
    sigma_pred_data = torch.as_tensor(sigma_pred).squeeze()

if sigma_pred_data.dim() == 2:
    epistemic_var = torch.diagonal(sigma_pred_data, dim1=0, dim2=1)
else:
    epistemic_var = sigma_pred_data

tau_mean = model.get_tau_mean()
aleatoric_var = 1.0 / tau_mean
aleatoric_std = torch.sqrt(torch.tensor(aleatoric_var))

epistemic_std = torch.sqrt(torch.abs(epistemic_var))
total_var = epistemic_var + aleatoric_var
total_std = torch.sqrt(torch.abs(total_var))

print(f"  Mean epistemic std: {epistemic_std.mean():.5f}")
print(f"  Aleatoric std (from τ): {aleatoric_std:.5f}")
print(f"  Mean total std: {total_std.mean():.5f}")

fig3, axes = plt.subplots(1, 2, figsize=(16, 6))
fig3.suptitle('BTN Final Predictions - Oscillating Function', fontsize=16, fontweight='bold')

# LEFT: Predictions
ax = axes[0]
ax.plot(x_test.squeeze().numpy(), y_test_true.squeeze().numpy(), 
        'k--', linewidth=2, label='True function', alpha=0.7)
ax.plot(x_test.squeeze().numpy(), mu_pred_data.numpy(), 
        'b-', linewidth=2, label='BTN mean')
ax.fill_between(x_test.squeeze().numpy(), 
                (mu_pred_data - 2*total_std).numpy(),
                (mu_pred_data + 2*total_std).numpy(),
                alpha=0.3, color='red', label='±2σ total')
ax.fill_between(x_test.squeeze().numpy(), 
                (mu_pred_data - 2*epistemic_std).numpy(),
                (mu_pred_data + 2*epistemic_std).numpy(),
                alpha=0.5, color='blue', label='±2σ epistemic')

# Training data sample
indices = np.random.choice(len(x_raw), 400, replace=False)
ax.scatter(x_raw[indices].squeeze().numpy(), y_raw[indices].squeeze().numpy(), 
          alpha=0.15, s=12, color='gray', label='Training data')

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Predictions: y = sin(5πx) + 0.3x² - 0.2x', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# RIGHT: Residuals
ax = axes[1]
residuals = (mu_pred_data - y_test_true.squeeze()).numpy()
ax.scatter(x_test.squeeze().numpy(), residuals, alpha=0.5, s=10, c='purple', label='Residuals')
ax.axhline(y=0, color='k', linestyle='--', linewidth=2)
ax.fill_between(x_test.squeeze().numpy(),
                (-2*total_std).numpy(),
                (2*total_std).numpy(),
                alpha=0.2, color='red', label='±2σ expected')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('Residual', fontsize=12)
ax.set_title(f'Residuals (RMSE={np.sqrt(np.mean(residuals**2)):.4f})', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_figure_high_quality(fig3, 'outputs/03_predictions_uncertainty.png', dpi=200)

# ==============================================================================
# Summary
# ==============================================================================
print("\n" + "="*70)
print("TRAINING SUMMARY")
print("="*70)

final_mse = history['mse'][-1]
final_elbo = history['elbo'][-1]

print(f"\nFinal ELBO:    {final_elbo:+.2f}")
print(f"Final MSE:     {final_mse:.6f}")

rmse = np.sqrt(np.mean(residuals**2))
print(f"Final RMSE:    {rmse:.6f}")

if final_mse < 0.02:  # More relaxed for oscillating function
    print("\n✅ SUCCESS: Model learned the oscillating function!")
else:
    print("\n⚠️  Model may need more epochs or tuning")

print("\n" + "="*70)
print("SHOWCASE 03 COMPLETE ✓")
print("="*70)
print("\nGenerated visualizations:")
print(f"  1. {snapshot_count} block-by-block snapshots in {snapshot_dir}/")
print("  2. ELBO components stacked")
print("  3. Training metrics grid")
print("  4. Final predictions with Bayesian uncertainty")
print("\nTo create GIF animation:")
print(f"  convert -delay 50 -loop 0 {snapshot_dir}/epoch_*.png outputs/03_training_evolution.gif")
print("\nKey insights:")
print("  - 6-node MPS can capture oscillating patterns")
print("  - Epistemic uncertainty high where data is sparse")
print("  - Aleatoric uncertainty from noise level (1/√τ)")
print("  - Watch the GIF to see learning dynamics!")
