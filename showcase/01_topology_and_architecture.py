"""
BTN Architecture Showcase
=========================
Demonstrates different tensor network topologies and the BTNBuilder process.
"""

import sys
sys.path.append('..')

import quimb.tensor as qt
import torch
import matplotlib.pyplot as plt
import numpy as np
from tensor.btn import BTN
from tensor.builder import Inputs

# Set double precision for stability
torch.set_default_dtype(torch.float64)

print("="*70)
print("BTN ARCHITECTURE SHOWCASE")
print("="*70)

# ==============================================================================
# PART A: Standard MPS Architecture
# ==============================================================================
print("\n[1/5] Building Standard MPS Architecture...")

bond_dim = 4
input_dim = 2
output_dim = 1

# Helper to initialize weights
def weights(shape):
    w = torch.randn(*shape)
    return w / torch.norm(w)

# Create three-node chain
t1 = qt.Tensor(data=weights((input_dim, bond_dim, bond_dim)), 
               inds=('x1', 'b1', 'b3'), tags={'Node1', 'input'})
t2 = qt.Tensor(data=weights((bond_dim, input_dim, bond_dim)), 
               inds=('b1', 'x2', 'b2'), tags={'Node2', 'hidden'})
t3 = qt.Tensor(data=weights((bond_dim, input_dim, bond_dim, output_dim)), 
               inds=('b2', 'x3', 'b3', 'y'), tags={'Node3', 'output'})

mu_standard = qt.TensorNetwork([t1, t2, t3])

total_params = sum(t.data.numel() if hasattr(t.data, 'numel') else t.data.size for t in mu_standard)
print(f"  Nodes: {len(mu_standard.tensors)}, Params: {total_params}")

# Print network structure
print("\n  Network Structure:")
for i, tensor in enumerate(mu_standard.tensors, 1):
    print(f"    Tensor {i}: shape={tensor.shape}, indices={tensor.inds}, tags={tensor.tags}")

# Let quimb draw - no constraints
mu_standard.draw(
    color=['input', 'hidden', 'output'],
    custom_colors=['blue', 'gray', 'red'],
    show_inds='bond-size',
    show_tags=True,
    title="Standard MPS Architecture",
    figsize=(14, 10),
    layout='kamada_kawai'
)
# plt.savefig('showcase/outputs/01_standard_mps.png')
plt.close()
print("  ✓ Saved: showcase/outputs/01_standard_mps.png")

# ==============================================================================
# PART B: Binary Tree Architecture
# ==============================================================================
print("\n[2/5] Building Binary Tree Architecture...")

# Leaf nodes
t_l1 = qt.Tensor(data=weights((input_dim, bond_dim)), inds=('x1', 'b1'), 
                 tags={'Leaf1', 'depth_2'})
t_l2 = qt.Tensor(data=weights((input_dim, bond_dim)), inds=('x2', 'b2'), 
                 tags={'Leaf2', 'depth_2'})
t_l3 = qt.Tensor(data=weights((input_dim, bond_dim)), inds=('x3', 'b3'), 
                 tags={'Leaf3', 'depth_2'})

# Internal nodes
t_i1 = qt.Tensor(data=weights((bond_dim, bond_dim, bond_dim)), 
                 inds=('b1', 'b2', 'b4'), tags={'Internal1', 'depth_1'})
t_i2 = qt.Tensor(data=weights((bond_dim, input_dim, bond_dim)), 
                 inds=('b3', 'x4', 'b5'), tags={'Internal2', 'depth_1'})

# Root
t_root = qt.Tensor(data=weights((bond_dim, bond_dim, output_dim)), 
                   inds=('b4', 'b5', 'y'), tags={'Root', 'depth_0'})

mu_tree = qt.TensorNetwork([t_l1, t_l2, t_l3, t_i1, t_i2, t_root])

total_params_tree = sum(t.data.numel() if hasattr(t.data, 'numel') else t.data.size for t in mu_tree)
print(f"  Nodes: {len(mu_tree.tensors)}, Params: {total_params_tree}")

# Print tree structure
print("\n  Tree Structure (printed by quimb.draw):")

# Let quimb draw the tree - no constraints
mu_tree.draw(
    color=['depth_2', 'depth_1', 'depth_0'],
    custom_colors=['blue', 'orange', 'red'],
    show_inds='bond-size',
    show_tags=True,
    layout='spring',
    title="Binary Tree Architecture",
    figsize=(14, 10)
)
# plt.savefig('showcase/outputs/01_binary_tree.png', dpi=200)
plt.close()
print("  ✓ Saved: showcase/outputs/01_binary_tree.png")

# ==============================================================================
# PART C: Generate Polynomial Data
# ==============================================================================
print("\n[3/5] Generating polynomial dataset...")

# Polynomial: f(x) = 2x³ - x² + 0.5x + 0.2
n_samples = 1000
torch.manual_seed(42)
x_raw = 2 * torch.rand(n_samples, 1) - 1

# Target
y = 2 * (x_raw**3) - (x_raw**2) + 0.5 * x_raw + 0.2 + 0.05 * torch.randn_like(x_raw)

# Feature engineering: [x, 1]
x_features = torch.cat([x_raw, torch.ones_like(x_raw)], dim=1)

print(f"  Formula: y = 2x³ - x² + 0.5x + 0.2")
print(f"  Samples: {n_samples}, Range: y ∈ [{y.min():.2f}, {y.max():.2f}]")
print(f"  Features: [x, 1] (dimension {x_features.shape[1]})")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.scatter(x_raw.numpy(), y.numpy(), alpha=0.5, s=15, c='blue', edgecolors='none')
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Polynomial Data: y = 2x³ - x² + 0.5x + 0.2', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)

x_plot = torch.linspace(-1, 1, 200).reshape(-1, 1)
y_true = 2 * (x_plot**3) - (x_plot**2) + 0.5 * x_plot + 0.2
ax2 = axes[1]
ax2.plot(x_plot.numpy(), y_true.numpy(), 'r-', linewidth=2.5, label='True Function')
ax2.scatter(x_raw.numpy(), y.numpy(), alpha=0.3, s=10, c='blue', label='Noisy Data')
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('y', fontsize=12)
ax2.set_title('True Polynomial vs Noisy Observations', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig('showcase/outputs/01_polynomial_data.png', dpi=200)
plt.close()
print("  ✓ Saved: showcase/outputs/01_polynomial_data.png")

# ==============================================================================
# PART D: BTNBuilder Process
# ==============================================================================
print("\n[4/5] Demonstrating BTNBuilder process...")

data_loader = Inputs(
    inputs=[x_features],
    outputs=[y],
    input_labels=['x1', 'x2', 'x3'],
    outputs_labels=['y'],
    batch_dim='s',
    batch_size=200
)

print("\n  BEFORE BTNBuilder:")
print(f"    Mu tensors: {[list(t.tags) for t in mu_standard]}")
print(f"    Mu indices: {list(mu_standard.ind_map.keys())}")

a1 = qt.Tensor(weights((3, 3, 3)), inds = ["x1", "f1", "a1"], tags=["Node1"])
b1 = qt.Tensor(weights(( 3, 3)), inds = ["f1", "b1"], tags=["Node2"])
a3 = qt.Tensor(weights((3, 3, 3)), inds = ["x3", "f3", "a2"], tags=["Node3"])
b3 = qt.Tensor(weights(( 3, 3)), inds = [ "f3", "b2"], tags=["Node4"])
a2 = qt.Tensor(weights((3,3,3,3)), inds = ["x2", "a1", "f2", "a2"], tags=["Node5"])
b2 = qt.Tensor(weights((3,3,3)), inds = ["b1", "f2", "b2"], tags=["Node6"])
tn = qt.TensorNetwork([a1, b1, a3, b3, a2, b2])

model = BTN(mu=tn, data_stream=data_loader, batch_dim='s')

print("\n  AFTER BTNBuilder:")
print(f"    Created distributions:")
print(f"      - Prior bonds (p_bonds): {len(model.p_bonds)} bonds → {list(model.p_bonds.keys())}")
print(f"      - Posterior bonds (q_bonds): {len(model.q_bonds)} bonds → {list(model.q_bonds.keys())}")
print(f"      - Prior nodes (p_nodes): {len(model.p_nodes)} nodes → {list(model.p_nodes.keys())}")
print(f"      - Posterior nodes (q_nodes): {len(model.q_nodes)} nodes → {list(model.q_nodes.keys())}")

tau_alpha = model.p_tau.concentration if isinstance(model.p_tau.concentration, (int, float)) else model.p_tau.concentration.item()
tau_beta = model.p_tau.rate if isinstance(model.p_tau.rate, (int, float)) else model.p_tau.rate.item()
print(f"      - Prior/Posterior tau: α={tau_alpha:.1f}, β={tau_beta:.1f}")

print("\n  Bond Dimensions:")
for bond_name in ['b1', 'b2', 'b3', 'x1', 'x2', 'x3', 'y']:
    if bond_name in model.q_bonds:
        dim = len(model.q_bonds[bond_name].mean().data)
        print(f"    {bond_name}: dimension={dim}")

# ==============================================================================
# PART E: Visualize Distributions and Networks
# ==============================================================================
print("\n[5/5] Visualizing distributions and networks...")

# Bond distributions
bond_name = 'b1'
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

p_bond = model.p_bonds[bond_name]
q_bond = model.q_bonds[bond_name]

ax1 = axes[0]
p_mean = p_bond.mean().data.numpy()
ax1.bar(range(len(p_mean)), p_mean, alpha=0.7, color='blue', edgecolor='black')
ax1.set_xlabel('Dimension', fontsize=12)
ax1.set_ylabel('E[λ]', fontsize=12)
ax1.set_title(f"Prior Bond Distribution: {bond_name}", fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

ax2 = axes[1]
q_mean = q_bond.mean().data.numpy()
ax2.bar(range(len(q_mean)), q_mean, alpha=0.7, color='red', edgecolor='black')
ax2.set_xlabel('Dimension', fontsize=12)
ax2.set_ylabel('E[λ]', fontsize=12)
ax2.set_title(f"Posterior Bond Distribution: {bond_name}", fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
# plt.savefig('showcase/outputs/01_bond_distributions.png', dpi=200)
plt.close()
print("  ✓ Saved: showcase/outputs/01_bond_distributions.png")

print(f"\n  Bond '{bond_name}' Statistics:")
print(f"    Prior:     E[λ] mean={p_mean.mean():.4f}, std={p_mean.std():.4f}")
print(f"    Posterior: E[λ] mean={q_mean.mean():.4f}, std={q_mean.std():.4f}")

# Mu network - let quimb draw
print("\n  Drawing μ network...")
model.mu.draw(
    color=['Node1', 'Node2', 'Node3'],
    custom_colors=['#3498db', '#95a5a6', '#e74c3c'],
    show_inds='bond-size',
    show_tags=True,
    title="μ Network (Mean Parameters)",
    figsize=(12, 8),
    layout='kamada_kawai'
)
# plt.savefig('showcase/outputs/01_mu_network.png', dpi=200)
plt.close()
print("  ✓ Saved: showcase/outputs/01_mu_network.png")

# Sigma network - let quimb draw
print("  Drawing Σ network...")
model.sigma.draw(
    color=['Node1', 'Node2', 'Node3'],
    custom_colors=['#aed6f1', '#d5d8dc', '#f1948a'],
    show_inds='bond-size',
    show_tags=True,
    title="Σ Network (Covariance Parameters)",
    figsize=(12, 8),
    layout='kamada_kawai'
)

# plt.savefig('showcase/outputs/01_sigma_network.png', dpi=200)
plt.close()
print("  ✓ Saved: showcase/outputs/01_sigma_network.png")

# Network comparison
mu_params = sum(t.data.numel() for t in model.mu)
sigma_params = sum(t.data.numel() for t in model.sigma)
print(f"\n  Network Comparison:")
print(f"    μ network:  {len(model.mu.tensors)} tensors, {mu_params:,} parameters")
print(f"    Σ network:  {len(model.sigma.tensors)} tensors, {sigma_params:,} parameters")
print(f"    Total: {mu_params + sigma_params:,} parameters")

# ==============================================================================
# Summary
# ==============================================================================
print("\n" + "="*70)
print("SHOWCASE 01 COMPLETE ✓")
print("="*70)
print("\nGenerated 6 visualizations (all drawn by quimb):")
print("  1. 01_standard_mps.png         - Standard MPS topology")
print("  2. 01_binary_tree.png          - Binary tree topology")
print("  3. 01_polynomial_data.png      - Training data")
print("  4. 01_bond_distributions.png   - Prior/Posterior bonds")
print("  5. 01_mu_network.png           - μ network")
print("  6. 01_sigma_network.png        - Σ network")
print("\nKey Takeaways:")
print("  • BTNs support flexible topologies (MPS, trees, etc.)")
print("  • BTNBuilder automatically creates prior/posterior distributions")
print("  • Each bond has a Gamma distribution over its dimensions")
print("  • Separate networks for mean (μ) and covariance (Σ)")
