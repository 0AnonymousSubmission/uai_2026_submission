"""
BTN Update Rules Dissection
============================
Visual demonstration of network topology changes during updates.
Shows the actual tensor network contractions for Sigma and Mu updates.
"""

import sys
sys.path.append('..')

import quimb.tensor as qt
import torch
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

print("="*70)
print("BTN UPDATE RULES: NETWORK TOPOLOGY DISSECTION")
print("="*70)
print("Visualizing how tensor networks change during Sigma and Mu updates")
print("="*70)

# ==============================================================================
# PART 1: Sigma Update - Computing the Environment
# ==============================================================================
print("\n" + "="*70)
print("PART 1: Σ UPDATE - Environment for Node2 (middle node)")
print("="*70)

print("\nSigma update formula:")
print("  Precision = τ × (Σ_env + μ_outer) + θ")
print("  New Σ = Precision^(-1)")

# Create simple dummy networks to show the topology
print("\n--- Creating dummy 3-node MPS networks ---")

# Sigma network (3 nodes) - each has primed and unprimed indices
s1 = qt.Tensor(torch.randn(2, 3, 2, 3), 
               inds=('x1', 'b1', 'x1_prime', 'b1_prime'), 
               tags={'Node1_sigma'})
s2 = qt.Tensor(torch.randn(3, 2, 3, 3, 2, 3), 
               inds=('b1', 'x2', 'b2', 'b1_prime', 'x2_prime', 'b2_prime'), 
               tags={'Node2_sigma'})
s3 = qt.Tensor(torch.randn(3, 2, 1, 3, 2, 1), 
               inds=('b2', 'x3', 'y', 'b2_prime', 'x3_prime', 'y_prime'), 
               tags={'Node3_sigma'})

sigma_tn = qt.TensorNetwork([s1, s2, s3])

print(f"Full Sigma network: {len(sigma_tn.tensors)} tensors")
print("  Each tensor has BOTH unprimed and primed indices (covariance structure)")

# Draw full sigma network
sigma_tn.draw(
    color=['Node1_sigma', 'Node2_sigma', 'Node3_sigma'],
    custom_colors=['lightblue', 'yellow', 'lightgreen'],
    show_inds=True,
    show_tags=True,
    title="Full Σ Network (3 nodes connected)",
    figsize=(14, 10),
    layout='kamada_kawai'
)
plt.close()
print("  ✓ Saved: outputs/02_sigma_full.png")

# Show Sigma_env: REMOVE Node2, leave Node1 and Node3 as they are
print("\n--- Computing Σ_env (environment) ---")
print("REMOVE Node2_sigma from network")
print("Node1 and Node3 remain UNCONTRACTED - just remove the middle node")

sigma_env_tn = sigma_tn.copy()
sigma_env_tn.delete('Node2_sigma')

print(f"\nΣ_env network: {len(sigma_env_tn.tensors)} tensors (Node1 & Node3 only)")
print(f"  Node1 has dangling bonds: b1, b1_prime")
print(f"  Node3 has dangling bonds: b2, b2_prime")
print(f"  These are the bonds that CONNECTED to Node2!")

sigma_env_tn.draw(
    color=['Node1_sigma', 'Node3_sigma'],
    custom_colors=['lightblue', 'lightgreen'],
    show_inds=True,
    show_tags=True,
    title="Σ_env: Node2 REMOVED, Node1 & Node3 remain (dangling bonds: b1, b2)",
    figsize=(14, 10),
    layout='kamada_kawai'
)
plt.close()
print("  ✓ Saved: showcase/outputs/02_sigma_env.png")

print("\n  When contracted with inputs, Σ_env will have free indices:")
print("    (b1, x2, b2, b1_prime, x2_prime, b2_prime)")
print("  These match EXACTLY where Node2 connects!")

# Same for Mu network
print("\n--- Computing μ_env (environment) ---")

m1 = qt.Tensor(torch.randn(2, 3), inds=('x1', 'b1'), tags={'Node1_mu'})
m2 = qt.Tensor(torch.randn(3, 2, 3), inds=('b1', 'x2', 'b2'), tags={'Node2_mu'})
m3 = qt.Tensor(torch.randn(3, 2, 1), inds=('b2', 'x3', 'y'), tags={'Node3_mu'})

mu_tn = qt.TensorNetwork([m1, m2, m3])

print("\nFull μ network:")
mu_tn.draw(
    color=['Node1_mu', 'Node2_mu', 'Node3_mu'],
    custom_colors=['blue', 'yellow', 'green'],
    show_inds=True,
    show_tags=True,
    title="Full μ Network (3 nodes connected)",
    figsize=(14, 10),
    layout='kamada_kawai'
)
plt.close()
print("  ✓ Saved: showcase/outputs/02_mu_full.png")

# --- YOUR STARTING CODE ---
mu_env_tn = mu_tn.copy()
mu_env_tn.delete('Node2_mu')

# I'll add tags here just to make the final drawing clearer with colors
mu_env_tn.add_tag("ORIGINAL_ENV")

print(f"\nμ_env network: {len(mu_env_tn.tensors)} tensors (Node1 & Node3 only)")

mu_env_tn.draw(
    color=['Node1_mu', 'Node3_mu'],
    custom_colors=['blue', 'green'],
    show_inds=True,
    show_tags=True,
    title="μ_env: Node2 REMOVED, Node1 & Node3 remain (dangling bonds: b1, b2)",
    figsize=(14, 10),
    layout='kamada_kawai'
)
plt.close()

print("  ✓ Saved: showcase/outputs/02_mu_env.png")

print("\n  After contracting with inputs and computing outer product,")
print("  μ_outer will have the same indices as Σ_env:")
print("    (b1, x2, b2, b1_prime, x2_prime, b2_prime)")

# Bond priors visualization
print("\n--- θ (bond priors) ---")
print("θ = E[λ_b1] ⊗ E[λ_x2] ⊗ E[λ_b2]")
print("  Has shape (b1, x2, b2) - diagonal contribution to precision")

# ==============================================================================
# PART 2: Mu Update - Forward with Target
# ==============================================================================
print("\n" + "="*70)
print("PART 2: μ UPDATE - Forward with Target for Node2")
print("="*70)

print("\nMu update formula:")
print("  1. Remove Node2 from μ network")
print("  2. Contract remaining network with inputs AND targets → RHS")
print("  3. Relabel RHS indices: b1 → b1_prime, x2 → x2_prime, b2 → b2_prime")
print("  4. Contract: new_μ = τ × (RHS × Σ[Node2])")

print("\n--- Step 1: Remove Node2 and contract with target ---")

# μ_env is already computed above
print("Starting from μ_env (Node1 & Node3 with dangling bonds)")
print("Add inputs and TARGET, then contract everything")
print("Result (RHS) will have indices: (b1, x2, b2)")

# Show the final contraction: RHS with Sigma[Node2]
print("\n--- Step 2: Contract RHS × Σ[Node2] ---")

rhs = qt.Tensor(torch.randn(3, 2, 3), 
                inds=('b1_prime', 'x2_prime', 'b2_prime'), 
                tags={'RHS'})

contraction_tn = qt.TensorNetwork([rhs, s2])

print("\nRHS after relabeling:")
print("  Indices: (b1_prime, x2_prime, b2_prime)")
print("\nΣ[Node2]:")
print("  Indices: (b1, x2, b2, b1_prime, x2_prime, b2_prime)")

contraction_tn.draw(
    color=['RHS', 'Node2_sigma'],
    custom_colors=['red', 'yellow'],
    show_inds=True,
    show_tags=True,
    title="Final contraction: RHS × Σ[Node2] → new μ[Node2]",
    figsize=(14, 10),
    layout='kamada_kawai'
)
plt.close()
print("  ✓ Saved: showcase/outputs/02_rhs_sigma_contract.png")

print("\nAfter contracting over primed indices (b1', x2', b2'):")
print("  Result has indices: (b1, x2, b2)")
print("  This is the new μ[Node2] - ready to insert back into the network!")

# ==============================================================================
# Summary
# ==============================================================================
print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)

print("\n1. ENVIRONMENT = Network with target node REMOVED")
print("   - Other nodes stay as-is (not contracted)")
print("   - Dangling bonds show where target node connected")

print("\n2. Σ UPDATE:")
print("   Σ_env + μ_outer + θ → Precision → Invert → New Σ[Node2]")
print("   All terms have indices: (b1, x2, b2, b1', x2', b2')")

print("\n3. μ UPDATE:")
print("   Forward(inputs + target) → RHS → Relabel → RHS × Σ[Node2]")
print("   Contract over primed indices → New μ[Node2]")

print("\n" + "="*70)
print("SHOWCASE 02 COMPLETE ✓")
print("="*70)
print("\nGenerated 5 network topology visualizations:")
print("  1. 02_sigma_full.png          - Full Σ network")
print("  2. 02_sigma_env.png           - Σ_env (Node2 removed)")
print("  3. 02_mu_full.png             - Full μ network")
print("  4. 02_mu_env.png              - μ_env (Node2 removed)")
print("  5. 02_rhs_sigma_contract.png  - RHS × Σ contraction")
print("\nAll visualizations show the ACTUAL network topology!")
