# BTN Showcase

Comprehensive demonstration of the Bayesian Tensor Network (BTN) model.

## Overview

This showcase contains 5 demonstration scripts that walk through different aspects of the BTN model:

1. **Topology and Architecture** - Different network structures and BTNBuilder
2. **Update Rules Dissection** - Visual tensor network update mechanics  
3. **Training and Learning** - Training dynamics with full ELBO tracking
4. **Bond Trimming Analysis** - Automatic model compression
5. **Architecture Comparison** - Compare MPS vs Binary Tree performance

## Current Status

✅ **All showcase files completed!**
- `01_topology_and_architecture.py` - Network architectures and BTNBuilder
- `02_update_rules_dissection.py` - Visual tensor network update mechanics
- `03_training_and_learning.py` - Training with enhanced `fit()` method returning history dict
- `04_bond_trimming_analysis.py` - Automatic model compression via bond trimming
- `05_architecture_comparison.py` - Head-to-head comparison of architectures

## Running the Showcase

```bash
cd showcase
python 01_topology_and_architecture.py
```

## Requirements

```bash
pip install torch quimb matplotlib numpy
```

## Outputs

All visualization outputs are saved to `showcase/outputs/`:

### File 01: Topology and Architecture
- `01_standard_mps.png` - Standard MPS architecture visualization
- `01_binary_tree.png` - Binary tree architecture visualization  
- `01_polynomial_data.png` - Polynomial training data (y = 2x³ - x² + 0.5x + 0.2)
- `01_bond_distributions.png` - Prior and posterior bond distributions
- `01_mu_sigma_networks.png` - μ (mean) and Σ (covariance) networks

## Key Features Demonstrated

### Architecture Flexibility
- Standard MPS (linear chain)
- Binary tree structures
- Custom topologies supported

### BTNBuilder Process
- Automatic creation of prior/posterior distributions
- Bond distributions (Gamma)
- Node distributions (Multivariate Gaussian)
- Precision parameter (τ)

### Polynomial Learning
- Feature engineering: `[x, 1]` allows network to learn polynomial terms
- 3-site MPS can represent up to degree-3 polynomials
- Beautiful visualization of the target function

## Next Steps

The remaining showcase files will demonstrate:
- Detailed update mechanics with network visualizations at each step
- Training dynamics with comprehensive ELBO tracking
- Bond trimming strategies and compression analysis
