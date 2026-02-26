# type: ignore
"""
Dataset loader utility for BTN experiments.
Uses UCI ML Repository for loading regression datasets.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.load_ucirepo import get_ucidata, datasets as uci_datasets


def load_dataset(
    dataset_name, n_samples=None, seed=0, val_split=0.2, test_split=0.2, device="cpu", cap=50
):
    """
    Load a regression dataset by name from UCI ML Repository.

    Note: Train/val/test splits are FIXED (seed=42) for reproducibility and fair
    comparison across experiments. The experiment seed controls model initialization,
    not data splits.

    Args:
        dataset_name: Name of dataset to load (e.g., 'concrete', 'abalone')
        n_samples: Number of samples to use (None = use all) - NOT IMPLEMENTED
        seed: Unused (splits are fixed; experiment seed controls model init)
        val_split: Unused (fixed 70/15/15 split)
        test_split: Unused (fixed 70/15/15 split)
        device: Device to load tensors on (default: 'cpu')
        cap: Maximum number of features after one-hot encoding (default: 50)

    Returns:
        data: dict with keys 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'
        dataset_info: dict with metadata
    """
    uci_dataset_map = {name: (dataset_id, task) for name, dataset_id, task in uci_datasets}

    if dataset_name not in uci_dataset_map:
        available = [name for name, _, task in uci_datasets if task == "regression"]
        raise ValueError(
            f"Dataset '{dataset_name}' not found. "
            f"Available regression datasets: {available}"
        )

    dataset_id, task = uci_dataset_map[dataset_name]

    if task != "regression":
        raise ValueError(f"Dataset '{dataset_name}' is a {task} dataset. BTN only supports regression.")

    X_train, y_train, X_val, y_val, X_test, y_test = get_ucidata(
        dataset_id=dataset_id, task=task, device=device, cap=cap
    )

    if y_train.ndim == 1:
        y_train = y_train.unsqueeze(1)
    if y_val.ndim == 1:
        y_val = y_val.unsqueeze(1)
    if y_test.ndim == 1:
        y_test = y_test.unsqueeze(1)

    data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

    dataset_info = {
        "name": dataset_name,
        "dataset_id": dataset_id,
        "n_samples": len(X_train) + len(X_val) + len(X_test),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "n_features": X_train.shape[1],
        "task": task,
    }

    return data, dataset_info


def normalize_targets(data, method="zscore"):
    """
    Normalize target values for regression.

    Args:
        data: dict with 'y_train', 'y_val', 'y_test'
        method: 'zscore' (mean=0, std=1) or 'minmax' (0-1 range)

    Returns:
        Normalized data dict and normalization stats for inverse transform
    """
    y_train = data["y_train"]

    if method == "zscore":
        y_mean = y_train.mean()
        y_std = y_train.std()
        if y_std == 0:
            y_std = 1.0

        data["y_train"] = (y_train - y_mean) / y_std
        data["y_val"] = (data["y_val"] - y_mean) / y_std
        data["y_test"] = (data["y_test"] - y_mean) / y_std

        norm_stats = {"method": "zscore", "mean": y_mean, "std": y_std}

    elif method == "minmax":
        y_min = y_train.min()
        y_max = y_train.max()
        y_range = y_max - y_min
        if y_range == 0:
            y_range = 1.0

        data["y_train"] = (y_train - y_min) / y_range
        data["y_val"] = (data["y_val"] - y_min) / y_range
        data["y_test"] = (data["y_test"] - y_min) / y_range

        norm_stats = {"method": "minmax", "min": y_min, "max": y_max}

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return data, norm_stats


def append_bias(data):
    """
    Append a column of ones (bias term) to input features.

    Args:
        data: dict with 'X_train', 'X_val', 'X_test'

    Returns:
        Modified data dict with bias column appended
    """
    for key in ["X_train", "X_val", "X_test"]:
        X = data[key]
        bias = torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)
        data[key] = torch.cat([X, bias], dim=1)

    return data


def expand_features_polynomial(data, degree: int, device: str = "cpu"):
    """
    Expand each feature to [x^0, x^1, ..., x^degree].
    Transforms X from [n_samples, n_features] to [n_samples, n_features, degree+1].
    """
    phys_dim = degree + 1
    for key in ["X_train", "X_val", "X_test"]:
        X = data[key]
        n_samples, n_features = X.shape
        X_expanded = torch.zeros(n_samples, n_features, phys_dim, dtype=X.dtype, device=device)
        for p in range(phys_dim):
            X_expanded[:, :, p] = X.to(device) ** p
        data[key] = X_expanded
    for key in ["y_train", "y_val", "y_test"]:
        data[key] = data[key].to(device)
    return data


def get_available_datasets():
    """Return list of available regression datasets."""
    return [name for name, _, task in uci_datasets if task == "regression"]


def load_polynomial_dataset(
    n_samples: int = 1000,
    n_features: int = 5,
    degree: int = 2,
    n_terms: int = None,
    noise_std: float = 0.1,
    seed: int = 42,
    val_split: float = 0.15,
    test_split: float = 0.15,
    device: str = "cpu",
    interaction_only: bool = False,
    coefficients: list = None,
):
    """
    Generate a synthetic polynomial regression dataset.

    The target y is computed as a multivariate polynomial of the input features X.
    Supports polynomials up to arbitrary degree with configurable number of terms.

    Args:
        n_samples: Total number of samples to generate.
        n_features: Number of input features (dimensions).
        degree: Maximum degree of polynomial terms.
        n_terms: Number of polynomial terms to include. If None, includes all terms
                 up to the specified degree. If specified, randomly selects n_terms.
        noise_std: Standard deviation of Gaussian noise added to y.
        seed: Random seed for reproducibility.
        val_split: Fraction of data for validation set.
        test_split: Fraction of data for test set.
        device: Device to load tensors on ('cpu' or 'cuda').
        interaction_only: If True, only include interaction terms (no pure powers).
        coefficients: Optional list of coefficients for each term. If None, 
                      coefficients are sampled from N(0, 1).

    Returns:
        data: dict with keys 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'
        dataset_info: dict with metadata including polynomial specification

    Example:
        # Simple quadratic: y = sum of x_i^2 terms + interactions
        data, info = load_polynomial_dataset(n_features=3, degree=2)

        # Sparse polynomial with 10 random terms up to degree 3
        data, info = load_polynomial_dataset(n_features=5, degree=3, n_terms=10)
    """
    torch.manual_seed(seed)
    
    X = torch.randn(n_samples, n_features, dtype=torch.float64, device=device)
    terms = _generate_polynomial_terms(n_features, degree, interaction_only)
    
    if n_terms is not None and n_terms < len(terms):
        bias_term = tuple([0] * n_features)
        linear_terms = [tuple([1 if j == i else 0 for j in range(n_features)]) 
                        for i in range(n_features)]
        required_terms = [bias_term] + linear_terms
        required_terms = [t for t in required_terms if t in terms]
        
        other_terms = [t for t in terms if t not in required_terms]
        n_remaining = max(0, n_terms - len(required_terms))
        if n_remaining > 0 and other_terms:
            indices = torch.randperm(len(other_terms))[:n_remaining].tolist()
            selected_other = [other_terms[i] for i in indices]
        else:
            selected_other = []
        
        terms = required_terms + selected_other
    
    n_total_terms = len(terms)
    if coefficients is not None:
        if len(coefficients) != n_total_terms:
            raise ValueError(f"Expected {n_total_terms} coefficients, got {len(coefficients)}")
        coef = torch.tensor(coefficients, dtype=torch.float64, device=device)
    else:
        coef = torch.randn(n_total_terms, dtype=torch.float64, device=device)
    
    # y = sum_i coef_i * prod_j x_j^{exponent_ij}
    y = torch.zeros(n_samples, dtype=torch.float64, device=device)
    for i, term in enumerate(terms):
        term_value = torch.ones(n_samples, dtype=torch.float64, device=device)
        for j, exp in enumerate(term):
            if exp > 0:
                term_value = term_value * (X[:, j] ** exp)
        y = y + coef[i] * term_value
    
    if noise_std > 0:
        y = y + torch.randn_like(y) * noise_std
    
    y_mean = y.mean()
    y_std = y.std()
    if y_std > 0:
        y = (y - y_mean) / y_std
    
    n_test = int(n_samples * test_split)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val - n_test
    
    indices = torch.randperm(n_samples, device=device)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    X_mean = X_train.mean(dim=0, keepdim=True)
    X_std = X_train.std(dim=0, keepdim=True)
    X_std = torch.where(X_std > 0, X_std, torch.ones_like(X_std))
    
    X_train = (X_train - X_mean) / X_std
    X_val = (X_val - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    
    bias_train = torch.ones(X_train.shape[0], 1, dtype=torch.float64, device=device)
    bias_val = torch.ones(X_val.shape[0], 1, dtype=torch.float64, device=device)
    bias_test = torch.ones(X_test.shape[0], 1, dtype=torch.float64, device=device)
    X_train = torch.cat([X_train, bias_train], dim=1)
    X_val = torch.cat([X_val, bias_val], dim=1)
    X_test = torch.cat([X_test, bias_test], dim=1)
    
    y_train = y_train.unsqueeze(1)
    y_val = y_val.unsqueeze(1)
    y_test = y_test.unsqueeze(1)
    
    data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }
    
    dataset_info = {
        "name": f"polynomial_d{n_features}_deg{degree}",
        "n_samples": n_samples,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_features": n_features + 1,
        "task": "regression",
        "polynomial_degree": degree,
        "polynomial_terms": len(terms),
        "noise_std": noise_std,
        "seed": seed,
        "terms": terms,
        "coefficients": coef.tolist(),
    }
    
    return data, dataset_info


def _generate_polynomial_terms(n_features: int, degree: int, interaction_only: bool = False):
    """
    Generate all polynomial term exponent tuples up to a given degree.

    Each term is represented as a tuple of exponents, one per feature.
    E.g., for 2 features: (0, 0) = bias, (1, 0) = x1, (0, 1) = x2, 
                          (2, 0) = x1^2, (1, 1) = x1*x2, (0, 2) = x2^2

    Args:
        n_features: Number of input features.
        degree: Maximum total degree of terms.
        interaction_only: If True, exclude pure power terms (e.g., x1^2).

    Returns:
        List of tuples, each tuple contains exponents for each feature.
    """
    from itertools import combinations_with_replacement
    
    terms = []
    
    # Generate terms of each total degree from 0 to degree
    for d in range(degree + 1):
        # Generate all ways to distribute degree d among n_features
        for combo in combinations_with_replacement(range(n_features), d):
            exponents = [0] * n_features
            for idx in combo:
                exponents[idx] += 1
            
            # Skip pure powers if interaction_only
            if interaction_only and d > 1:
                non_zero_count = sum(1 for e in exponents if e > 0)
                if non_zero_count < 2:
                    continue
            
            term = tuple(exponents)
            if term not in terms:
                terms.append(term)
    
    return terms


def load_featurewise_polynomial_dataset(
    n_samples: int = 1000,
    n_features: int = 5,
    degree: int = 3,
    noise_std: float = 0.1,
    seed: int = 42,
    val_split: float = 0.15,
    test_split: float = 0.15,
    device: str = "cpu",
    target_coefficients: list = None,
):
    """
    Generate polynomial dataset with feature-wise power expansion.
    
    Each feature is expanded to [x^0, x^1, ..., x^degree], giving each input 
    node a physical dimension of (degree+1). This is the conventional tensor 
    network approach where each site corresponds to one feature.
    
    Returns:
        data: dict with X tensors of shape [n_samples, n_features, degree+1]
        dataset_info: dict with metadata
    
    Example:
        For n_features=2, degree=2:
        X[i, 0, :] = [1, feat0, feat0^2]  # powers of feature 0
        X[i, 1, :] = [1, feat1, feat1^2]  # powers of feature 1
    """
    torch.manual_seed(seed)
    
    phys_dim = degree + 1
    raw_features = torch.randn(n_samples, n_features, dtype=torch.float64, device=device)
    
    X = torch.zeros(n_samples, n_features, phys_dim, dtype=torch.float64, device=device)
    for p in range(phys_dim):
        X[:, :, p] = raw_features ** p
    
    terms = _generate_polynomial_terms(n_features, degree, interaction_only=False)
    n_terms = len(terms)
    
    if target_coefficients is not None:
        if len(target_coefficients) != n_terms:
            raise ValueError(f"Expected {n_terms} coefficients, got {len(target_coefficients)}")
        coef = torch.tensor(target_coefficients, dtype=torch.float64, device=device)
    else:
        coef = torch.randn(n_terms, dtype=torch.float64, device=device)
    
    # y = sum_i coef_i * prod_j feat_j^{exp_ij}
    y = torch.zeros(n_samples, dtype=torch.float64, device=device)
    for i, term in enumerate(terms):
        term_value = torch.ones(n_samples, dtype=torch.float64, device=device)
        for j, exp in enumerate(term):
            if exp > 0:
                term_value = term_value * (raw_features[:, j] ** exp)
        y = y + coef[i] * term_value
    
    if noise_std > 0:
        y = y + torch.randn_like(y) * noise_std
    
    y_mean, y_std = y.mean(), y.std()
    if y_std > 0:
        y = (y - y_mean) / y_std
    
    n_test = int(n_samples * test_split)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val - n_test
    
    indices = torch.randperm(n_samples, device=device)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    y_train = y_train.unsqueeze(1)
    y_val = y_val.unsqueeze(1)
    y_test = y_test.unsqueeze(1)
    
    data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }
    
    dataset_info = {
        "name": f"featurewise_poly_f{n_features}_deg{degree}",
        "n_samples": n_samples,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_features": n_features,
        "phys_dim": phys_dim,
        "task": "regression",
        "polynomial_degree": degree,
        "polynomial_terms": n_terms,
        "noise_std": noise_std,
        "seed": seed,
        "terms": terms,
        "coefficients": coef.tolist(),
    }
    
    return data, dataset_info


def load_uci_featurewise(dataset_name: str, degree: int = 2, device: str = "cpu", cap: int = 20):
    """
    Load UCI dataset with featurewise polynomial expansion.
    
    Each feature is expanded to [x^0, x^1, ..., x^degree].
    Returns X with shape [n_samples, n_features, degree+1].
    """
    data, info = load_dataset(dataset_name, device=device, cap=cap)
    
    phys_dim = degree + 1
    n_features = info["n_features"]
    
    def expand_features(X):
        n_samples = X.shape[0]
        X_expanded = torch.zeros(n_samples, n_features, phys_dim, dtype=X.dtype, device=X.device)
        for p in range(phys_dim):
            X_expanded[:, :, p] = X ** p
        return X_expanded
    
    data["X_train"] = expand_features(data["X_train"])
    data["X_val"] = expand_features(data["X_val"])
    data["X_test"] = expand_features(data["X_test"])
    
    info["phys_dim"] = phys_dim
    info["polynomial_degree"] = degree
    info["name"] = f"{dataset_name}_deg{degree}"
    
    return data, info


def get_available_synthetic_datasets():
    """Return list of available synthetic dataset generators."""
    return ["polynomial", "featurewise_polynomial"]
