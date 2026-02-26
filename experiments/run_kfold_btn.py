# type: ignore
"""
K-Fold Cross-Validation runner for BTN (Bayesian Tensor Network).
Runs k-fold validation across all datasets with proper train/val/test splits.

For each fold:
1. Split data into train/val/test
2. Train using validation as stopping criterion
3. Save best model at validation performance
4. Evaluate on test set
5. Record parameters and configuration

BTN-specific tracking includes:
- Standard metrics: train/val loss, R², MSE
- ELBO (raw and relative)
- KL divergences (bond and node)
- Tau expectation (noise precision)
- Bond parameter expectations
"""

import os
import sys
import json
import argparse
import time
import math
import copy
import torch
import quimb.tensor as qt
import numpy as np
from sklearn.model_selection import KFold
from typing import List, Dict, Any, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.dataset_loader import load_dataset, append_bias, get_available_datasets
from experiments.trackers import create_tracker, TrackerError
from experiments.device_utils import DEVICE, move_tn_to_device, move_data_to_device
from tensor.builder import Inputs
from tensor.btn import BTN
from model import MODELS
from model.utils import REGRESSION_METRICS, compute_quality
from model.load_ucirepo import datasets as uci_datasets

torch.set_default_dtype(torch.float64)

# All regression datasets
ALL_REGRESSION_DATASETS = [name for name, _, task in uci_datasets if task == "regression"]


def get_result_filepath(output_dir: str, run_id: str) -> str:
    return os.path.join(output_dir, f"{run_id}.json")


def run_already_completed(output_dir: str, run_id: str) -> tuple:
    """Check if a run has already been attempted."""
    result_file = get_result_filepath(output_dir, run_id)

    if not os.path.exists(result_file):
        return False, False, False, None

    try:
        with open(result_file, "r") as f:
            result = json.load(f)
        success = result.get("success", False)
        singular = result.get("singular", False)
        error = result.get("error", None)
        return True, success, singular, error
    except:
        return False, False, False, None


def create_model(params: dict, n_features: int):
    """Create a tensor network model from parameters."""
    model_name = params.get("model", "MPO2")
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

    model_cls = MODELS[model_name]

    base_params = {
        "L": params["L"],
        "bond_dim": params["bond_dim"],
        "phys_dim": n_features,
        "output_dim": 1,
        "init_strength": params.get("init_strength", 0.1),
        "use_tn_normalization": True,
    }

    if "output_site" in params:
        base_params["output_site"] = params["output_site"]

    if model_name == "LMPO2":
        base_params["reduction_factor"] = params.get("reduction_factor", 0.5)
        base_params["mpo_bond_dim"] = params.get("mpo_bond_dim", 1)

    return model_cls(**base_params)


def count_parameters(tn) -> int:
    """Count total number of parameters in a tensor network."""
    return sum(t.data.numel() for t in tn.tensors)


def _safe_float(value, default=0.0):
    """Convert value to float, returning default if not finite."""
    try:
        if torch.is_tensor(value):
            val = value.item() if value.numel() == 1 else float(value)
        else:
            val = float(value)
        return val if math.isfinite(val) else default
    except:
        return default


def extract_btn_metrics(btn) -> dict:
    """Extract BTN-specific metrics for tracking."""
    metrics = {}

    try:
        metrics["elbo_raw"] = _safe_float(btn._compute_raw_elbo())
    except:
        metrics["elbo_raw"] = 0.0

    try:
        metrics["elbo_relative"] = _safe_float(btn.compute_elbo(verbose=False, relative=True))
    except:
        metrics["elbo_relative"] = 0.0

    try:
        metrics["e_log_p_nodes"] = _safe_float(btn._compute_e_log_p_nodes())
    except:
        metrics["e_log_p_nodes"] = 0.0

    try:
        metrics["e_log_p_bonds"] = _safe_float(btn._compute_e_log_p_bonds())
    except:
        metrics["e_log_p_bonds"] = 0.0

    try:
        metrics["e_log_p_tau"] = _safe_float(btn._compute_e_log_p_tau())
    except:
        metrics["e_log_p_tau"] = 0.0

    try:
        metrics["h_nodes"] = _safe_float(btn._H_nodes())
    except:
        metrics["h_nodes"] = 0.0

    try:
        metrics["h_bonds"] = _safe_float(btn._H_bonds())
    except:
        metrics["h_bonds"] = 0.0

    try:
        metrics["h_tau"] = _safe_float(btn._H_tau())
    except:
        metrics["h_tau"] = 0.0

    try:
        tau_mean = btn.q_tau.mean()
        if isinstance(tau_mean, qt.Tensor):
            tau_mean = tau_mean.data
        metrics["tau_mean"] = float(tau_mean.item() if torch.is_tensor(tau_mean) else tau_mean)
    except:
        metrics["tau_mean"] = 0.0

    try:
        bond_expectations = {}
        bond_dims = {}
        excluded = set(btn.output_dimensions)
        for bond_tag in btn.q_bonds:
            if bond_tag not in excluded:
                mean = btn.q_bonds[bond_tag].mean()
                if isinstance(mean, qt.Tensor):
                    mean = mean.data
                bond_expectations[bond_tag] = float(mean.mean().item())
                bond_dims[bond_tag] = int(btn.mu.ind_size(bond_tag))
        metrics["bond_expectations"] = bond_expectations
        metrics["bond_dims"] = bond_dims
        metrics["bond_mean_avg"] = (
            sum(bond_expectations.values()) / len(bond_expectations) if bond_expectations else 0.0
        )
        metrics["n_parameters"] = count_parameters(btn.mu)
    except:
        metrics["bond_expectations"] = {}
        metrics["bond_dims"] = {}
        metrics["bond_mean_avg"] = 0.0
        metrics["n_parameters"] = 0

    return metrics


def prepare_fold_data(
    X: torch.Tensor,
    y: torch.Tensor,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    val_ratio: float = 0.15,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Prepare data for a single fold.
    
    Splits train_idx into train and validation sets.
    test_idx becomes the test set.
    """
    # Split train indices into train and validation
    n_train_total = len(train_idx)
    n_val = int(n_train_total * val_ratio)
    
    # Shuffle train indices for random val split
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility within fold
    shuffled_train_idx = train_idx.copy()
    rng.shuffle(shuffled_train_idx)
    
    val_idx = shuffled_train_idx[:n_val]
    actual_train_idx = shuffled_train_idx[n_val:]
    
    data = {
        "X_train": X[actual_train_idx].to(device),
        "y_train": y[actual_train_idx].to(device),
        "X_val": X[val_idx].to(device),
        "y_val": y[val_idx].to(device),
        "X_test": X[test_idx].to(device),
        "y_test": y[test_idx].to(device),
    }
    
    return data


def load_full_dataset(dataset_name: str, device: str = "cpu", cap: int = 50) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Load full dataset without splitting (for k-fold).
    
    Returns X, y tensors and dataset info.
    """
    from ucimlrepo import fetch_ucirepo
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from model.load_ucirepo import one_hot_with_cap, DATASETS_WITH_TARGET_FIX
    
    # Find dataset ID
    dataset_map = {name: (dataset_id, task) for name, dataset_id, task in uci_datasets}
    if dataset_name not in dataset_map:
        raise ValueError(f"Dataset '{dataset_name}' not found")
    
    dataset_id, task = dataset_map[dataset_name]
    
    dataset = fetch_ucirepo(id=dataset_id)
    X = dataset.data.features
    y = dataset.data.targets
    
    if dataset_id in DATASETS_WITH_TARGET_FIX:
        target_col = DATASETS_WITH_TARGET_FIX[dataset_id]
        y = X[[target_col]]
        X = X.drop(columns=[target_col])
    
    X = X.dropna(axis=1)
    X_all, orig_num_cols, dummy_cols = one_hot_with_cap(X, cap=cap)
    
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    if (
        pd.api.types.is_string_dtype(y)
        or pd.api.types.is_categorical_dtype(y)
        or y.dtype == "object"
    ):
        y = y.astype("category").cat.codes.astype(float)
    
    # Convert to tensors (no splitting yet)
    X_tensor = torch.tensor(X_all.values, dtype=torch.float64, device=device)
    y_tensor = torch.tensor(y.values, dtype=torch.float64, device=device)
    
    if y_tensor.ndim == 1:
        y_tensor = y_tensor.unsqueeze(1)
    
    dataset_info = {
        "name": dataset_name,
        "dataset_id": dataset_id,
        "n_samples": len(X_tensor),
        "n_features": X_tensor.shape[1],
        "task": task,
        "orig_num_cols": orig_num_cols,
    }
    
    return X_tensor, y_tensor, dataset_info


def normalize_fold_data(data: Dict[str, torch.Tensor], orig_num_cols: List[str], n_features: int) -> Dict[str, torch.Tensor]:
    """
    Normalize features based on training set statistics.
    Only normalizes numeric columns (not one-hot encoded).
    """
    n_numeric = len(orig_num_cols) if orig_num_cols else n_features
    
    if n_numeric > 0:
        # Compute mean and std from training data (numeric columns only)
        X_train_numeric = data["X_train"][:, :n_numeric]
        mean = X_train_numeric.mean(dim=0, keepdim=True)
        std = X_train_numeric.std(dim=0, keepdim=True)
        std = torch.where(std > 0, std, torch.ones_like(std))
        
        # Normalize all splits
        for key in ["X_train", "X_val", "X_test"]:
            X = data[key]
            X_numeric = X[:, :n_numeric]
            X_normalized = (X_numeric - mean) / std
            if n_numeric < X.shape[1]:
                # Keep one-hot columns unchanged
                data[key] = torch.cat([X_normalized, X[:, n_numeric:]], dim=1)
            else:
                data[key] = X_normalized
    
    return data


def run_single_fold(
    params: dict,
    data: dict,
    dataset_info: dict,
    seed: int,
    fold: int,
    verbose: bool = False,
    tracker=None,
) -> dict:
    """
    Run a single fold of k-fold cross-validation for BTN.
    
    Uses validation set for early stopping, evaluates on test set.
    Saves best model state at best validation performance.
    """
    model_name = params.get("model", "MPO2")
    
    torch.manual_seed(seed)
    
    n_epochs = params.get("n_epochs", 50)
    batch_size = params.get("batch_size", 64)
    patience = params.get("patience", None)
    min_delta = params.get("min_delta", 0.0)
    bond_prior_alpha = params.get("bond_prior_alpha", 5.0)
    added_bias = params.get("added_bias", False)
    
    # BTN-specific parameters
    trim_every = 3
    trimming_threshold = 0.3
    trim_method = params.get("trim_method", "relevance")
    soft_trim_relaxation = params.get("soft_trim_relaxation", 2)
    warmup_epochs = params.get("warmup_epochs", 5)
    
    # Deep copy data to avoid modifying original
    data = {k: v.clone() for k, v in data.items()}
    
    if added_bias:
        data = append_bias(data)
    
    n_features = data["X_train"].shape[1]
    
    try:
        model = create_model(params, n_features)
        mu_tn = model.tn
        move_tn_to_device(mu_tn)
        input_dims = model.input_dims
        output_dims = model.output_dims
        
        y_train = data["y_train"].squeeze(-1) if not output_dims else data["y_train"]
        y_val = data["y_val"].squeeze(-1) if not output_dims else data["y_val"]
        y_test = data["y_test"].squeeze(-1) if not output_dims else data["y_test"]
        
        train_loader = Inputs(
            inputs=[data["X_train"]],
            outputs=[y_train],
            outputs_labels=output_dims,
            input_labels=input_dims,
            batch_dim="s",
            batch_size=batch_size,
        )
        
        val_loader = Inputs(
            inputs=[data["X_val"]],
            outputs=[y_val],
            outputs_labels=output_dims,
            input_labels=input_dims,
            batch_dim="s",
            batch_size=batch_size,
        )
        
        test_loader = Inputs(
            inputs=[data["X_test"]],
            outputs=[y_test],
            outputs_labels=output_dims,
            input_labels=input_dims,
            batch_dim="s",
            batch_size=data["X_test"].shape[0],
        )
        
        btn = BTN(
            mu=mu_tn,
            data_stream=train_loader,
            batch_dim="s",
            method="cholesky",
            device=DEVICE,
            bond_prior_alpha=bond_prior_alpha,
        )
        btn.register_data_streams(val_loader, test_loader)
        btn.threshold = trimming_threshold
        
        if tracker:
            hparams = {
                "seed": seed,
                "fold": fold,
                "dataset": dataset_info["name"],
                "n_features": n_features,
                "n_train": len(data["X_train"]),
                "n_val": len(data["X_val"]),
                "n_test": len(data["X_test"]),
                "L": params["L"],
                "bond_dim": params["bond_dim"],
                **params,
            }
            tracker.log_hparams(hparams)
        
        best_val_quality = float("-inf")
        best_epoch = -1
        best_model_state = None
        stopped_early = False
        patience_counter = 0
        
        excluded = set(btn.output_dimensions)
        bonds = [i for i in btn.mu.ind_map if i not in excluded]
        nodes = list(btn.mu.tag_map.keys())
        
        # Initialize elbo_initial BEFORE training
        _ = btn.compute_elbo(verbose=False, relative=True)
        
        # Capture initial bond dimensions before training
        initial_btn_metrics = extract_btn_metrics(btn)
        initial_bond_dims = initial_btn_metrics.get("bond_dims", {})
        initial_n_params = initial_btn_metrics.get("n_parameters", 0)
        
        # Log initial metrics at step 0 (before training)
        if tracker:
            init_train_scores = btn.evaluate(REGRESSION_METRICS, data_stream=train_loader)
            init_val_scores = btn.evaluate(REGRESSION_METRICS, data_stream=val_loader)
            init_train_quality = compute_quality(init_train_scores)
            init_val_quality = compute_quality(init_val_scores)
            init_train_loss = init_train_scores.get("loss", (0, 1))
            if isinstance(init_train_loss, tuple):
                init_train_loss = (
                    init_train_loss[0] / init_train_loss[1] if init_train_loss[1] != 0 else 0
                )
            init_val_loss = init_val_scores.get("loss", (0, 1))
            if isinstance(init_val_loss, tuple):
                init_val_loss = init_val_loss[0] / init_val_loss[1] if init_val_loss[1] != 0 else 0

            init_metrics = {
                "train_loss": float(init_train_loss)
                if torch.is_tensor(init_train_loss)
                else init_train_loss,
                "train_quality": float(init_train_quality)
                if init_train_quality is not None
                else 0.0,
                "val_loss": float(init_val_loss)
                if torch.is_tensor(init_val_loss)
                else init_val_loss,
                "val_quality": float(init_val_quality) if init_val_quality is not None else 0.0,
                "patience_counter": 0,
                **{
                    k: v
                    for k, v in initial_btn_metrics.items()
                    if k not in ("bond_expectations", "bond_dims")
                },
            }
            if "bond_dims" in initial_btn_metrics:
                for bond_tag, dim in initial_btn_metrics["bond_dims"].items():
                    init_metrics[f"dim_{bond_tag}"] = dim
            tracker.log_metrics(init_metrics, step=0)
        
        soft_trim_started_epoch = None
        
        for epoch in range(n_epochs):
            for node_tag in nodes:
                btn.update_sigma_node(node_tag)
                btn.update_mu_node(node_tag)
            
            if epoch >= warmup_epochs:
                excluded_bonds_trim = btn.get_soft_trim_excluded_bonds()
                for bond_tag in bonds:
                    if bond_tag not in excluded_bonds_trim:
                        btn.update_bond(bond_tag)
            
            btn.update_tau()
            
            if btn.has_pending_soft_trims():
                epochs_in_relaxation = epoch - soft_trim_started_epoch
                if epochs_in_relaxation >= soft_trim_relaxation:
                    btn.finalize_soft_trim_bonds(verbose=False)
                    soft_trim_started_epoch = None
            
            if trim_every is not None and epoch >= warmup_epochs:
                epochs_since_warmup = epoch - warmup_epochs + 1
                if epochs_since_warmup % trim_every == 0 and not btn.has_pending_soft_trims():
                    if trim_method == "gamma":
                        btn.trim_bonds_by_gamma(threshold=trimming_threshold, verbose=False)
                    elif trim_method == "relevance":
                        btn.threshold = trimming_threshold
                        btn.trim_bonds(verbose=False)
            
            train_scores = btn.evaluate(REGRESSION_METRICS, data_stream=train_loader)
            val_scores = btn.evaluate(REGRESSION_METRICS, data_stream=val_loader)
            test_scores = btn.evaluate(REGRESSION_METRICS, data_stream=test_loader)
            
            train_quality = compute_quality(train_scores)
            val_quality = compute_quality(val_scores)
            test_quality = compute_quality(test_scores)
            train_loss = train_scores.get("loss", (0, 1))
            if isinstance(train_loss, tuple):
                train_loss = train_loss[0] / train_loss[1] if train_loss[1] != 0 else 0
            val_loss = val_scores.get("loss", (0, 1))
            if isinstance(val_loss, tuple):
                val_loss = val_loss[0] / val_loss[1] if val_loss[1] != 0 else 0
            
            btn_metrics = extract_btn_metrics(btn)
            
            # Check if this is the best validation quality
            if val_quality is not None and val_quality > best_val_quality + min_delta:
                best_val_quality = val_quality
                best_epoch = epoch
                patience_counter = 0
                # Save best model state (deep copy of mu tensors)
            else:
                patience_counter += 1
            
            if tracker:
                metrics = {
                    "train_loss": float(train_loss) if torch.is_tensor(train_loss) else train_loss,
                    "train_quality": float(train_quality) if train_quality is not None else 0.0,
                    "val_loss": float(val_loss) if torch.is_tensor(val_loss) else val_loss,
                    "val_quality": float(val_quality) if val_quality is not None else 0.0,
                    "test_quality": float(test_quality) if test_quality is not None else 0.0,
                    "patience_counter": patience_counter,
                    **{
                        k: v
                        for k, v in btn_metrics.items()
                        if k not in ("bond_expectations", "bond_dims")
                    },
                }
                # Log individual bond dimensions for tracking evolution
                if "bond_dims" in btn_metrics:
                    for bond_tag, dim in btn_metrics["bond_dims"].items():
                        metrics[f"dim_{bond_tag}"] = dim
                tracker.log_metrics(metrics, step=epoch + 1)
            
            if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
                print(
                    f"    Epoch {epoch + 1:3d} | Train: {train_quality:.4f} | Val: {val_quality:.4f} | "
                    f"ELBO: {btn_metrics['elbo_relative']:.2f} | Tau: {btn_metrics['tau_mean']:.4f}"
                )
            
            if patience is not None and patience_counter >= patience:
                if verbose:
                    print(f"    Early stopping at epoch {epoch + 1}")
                stopped_early = True
                break
        
        # Restore best model state for test evaluation
        if best_model_state is not None:
            for tag, data_tensor in best_model_state.items():
                btn.mu[tag].modify(data=data_tensor)
        
        # Evaluate on test set using best model
        test_scores = btn.evaluate(REGRESSION_METRICS, data_stream=test_loader)
        test_quality = compute_quality(test_scores)
        test_loss = test_scores.get("loss", (0, 1))
        if isinstance(test_loss, tuple):
            test_loss = test_loss[0] / test_loss[1] if test_loss[1] != 0 else 0
        
        final_btn_metrics = extract_btn_metrics(btn)
        
        result = {
            "fold": fold,
            "seed": seed,
            "model": model_name,
            "dataset": dataset_info["name"],
            "params": params,
            "n_features": n_features,
            "n_train": len(data["X_train"]),
            "n_val": len(data["X_val"]),
            "n_test": len(data["X_test"]),
            "train_loss": float(train_loss) if torch.is_tensor(train_loss) else train_loss,
            "train_quality": float(train_quality) if train_quality is not None else 0.0,
            "val_loss": float(val_loss) if torch.is_tensor(val_loss) else val_loss,
            "val_quality": float(best_val_quality) if best_val_quality != float("-inf") else 0.0,
            "test_loss": float(test_loss) if torch.is_tensor(test_loss) else test_loss,
            "test_quality": float(test_quality) if test_quality is not None else 0.0,
            "best_epoch": best_epoch,
            "stopped_early": stopped_early,
            "patience_counter": patience_counter,
            "elbo_raw": final_btn_metrics["elbo_raw"],
            "elbo_relative": final_btn_metrics["elbo_relative"],
            "e_log_p_nodes": final_btn_metrics["e_log_p_nodes"],
            "e_log_p_bonds": final_btn_metrics["e_log_p_bonds"],
            "e_log_p_tau": final_btn_metrics["e_log_p_tau"],
            "h_nodes": final_btn_metrics["h_nodes"],
            "h_bonds": final_btn_metrics["h_bonds"],
            "h_tau": final_btn_metrics["h_tau"],
            "tau_mean": final_btn_metrics["tau_mean"],
            "bond_mean_avg": final_btn_metrics["bond_mean_avg"],
            "initial_bond_dims": initial_bond_dims,
            "initial_n_parameters": initial_n_params,
            "final_bond_dims": final_btn_metrics["bond_dims"],
            "final_n_parameters": final_btn_metrics["n_parameters"],
            "success": True,
            "singular": False,
        }
        
        if tracker:
            tracker.log_summary({
                "test_quality": result["test_quality"],
                "test_loss": result["test_loss"],
                "best_val_quality": result["val_quality"],
                "n_parameters": result["final_n_parameters"],
            })
        
        return result
    
    except torch.linalg.LinAlgError as e:
        return {
            "fold": fold,
            "seed": seed,
            "model": model_name,
            "dataset": dataset_info["name"],
            "params": params,
            "success": True,
            "singular": True,
            "error": str(e),
        }
    
    except ValueError as e:
        error_msg = str(e).lower()
        is_covariance_error = "positivedefinite" in error_msg or "covariance" in error_msg
        if is_covariance_error:
            return {
                "fold": fold,
                "seed": seed,
                "model": model_name,
                "dataset": dataset_info["name"],
                "params": params,
                "success": True,
                "singular": True,
                "error": str(e),
            }
        raise
    
    except Exception as e:
        import traceback
        return {
            "fold": fold,
            "seed": seed,
            "model": model_name,
            "dataset": dataset_info["name"],
            "params": params,
            "success": False,
            "singular": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def run_kfold_on_dataset(
    dataset_name: str,
    params: dict,
    seeds: List[int],
    n_folds: int = 5,
    val_ratio: float = 0.15,
    verbose: bool = False,
    output_dir: str = None,
    tracker_config: dict = None,
) -> List[dict]:
    """
    Run k-fold cross-validation on a single dataset.
    
    Args:
        dataset_name: Name of the dataset
        params: Model hyperparameters
        seeds: List of random seeds
        n_folds: Number of folds for cross-validation
        val_ratio: Ratio of training data to use for validation
        verbose: Print progress
        output_dir: Directory to save results
        tracker_config: Tracker configuration
    
    Returns:
        List of results for each fold and seed combination
    """
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*70}")
    
    # Load full dataset
    try:
        X, y, dataset_info = load_full_dataset(dataset_name, device="cpu")
    except Exception as e:
        print(f"  Failed to load dataset: {e}")
        return []
    
    print(f"  Samples: {dataset_info['n_samples']}, Features: {dataset_info['n_features']}")
    
    results = []
    
    for seed in seeds:
        print(f"\n  Seed: {seed}")
        
        # Set up k-fold splitter
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        
        for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
            run_id = f"{dataset_name}-{params['model']}-L{params['L']}-d{params['bond_dim']}-seed{seed}-fold{fold}"
            
            # Check if already completed
            if output_dir:
                was_attempted, was_successful, is_singular, error = run_already_completed(
                    output_dir, run_id
                )
                if was_attempted and (was_successful or is_singular):
                    if verbose:
                        status = "success" if was_successful else "singular"
                        print(f"    Fold {fold}: SKIPPED ({status})")
                    continue
            
            print(f"    Fold {fold + 1}/{n_folds}...", end=" ", flush=True)
            
            # Prepare fold data
            fold_data = prepare_fold_data(X, y, train_idx, test_idx, val_ratio, device=DEVICE)
            
            # Normalize based on training data
            fold_data = normalize_fold_data(
                fold_data, 
                dataset_info.get("orig_num_cols", []), 
                dataset_info["n_features"]
            )
            
            # Move to device
            fold_data = {k: v.to(DEVICE) for k, v in fold_data.items()}
            
            # Create tracker if configured
            tracker = None
            if tracker_config and tracker_config.get("backend") != "none":
                tracker = create_tracker(
                    experiment_name=f"kfold_{dataset_name}",
                    config={"params": params, "fold": fold, "seed": seed},
                    backend=tracker_config.get("backend", "file"),
                    output_dir=tracker_config.get("tracker_dir", "runs"),
                    repo=tracker_config.get("aim_repo"),
                    run_name=run_id,
                )
            
            # Run fold
            result = run_single_fold(
                params=params,
                data=fold_data,
                dataset_info=dataset_info,
                seed=seed,
                fold=fold,
                verbose=verbose,
                tracker=tracker,
            )
            result["run_id"] = run_id
            
            if tracker and hasattr(tracker, "close"):
                tracker.close()
            
            # Save individual result
            if output_dir:
                result_file = get_result_filepath(output_dir, run_id)
                with open(result_file, "w") as f:
                    json.dump(result, f, indent=2)
            
            results.append(result)
            
            if result.get("singular"):
                print("SINGULAR")
            elif result.get("success"):
                print(f"Test Q={result['test_quality']:.4f}")
            else:
                print(f"FAILED: {result.get('error', 'Unknown')[:50]}")
    
    return results


def aggregate_results(results: List[dict]) -> dict:
    """
    Aggregate results across folds and seeds.
    
    Returns mean and std of test quality for each dataset.
    """
    from collections import defaultdict
    
    by_dataset = defaultdict(list)
    
    for r in results:
        if r.get("success") and not r.get("singular"):
            by_dataset[r["dataset"]].append(r["test_quality"])
    
    aggregated = {}
    for dataset, qualities in by_dataset.items():
        aggregated[dataset] = {
            "mean": np.mean(qualities),
            "std": np.std(qualities),
            "n_runs": len(qualities),
            "min": np.min(qualities),
            "max": np.max(qualities),
        }
    
    return aggregated


def load_specs_file(specs_path: str) -> dict:
    """Load best specs from JSON file generated by generate_results_table.py --specs."""
    with open(specs_path, "r") as f:
        return json.load(f)


def get_params_for_dataset(specs: dict, dataset: str, model: str, base_params: dict) -> dict:
    """Get best parameters for a dataset/model from specs, falling back to base_params."""
    params = base_params.copy()
    
    if dataset in specs and model in specs[dataset]:
        spec = specs[dataset][model]
        params["L"] = spec.get("L", params["L"])
        params["bond_dim"] = spec.get("bond_dim", params["bond_dim"])
        params["added_bias"] = spec.get("added_bias", params["added_bias"])
        params["init_strength"] = spec.get("init_strength", params["init_strength"])
        params["bond_prior_alpha"] = spec.get("bond_prior_alpha", params["bond_prior_alpha"])
        params["trimming_threshold"] = spec.get("trimming_threshold", params["trimming_threshold"])
        params["trim_method"] = spec.get("trim_method", params["trim_method"])
        params["trim_every"] = spec.get("trim_every", params["trim_every"])
        params["soft_trim_relaxation"] = spec.get("soft_trim_relaxation", params["soft_trim_relaxation"])
    
    return params


def main():
    parser = argparse.ArgumentParser(
        description="Run K-Fold Cross-Validation for BTN across all datasets"
    )
    
    parser.add_argument("--specs-file", type=str, default=None,
                        help="JSON file with best specs per dataset (from generate_results_table.py --specs)")
    parser.add_argument("--model", type=str, default="MPO2", 
                        choices=list(MODELS.keys()), help="Model architecture")
    parser.add_argument("--L", type=int, default=3, help="Number of sites (ignored if --specs-file)")
    parser.add_argument("--bond-dim", type=int, default=10, help="Bond dimension (ignored if --specs-file)")
    parser.add_argument("--init-strength", type=float, default=0.01, help="Init strength")
    parser.add_argument("--bond-prior-alpha", type=float, default=5.0, help="Bond prior alpha")
    parser.add_argument("--added-bias", action="store_true", help="Add bias term (ignored if --specs-file)")
    
    parser.add_argument("--n-epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience")
    parser.add_argument("--min-delta", type=float, default=0.0001, help="Min delta for improvement")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Warmup epochs")
    
    parser.add_argument("--trim-every", type=int, default=3, help="Trim bonds every N epochs")
    parser.add_argument("--trimming-threshold", type=float, default=0.3, help="Trimming threshold")
    parser.add_argument("--trim-method", type=str, default="relevance",
                        choices=["relevance", "gamma"], help="Bond trimming method")
    parser.add_argument("--soft-trim-relaxation", type=int, default=2, help="Soft trim relaxation epochs")
    
    parser.add_argument("--n-folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123, 256, 999],
                        help="Random seeds")
    
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Datasets to run (default: all regression datasets)")
    
    parser.add_argument("--output-dir", type=str, default="results/kfold_btn",
                        help="Output directory")
    parser.add_argument("--tracker", type=str, default="file",
                        choices=["file", "aim", "both", "none"], help="Tracker backend")
    parser.add_argument("--tracker-dir", type=str, default="runs_kfold_btn",
                        help="Tracker directory")
    parser.add_argument("--aim-repo", type=str, default=None, help="AIM repo")
    
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--force", action="store_true", help="Force re-run")
    
    args = parser.parse_args()
    
    specs = load_specs_file(args.specs_file) if args.specs_file else None
    
    base_params = {
        "model": args.model,
        "L": args.L,
        "bond_dim": args.bond_dim,
        "init_strength": args.init_strength,
        "bond_prior_alpha": args.bond_prior_alpha,
        "added_bias": args.added_bias,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "warmup_epochs": args.warmup_epochs,
        "trim_every": args.trim_every,
        "trimming_threshold": args.trimming_threshold,
        "trim_method": args.trim_method,
        "soft_trim_relaxation": args.soft_trim_relaxation,
    }
    
    datasets = args.datasets if args.datasets else ALL_REGRESSION_DATASETS
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.force:
        import shutil
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)
    
    tracker_config = {
        "backend": args.tracker,
        "tracker_dir": args.tracker_dir,
        "aim_repo": args.aim_repo,
    }
    
    print("=" * 70)
    print("K-FOLD CROSS-VALIDATION FOR BTN")
    print("=" * 70)
    print(f"Model: {args.model}")
    if specs:
        print(f"Using specs from: {args.specs_file}")
    else:
        print(f"L: {base_params['L']}, Bond Dim: {base_params['bond_dim']}")
        print(f"Bias: {base_params['added_bias']}")
    print(f"Folds: {args.n_folds}, Seeds: {args.seeds}")
    print(f"Datasets: {datasets}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {DEVICE}")
    print()
    
    start_time = time.time()
    all_results = []
    
    for dataset_name in datasets:
        if specs:
            params = get_params_for_dataset(specs, dataset_name, args.model, base_params)
            if dataset_name not in specs or args.model not in specs.get(dataset_name, {}):
                print(f"\nSkipping {dataset_name}: no specs for model {args.model}")
                continue
        else:
            params = base_params.copy()
        
        results = run_kfold_on_dataset(
            dataset_name=dataset_name,
            params=params,
            seeds=args.seeds,
            n_folds=args.n_folds,
            val_ratio=args.val_ratio,
            verbose=args.verbose,
            output_dir=args.output_dir,
            tracker_config=tracker_config,
        )
        all_results.extend(results)
    
    elapsed = time.time() - start_time
    
    # Aggregate and print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    aggregated = aggregate_results(all_results)
    
    print(f"\n{'Dataset':<20} {'Mean Q':>10} {'Std':>10} {'N Runs':>10}")
    print("-" * 50)
    for dataset, stats in sorted(aggregated.items()):
        print(f"{dataset:<20} {stats['mean']:>10.4f} {stats['std']:>10.4f} {stats['n_runs']:>10}")
    
    # Save summary
    summary = {
        "params": params,
        "seeds": args.seeds,
        "n_folds": args.n_folds,
        "val_ratio": args.val_ratio,
        "datasets": datasets,
        "elapsed_time": elapsed,
        "aggregated": aggregated,
        "all_results": all_results,
    }
    
    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Results saved to: {args.output_dir}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    try:
        main()
    except TrackerError as e:
        print(f"\n[FATAL] Tracker error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Cancelled by user", file=sys.stderr)
        sys.exit(130)
