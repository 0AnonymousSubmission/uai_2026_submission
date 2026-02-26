"""
Metrics tracking utilities
"""
import torch
import numpy as np
from typing import Dict, List, Any


def compute_r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Compute R² score
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        
    Returns:
        R² score
    """
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2.item()


def compute_all_elbo_components(model: Any) -> Dict[str, float]:
    """
    Return dict with all ELBO components
    
    Args:
        model: BTN model instance
        
    Returns:
        Dictionary with keys: exp_ll, bond_kl, node_kl, tau_kl, elbo
    """
    exp_ll = model.compute_expected_log_likelihood().item()
    bond_kl = model.compute_bond_kl().item()
    node_kl = model.compute_node_kl().item()
    tau_kl = model.compute_tau_kl().item()
    elbo = exp_ll - bond_kl - node_kl - tau_kl
    
    return {
        'exp_ll': exp_ll,
        'bond_kl': bond_kl,
        'node_kl': node_kl,
        'tau_kl': tau_kl,
        'elbo': elbo
    }


def track_node_statistics(model: Any, node_tags: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Extract mu norms, sigma conditioning for all nodes
    
    Args:
        model: BTN model instance
        node_tags: List of node tag names
        
    Returns:
        Dictionary mapping node_tag to statistics dict
    """
    stats = {}
    
    for node_tag in node_tags:
        mu = model.mu[node_tag]
        sigma = model.sigma[node_tag]
        
        # Get mu norm
        mu_norm = torch.norm(mu.data).item()
        
        # Get sigma conditioning
        variational_idx, _, _ = model.get_variational_outputs_inds(mu)
        primed_idx = [i + '_prime' for i in variational_idx]
        
        try:
            sigma_dense = sigma.to_dense(variational_idx, primed_idx)
            eigs = torch.linalg.eigvalsh(sigma_dense)
            sigma_min_eig = eigs.min().item()
            sigma_max_eig = eigs.max().item()
            sigma_cond = (sigma_max_eig / sigma_min_eig) if sigma_min_eig > 0 else float('inf')
        except:
            sigma_min_eig = 0.0
            sigma_max_eig = 0.0
            sigma_cond = float('inf')
        
        stats[node_tag] = {
            'mu_norm': mu_norm,
            'sigma_cond': sigma_cond,
            'sigma_min_eig': sigma_min_eig,
            'sigma_max_eig': sigma_max_eig,
        }
    
    return stats


def track_bond_statistics(model: Any) -> Dict[str, Dict[str, Any]]:
    """
    Extract E[λ], alpha, beta for all bonds
    
    Args:
        model: BTN model instance
        
    Returns:
        Dictionary mapping bond name to statistics dict
    """
    stats = {}
    
    # Get all bonds (excluding batch dimension)
    bonds = [ind for ind in model.mu.ind_map if ind != model.batch_dim]
    
    for bond in bonds:
        if bond in model.q_bonds:
            gamma_dist = model.q_bonds[bond]
            alpha = gamma_dist.concentration.data.clone()
            beta = gamma_dist.rate.data.clone()
            mean = gamma_dist.mean().data.clone()
            
            stats[bond] = {
                'alpha': alpha,
                'beta': beta,
                'mean': mean,
                'dimension': len(mean)
            }
    
    return stats
