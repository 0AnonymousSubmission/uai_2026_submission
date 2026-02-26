"""
Shared plotting utilities for BTN showcase
"""
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import torch
from typing import Optional, Dict, Any, Callable


def plot_elbo_components_stacked(history: Dict[str, list], ax: Optional[Axes] = None) -> Axes:
    """
    Create stacked area chart of ELBO components
    
    Args:
        history: Dictionary with keys 'epoch', 'exp_ll', 'bond_kl', 'node_kl', 'tau_kl'
        ax: Optional matplotlib axis to plot on
        
    Returns:
        The matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    epochs = history['epoch']
    # Handle both naming conventions
    exp_ll_key = 'exp_ll' if 'exp_ll' in history else 'exp_log_lik'
    exp_ll = np.array(history[exp_ll_key])
    bond_kl = np.array(history['bond_kl'])
    node_kl = np.array(history['node_kl'])
    tau_kl = np.array(history['tau_kl'])
    
    # Stack the KL terms (negative for visualization)
    ax.fill_between(epochs, 0, exp_ll, label='Expected LL', alpha=0.7, color='green')
    ax.fill_between(epochs, exp_ll, exp_ll - bond_kl, label='Bond KL', alpha=0.7, color='red')
    ax.fill_between(epochs, exp_ll - bond_kl, exp_ll - bond_kl - node_kl, 
                     label='Node KL', alpha=0.7, color='orange')
    ax.fill_between(epochs, exp_ll - bond_kl - node_kl, 
                     exp_ll - bond_kl - node_kl - tau_kl, 
                     label='Tau KL', alpha=0.7, color='purple')
    
    # Plot total ELBO
    elbo = exp_ll - bond_kl - node_kl - tau_kl
    ax.plot(epochs, elbo, 'k--', linewidth=2, label='Total ELBO')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.set_title('ELBO Components (Stacked)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_predictions_with_uncertainty(
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma: np.ndarray,
    epistemic: Optional[np.ndarray] = None,
    aleatoric: Optional[np.ndarray] = None,
    title: str = "",
    ax: Optional[Axes] = None
) -> Axes:
    """
    Plot predictions with uncertainty bands
    
    Args:
        x: Input values
        y_true: True target values
        y_pred: Predicted mean values
        sigma: Total uncertainty (standard deviation)
        epistemic: Optional epistemic uncertainty component
        aleatoric: Optional aleatoric uncertainty component
        title: Plot title
        ax: Optional matplotlib axis
        
    Returns:
        The matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Sort by x for clean lines
    sort_idx = np.argsort(x.flatten())
    x_sorted = x.flatten()[sort_idx]
    y_true_sorted = y_true.flatten()[sort_idx]
    y_pred_sorted = y_pred.flatten()[sort_idx]
    sigma_sorted = sigma.flatten()[sort_idx]
    
    # Plot true function
    ax.plot(x_sorted, y_true_sorted, 'k-', linewidth=2, label='True', alpha=0.7)
    
    # Plot predictions
    ax.plot(x_sorted, y_pred_sorted, 'b-', linewidth=2, label='Prediction')
    
    # Plot total uncertainty
    ax.fill_between(x_sorted, 
                     y_pred_sorted - 2*sigma_sorted, 
                     y_pred_sorted + 2*sigma_sorted,
                     alpha=0.3, color='blue', label='±2σ (Total)')
    
    # If we have decomposed uncertainty, plot it
    if epistemic is not None and aleatoric is not None:
        epistemic_sorted = epistemic.flatten()[sort_idx]
        aleatoric_sorted = aleatoric.flatten()[sort_idx]
        
        ax.fill_between(x_sorted,
                         y_pred_sorted - epistemic_sorted,
                         y_pred_sorted + epistemic_sorted,
                         alpha=0.4, color='red', label='Epistemic')
        
        # Show aleatoric as error bars at selected points
        sample_points = np.linspace(0, len(x_sorted)-1, 10).astype(int)
        ax.errorbar(x_sorted[sample_points], y_pred_sorted[sample_points],
                     yerr=aleatoric_sorted[sample_points],
                     fmt='none', ecolor='green', alpha=0.5, linewidth=2,
                     label='Aleatoric')
    
    ax.set_xlabel('Input')
    ax.set_ylabel('Output')
    ax.set_title(title if title else 'Predictions with Uncertainty')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_training_curves(
    metrics_dict: Dict[str, list],
    figsize: tuple = (12, 5)
) -> tuple:
    """
    Plot MSE and R² curves side-by-side
    
    Args:
        metrics_dict: Dictionary with keys 'epoch', 'mse', 'r2'
        figsize: Figure size
        
    Returns:
        Tuple of (fig, axes)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    epochs = metrics_dict['epoch']
    
    # Plot MSE
    ax1 = axes[0]
    ax1.plot(epochs, metrics_dict['mse'], 'b-', marker='o', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE')
    ax1.set_title('Mean Squared Error')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot R²
    ax2 = axes[1]
    ax2.plot(epochs, metrics_dict['r2'], 'g-', marker='s', linewidth=2)
    ax2.axhline(1.0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('R²')
    ax2.set_title('R² Score')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    return fig, axes


def create_multipanel_evolution(
    data_dict: Dict[str, list],
    n_rows: int,
    n_cols: int,
    plot_func: Callable,
    figsize: Optional[tuple] = None
) -> tuple:
    """
    Create multi-panel figure showing evolution over epochs
    
    Args:
        data_dict: Dictionary of data to plot
        n_rows: Number of rows
        n_cols: Number of columns
        plot_func: Function to call for each panel (signature: func(ax, data, index))
        figsize: Optional figure size
        
    Returns:
        Tuple of (fig, axes)
    """
    if figsize is None:
        figsize = (4 * n_cols, 4 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)
    
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            ax = axes[i, j]
            plot_func(ax, data_dict, idx)
    
    plt.tight_layout()
    
    return fig, axes


def save_figure_high_quality(fig: Figure, filename: str, dpi: int = 300) -> None:
    """
    Save figure with consistent styling
    
    Args:
        fig: Matplotlib figure
        filename: Output filename
        dpi: Resolution in dots per inch
    """
    fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved figure: {filename}")
