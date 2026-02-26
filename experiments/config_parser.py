# type: ignore
"""
Configuration parser for BTN grid search experiments.
Handles parameter grid expansion and validation.

BTN-specific hyperparameters:
- trimming_threshold: Bond trimming threshold (0 = no trimming)
- model: Model architecture type
- init_strength: Weight initialization strength
- seed: Random seed
- L: Number of sites
- bond_dim: Bond dimension
- n_epochs: Training epochs
- patience: Early stopping patience
- trim_every: Trim bonds every N epochs
"""

import json
import itertools
from typing import Dict, List, Any, Tuple


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        config = json.load(f)

    required_fields = ["experiment_name", "dataset", "parameter_grid"]
    missing = [f for f in required_fields if f not in config]
    if missing:
        raise ValueError(f"Missing required fields in config: {missing}")

    if "fixed_params" not in config:
        config["fixed_params"] = {}

    if "tracker" not in config:
        config["tracker"] = {"backend": "file", "tracker_dir": "experiment_logs", "aim_repo": None}

    if "output" not in config:
        config["output"] = {
            "results_dir": f"results/{config['experiment_name']}",
            "save_models": False,
            "save_individual_runs": True,
        }

    return config


def expand_parameter_grid(parameter_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Expand parameter grid into list of all combinations.

    Args:
        parameter_grid: Dict mapping parameter names to lists of values

    Returns:
        List of dicts, each representing one parameter combination
    """
    param_names = list(parameter_grid.keys())
    param_values = [parameter_grid[name] for name in param_names]
    combinations = list(itertools.product(*param_values))
    expanded = [dict(zip(param_names, combo)) for combo in combinations]
    return expanded


def merge_params(grid_params: Dict[str, Any], fixed_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge grid parameters with fixed parameters.
    Grid parameters take precedence over fixed parameters.
    """
    merged = fixed_params.copy()
    merged.update(grid_params)
    return merged


def generate_run_name(grid_params: Dict[str, Any], seed: int, method: str) -> str:
    """
    Generate a unique run name from grid parameters and seed.
    Format: L{L}_d{bond_dim}_init{init}_trim{trim}_{method}_s{seed}
    """
    L = grid_params.get("L", "X")
    bond_dim = grid_params.get("bond_dim", "X")
    init = grid_params.get("init_strength", "X")
    trim = grid_params.get("trimming_threshold", 0)
    trim_method = grid_params.get("trim_method", "relevance")
    bias = grid_params.get("added_bias", False)

    bias = "(yes)" if bias else "(no)"
    if isinstance(init, (int, float)):
        if init < 0.01:
            init_str = f"{init:.0e}".replace("-", "m")
        else:
            init_str = f"{init:.3f}".rstrip("0").rstrip(".")
    else:
        init_str = str(init)

    if isinstance(trim, (int, float)):
        trim_str = f"{trim:.2f}".rstrip("0").rstrip(".")
    else:
        trim_str = str(trim)

    return f"{method}_L{L}_d{bond_dim}_init{init_str}_trim{trim_str}_{trim_method}_s{seed}_bias{bias}"


def generate_run_id(grid_params: Dict[str, Any], seed: int, method: str = None) -> str:
    """
    Generate a unique, filesystem-safe run ID from grid parameters and seed.
    Format: {method}-{model}-L{L}-d{bond_dim}-trim{threshold}-{trim_method}-bias{0/1}-seed{seed}
    
    Args:
        grid_params: Dictionary of grid search parameters
        seed: Random seed
        method: Training method identifier (e.g., "BTN", "ALS")
    """
    parts = []

    if method:
        parts.append(method)

    parts.append(grid_params.get("model", "MPO2"))

    if "L" in grid_params:
        parts.append(f"L{grid_params['L']}")
    if "bond_dim" in grid_params:
        parts.append(f"d{grid_params['bond_dim']}")

    if "trimming_threshold" in grid_params:
        trim_val = grid_params["trimming_threshold"]
        parts.append(f"trim{trim_val:.2f}".rstrip("0").rstrip("."))

    if "trim_method" in grid_params:
        parts.append(grid_params["trim_method"])

    if "reduction_factor" in grid_params:
        rf = grid_params["reduction_factor"]
        parts.append(f"rf{rf:.2f}".rstrip("0").rstrip("."))

    if "added_bias" in grid_params:
        bias_val = 1 if grid_params["added_bias"] else 0
        parts.append(f"bias{bias_val}")

    parts.append(f"seed{seed}")

    return "-".join(parts)


BTN_RELEVANT_PARAMS = [
    "L",
    "bond_dim",
    "init_strength",
    "trimming_threshold",
    "trim_method",
    "n_epochs",
    "patience",
    "trim_every",
    "batch_size",
    "warmup_epochs",
    "bond_prior_alpha",
]


def get_relevant_params() -> List[str]:
    """Get list of parameters that affect BTN training."""
    return BTN_RELEVANT_PARAMS


def validate_params(params: Dict[str, Any]) -> None:
    """
    Validate BTN parameters.

    Raises:
        ValueError: If required parameters are missing
    """
    required = ["L", "bond_dim"]
    missing = [p for p in required if p not in params]
    if missing:
        raise ValueError(f"Missing required parameters: {missing}")


def get_default_params() -> Dict[str, Any]:
    """
    Get default parameters for BTN.
    """
    return {
        "init_strength": 0.1,
        "trimming_threshold": 0.95,
        "trim_method": "relevance",
        "n_epochs": 50,
        "patience": 10,
        "trim_every": None,
        "batch_size": 64,
        "warmup_epochs": 0,
        "bond_prior_alpha": 5.0,
    }


def create_experiment_plan(config: Dict[str, Any], method: str = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Create complete experiment plan from config.

    Args:
        config: Parsed config dictionary
        method: Training method identifier (e.g., "BTN", "ALS"). If None, uses config["method"] or defaults to empty.

    Returns:
        experiments: List of experiment configs (one per grid combination × seed)
        metadata: Dict with experiment metadata (total count, grid size, etc.)
    """
    method = method or config.get("method", None)
    grid_combinations = expand_parameter_grid(config["parameter_grid"])

    seeds = config["fixed_params"].get("seeds", [0])
    if not isinstance(seeds, list):
        seeds = [seeds]

    experiments = []

    for grid_params in grid_combinations:
        defaults = get_default_params()

        full_params = {}
        full_params.update(defaults)
        full_params.update(config["fixed_params"])
        full_params.update(grid_params)

        for seed in seeds:
            try:
                validate_params(full_params)
            except ValueError as e:
                print(f"Warning: Skipping invalid configuration: {e}")
                continue

            experiment = {
                "experiment_name": config["experiment_name"],
                "dataset": config["dataset"],
                "task": "regression",
                "params": full_params,
                "seed": seed,
                "method": method,
                "run_name": generate_run_name(grid_params, seed, method=method),
                "run_id": generate_run_id(grid_params, seed, method=method),
                "grid_params": grid_params,
                "tracker": config["tracker"],
                "output": config["output"],
            }

            experiments.append(experiment)

    metadata = {
        "total_experiments": len(experiments),
        "grid_size": len(grid_combinations),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "parameter_grid": config["parameter_grid"],
        "fixed_params": config["fixed_params"],
    }

    return experiments, metadata


def print_experiment_summary(experiments: List[Dict[str, Any]], metadata: Dict[str, Any]) -> None:
    """Print a summary of the experiment plan."""
    print("=" * 70)
    print("EXPERIMENT PLAN SUMMARY")
    print("=" * 70)
    print(f"Total experiments: {metadata['total_experiments']}")
    print(f"Grid combinations: {metadata['grid_size']}")
    print(f"Seeds per combination: {metadata['n_seeds']}")
    print()

    print("Parameter Grid:")
    for param, values in metadata["parameter_grid"].items():
        print(f"  {param}: {values} (n={len(values)})")
    print()

    print("Fixed Parameters:")
    for param, value in metadata["fixed_params"].items():
        if param != "seeds":
            print(f"  {param}: {value}")
    print()

    print("Example runs (first 5):")
    for i, exp in enumerate(experiments[:5]):
        print(f"  {i + 1}. {exp['run_id']}")

    if len(experiments) > 5:
        print(f"  ... ({len(experiments) - 5} more)")
    print()
