#!/usr/bin/env python3
"""
Generate experiment config files for BTN on all UCI regression datasets.

BTN is regression-only, so classification datasets are excluded.

Supports two trimming methods:
- "relevance": E[lambda]-based cumulative relevance trimming
- "variance": sigma diagonal variance-based trimming

Supports warmup: during warmup_epochs, bond q distributions are not updated
(only mu and sigma nodes are updated).
"""

import json
import os

UCI_REGRESSION_DATASETS = [
    ("student_perf", 320),
    ("abalone", 1),
    ("obesity", 544),
    ("bike", 275),
    ("realstate", 477),
    ("energy_efficiency", 242),
    ("concrete", 165),
    ("ai4i", 601),
    ("appliances", 374),
    ("seoulBike", 560),
]

HIGH_FEATURE_DATASETS = [
    "ai4i",
    "appliances",
]

PARAMETER_GRID = {
    "L": [3, 4],
    "bond_dim": [18],
    "trimming_threshold": [0.1, 0.01],
    "trim_method": ["gamma"],
}

CPD_BOND_DIM = [64]

FIXED_PARAMS = {
    "batch_size": 256,
    "n_epochs": 53,
    "patience": 100,
    "min_delta": 0.0001,
    "soft_trim_relaxation": 1,
    "trim_every": 6,
    "init_strength": 0.01,
    "seeds": [42, 7, 123, 256, 999],
    "warmup_epochs": 5,
    "bond_prior_alpha": 1.0,
}

TRACKER_CONFIG = {
    "backend": "file",
    "tracker_dir": "runs_init001_prior1",
    "aim_repo": "",
}


def create_config(
    dataset_name: str,
    model: str,
    added_bias: bool = False,
    grid_overrides: dict | None = None,
    fixed_overrides: dict | None = None,
) -> dict:
    grid = {"model": [model], **PARAMETER_GRID}
    if model == "CPD":
        grid["bond_dim"] = CPD_BOND_DIM
    if grid_overrides:
        grid.update(grid_overrides)

    fixed = FIXED_PARAMS.copy()
    fixed["added_bias"] = added_bias
    if fixed_overrides:
        fixed.update(fixed_overrides)

    bias_tag = "bias" if added_bias else "nobias"
    init_val = fixed["init_strength"]
    prior_val = fixed["bond_prior_alpha"]
    
    base_dir = f"results_init{init_val}_prior{prior_val}"
    runs_dir = f"runs_init{init_val}_prior{prior_val}"
    
    tracker = TRACKER_CONFIG.copy()
    tracker["tracker_dir"] = f"{runs_dir}_{bias_tag}"
    
    return {
        "experiment_name": f"btn_{dataset_name}",
        "dataset": dataset_name,
        "task": "regression",
        "parameter_grid": grid,
        "fixed_params": fixed,
        "tracker": tracker,
        "output": {
            "results_dir": f"{base_dir}/btn_{dataset_name}_{model}_{bias_tag}",
            "save_models": False,
            "save_individual_runs": True,
        },
    }


def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    configs_created = []

    for dataset_name, _ in UCI_REGRESSION_DATASETS:
        include_btt = dataset_name not in HIGH_FEATURE_DATASETS

        for added_bias in [False, True]:
            bias_tag = "bias" if added_bias else "nobias"
            
            configs = [
                (f"btn_{dataset_name}_mpo2_{bias_tag}.json", create_config(dataset_name, "MPO2", added_bias=added_bias)),
                (
                    f"btn_{dataset_name}_lmpo2_{bias_tag}.json",
                    create_config(
                        dataset_name,
                        "LMPO2",
                        added_bias=added_bias,
                        grid_overrides={"reduction_factor": [1.0]},
                        fixed_overrides={"mpo_bond_dim": 1},
                    ),
                ),
                (f"btn_{dataset_name}_cpd_{bias_tag}.json", create_config(dataset_name, "CPD", added_bias=added_bias)),
            ]

            if include_btt:
                configs.append((f"btn_{dataset_name}_btt_{bias_tag}.json", create_config(dataset_name, "BTT", added_bias=added_bias)))

            for name, config in configs:
                filepath = os.path.join(output_dir, name)
                with open(filepath, "w") as f:
                    json.dump(config, f, indent=2)
                configs_created.append(name)
                print(f"Created: {name}")

    print(f"\n{'=' * 60}")
    print(f"Total: {len(configs_created)} config files")
    print(f"Regression datasets: {len(UCI_REGRESSION_DATASETS)}")
    print(f"BTT excluded for high-feature datasets: {HIGH_FEATURE_DATASETS}")


if __name__ == "__main__":
    main()
