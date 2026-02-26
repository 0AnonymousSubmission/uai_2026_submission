#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

DATASET_ABBREV = {
    "abalone": "AB",
    "ai4i": "AI",
    "appliances": "AP",
    "bike": "BK",
    "concrete": "CO",
    "energy_efficiency": "EE",
    "obesity": "OB",
    "realstate": "RS",
    "seoulBike": "SB",
    "student_perf": "SP",
}

def load_runs(runs_dir="runs_kfold_btn"):
    runs = []
    for subdir in Path(runs_dir).iterdir():
        if not subdir.is_dir():
            continue
        for f in subdir.glob("*.json"):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                hparams = data.get("hparams", {})
                config = data.get("config", {})
                
                max_val_per_params = defaultdict(lambda: float('-inf'))
                for m in data.get("metrics_log", []):
                    n_params = m.get("n_parameters")
                    val_q = m.get("val_quality")
                    if n_params is not None and val_q is not None:
                        max_val_per_params[n_params] = max(max_val_per_params[n_params], val_q)
                
                runs.append({
                    "dataset": hparams.get("dataset"),
                    "model": hparams.get("model"),
                    "seed": config.get("seed"),
                    "fold": config.get("fold"),
                    "max_val_per_params": dict(max_val_per_params),
                })
            except Exception as e:
                print(f"Error loading {f}: {e}")
    return runs

def main():
    runs = load_runs()
    print(f"Loaded {len(runs)} runs")
    excluded_dataset = ["student_perf", "energy_efficiency"]
    grouped = defaultdict(lambda: defaultdict(list))
    for r in runs:
        key = (r["dataset"], r["model"])
        for n_params, max_val in r["max_val_per_params"].items():
            grouped[key][n_params].append(max_val)
    
    datasets = sorted(set(r["dataset"] for r in runs if r["dataset"] not in excluded_dataset))
    models = sorted(set(r["model"] for r in runs))
    colors = {"MPS": "blue", "LMPO2": "green", "BTT": "red", "CPD": "orange"}
    num_datasets = len(datasets)
    cols = 4
    rows = (num_datasets + cols - 1) // cols
    
    # Use figsize that scales with rows
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        is_left_edge = (idx % cols == 0)
        is_bottom_edge = (idx >= (rows - 1) * cols) or (idx + cols >= num_datasets)
        for model in models:
            key = (dataset, model)
            if key not in grouped:
                continue
            params_data = grouped[key]
            x = sorted(params_data.keys())
            y = [np.mean(params_data[p]) for p in x]
            model_label = model if model != "MPO2" else "MPS"
            
            # Increased 's' (size) to 60 and added edgecolors for visibility
            ax.scatter(x, y, label=model_label, color=colors.get(model_label, "gray"), 
                       alpha=0.8, s=60, edgecolors='white', linewidth=0.5)
        
        ax.set_xscale('log')
        if is_left_edge:
            ax.set_ylabel("Validation Quality", fontsize=12)
        if is_bottom_edge:
            ax.set_xlabel("Number of Parameters", fontsize=12)
        # ax.set_xlabel("n_parameters", fontsize=10)
        # ax.set_ylabel("max val_quality", fontsize=10)
        title = DATASET_ABBREV.get(dataset, dataset)
        ax.set_title(title, fontweight='bold')
        ax.grid(True, which="both", ls="-", alpha=0.2) # Better grid for log scale
        ax.set_ylim(0, 1.1)
    
    for idx in range(len(datasets), len(axes)):
        axes[idx].set_visible(False)

    # Move legend to the bottom (below the plots)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), 
               ncol=len(models), fontsize=14, frameon=True, shadow=True)

    # Adjust layout: bottom margin (0.1) creates space for the legend
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    
    output_name = "btn_val_quality_vs_params.pdf"
    plt.savefig(output_name, bbox_inches='tight')
    print(f"Saved {output_name}")
if __name__ == "__main__":
    main()
