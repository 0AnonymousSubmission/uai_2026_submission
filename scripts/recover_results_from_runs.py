#!/usr/bin/env python3
"""
Rebuild results from run files by extracting test_quality at best val_quality epoch.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np


def extract_best_metrics(run_file: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(run_file, "r") as f:
            run_data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None
    metrics_log = run_data.get("metrics_log", [])
    if not metrics_log:
        return None
    best_step = None
    best_val = -float("inf")
    last_test_quality = None
    last_test_step = None
    for entry in metrics_log:
        val = entry.get("val_quality")
        if val is not None and val > best_val:
            best_val = val
            best_step = entry
        if entry.get("test_quality") is not None:
            last_test_quality = entry.get("test_quality")
            last_test_step = entry
    
    if best_step is None:
        return None
    
    # Get test_quality - prefer metrics_log (summary often has only initial value)
    # Priority: metrics_log at best_step > last logged in metrics_log > summary
    test_quality = best_step.get("test_quality")
    test_loss = best_step.get("test_loss")
    
    if test_quality is None and last_test_quality is not None:
        test_quality = last_test_quality
        test_loss = last_test_step.get("test_loss") if last_test_step else None
    
    # Fallback: check summary (only if not in metrics_log)
    if test_quality is None:
        summary = run_data.get("summary", {})
        test_quality = summary.get("test_quality")
        test_loss = summary.get("test_loss")
    
    if test_quality is None:
        return None
    
    config = run_data.get("config", {})
    hparams = run_data.get("hparams", {})
    return {
        "fold": config.get("fold"),
        "seed": config.get("seed"),
        "model": hparams.get("model"),
        "dataset": hparams.get("dataset"),
        "params": config.get("params", {}),
        "n_features": hparams.get("n_features"),
        "n_train": hparams.get("n_train"),
        "n_val": hparams.get("n_val"),
        "n_test": hparams.get("n_test"),
        "train_loss": best_step.get("train_loss"),
        "train_quality": best_step.get("train_quality"),
        "val_loss": best_step.get("val_loss"),
        "val_quality": best_step.get("val_quality"),
        "test_loss": test_loss,
        "test_quality": test_quality,
        "best_epoch": best_step.get("step"),
        "n_parameters": best_step.get("n_parameters"),
        "elbo_raw": best_step.get("elbo_raw"),
        "elbo_relative": best_step.get("elbo_relative"),
        "tau_mean": best_step.get("tau_mean"),
        "success": True,
        "singular": False,
        "recovered_from_run": True,
        "run_id": run_data.get("run_name"),
    }


def build_summary(results: list, results_dir: Path) -> dict:
    if not results:
        return {}
    
    datasets = set(r.get("dataset") for r in results if r.get("dataset"))
    seeds = set(r.get("seed") for r in results if r.get("seed") is not None)
    folds = set(r.get("fold") for r in results if r.get("fold") is not None)
    params = results[0].get("params", {}) if results else {}
    
    aggregated = {}
    for dataset in datasets:
        test_qualities = [
            r["test_quality"] for r in results
            if r.get("dataset") == dataset and r.get("test_quality") is not None
        ]
        if test_qualities:
            aggregated[dataset] = {
                "mean": float(np.mean(test_qualities)),
                "std": float(np.std(test_qualities)),
                "n_runs": len(test_qualities),
                "min": float(np.min(test_qualities)),
                "max": float(np.max(test_qualities)),
            }
    
    return {
        "params": params,
        "seeds": sorted(seeds),
        "n_folds": len(folds) if folds else 5,
        "val_ratio": 0.15,
        "datasets": sorted(datasets),
        "aggregated": aggregated,
        "all_results": results,
    }


def process_folder(runs_folder: Path, results_folder: Path, dry_run: bool = False) -> Dict[str, int]:
    stats = {"runs": 0, "extracted": 0}
    
    run_files = list(runs_folder.glob("*.json"))
    stats["runs"] = len(run_files)
    
    results = []
    for run_file in run_files:
        extracted = extract_best_metrics(run_file)
        if extracted:
            results.append(extracted)
            stats["extracted"] += 1
            
            if not dry_run:
                result_file = results_folder / f"{extracted['run_id']}.json"
                results_folder.mkdir(parents=True, exist_ok=True)
                with open(result_file, "w") as f:
                    json.dump(extracted, f, indent=2)
    
    if results and not dry_run:
        summary = build_summary(results, results_folder)
        with open(results_folder / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Rebuild results from run files")
    parser.add_argument("--runs-dir", default="runs_kfold_btn")
    parser.add_argument("--results-dir", default="results/kfold_btn")
    parser.add_argument("--folder", "-f", help="Process specific folder only")
    parser.add_argument("--dry-run", "-n", action="store_true")
    args = parser.parse_args()
    
    runs_base = Path(args.runs_dir)
    results_base = Path(args.results_dir)
    
    if not runs_base.exists():
        print(f"Error: {runs_base} not found")
        return 1
    
    if args.folder:
        folders = [runs_base / args.folder]
    else:
        folders = sorted([f for f in runs_base.iterdir() if f.is_dir()])
    
    total = {"runs": 0, "extracted": 0}
    
    for folder in folders:
        results_folder = results_base / folder.name
        stats = process_folder(folder, results_folder, args.dry_run)
        total["runs"] += stats["runs"]
        total["extracted"] += stats["extracted"]
        print(f"{folder.name}: {stats['extracted']}/{stats['runs']} runs extracted")
    
    print(f"\nTotal: {total['extracted']}/{total['runs']} runs extracted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
