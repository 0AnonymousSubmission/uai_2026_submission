#!/usr/bin/env python3
"""
Analyze K-Fold cross-validation results.
Computes mean and variance with and without outlier removal.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from typing import Dict, List, Tuple


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


def remove_outliers_iqr(values: List[float], k: float = 1.5) -> Tuple[List[float], int]:
    """
    Remove outliers using IQR method.
    Returns filtered values and count of removed outliers.
    """
    if len(values) < 4:
        return values, 0

    arr = np.array(values)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    filtered = [v for v in values if lower_bound <= v <= upper_bound]
    return filtered, len(values) - len(filtered)


def remove_negative(values: List[float]) -> Tuple[List[float], int]:
    """Remove negative values. Returns filtered values and count of removed."""
    filtered = [v for v in values if v >= 0]
    return filtered, len(values) - len(filtered)


def load_kfold_results(results_dir: str) -> Dict[str, Dict[str, List[float]]]:
    """
    Load all k-fold results from directory structure.
    Expects: results_dir/{dataset}_{model}/summary.json
    
    Returns: {dataset: {model: [test_quality values]}}
    """
    results = defaultdict(lambda: defaultdict(list))
    results_path = Path(results_dir)
    
    for summary_file in results_path.glob("*/summary.json"):
        try:
            with open(summary_file) as f:
                data = json.load(f)
            
            for result in data.get("all_results", []):
                if result.get("singular", False) or not result.get("success", False):
                    continue
                
                dataset = result.get("dataset", "unknown")
                model = result.get("model", "unknown")
                test_quality = result.get("test_quality")
                
                if test_quality is not None:
                    results[dataset][model].append(test_quality)
                    
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load {summary_file}: {e}")
    
    return dict(results)


def compute_stats(values: List[float]) -> Tuple[float, float, int]:
    """Compute mean, std, and count."""
    if not values:
        return 0.0, 0.0, 0
    return float(np.mean(values)), float(np.std(values)), len(values)


def generate_comparison_table(results: Dict[str, Dict[str, List[float]]], outlier_k: float = 1.5, remove_neg: bool = False) -> str:
    """Generate a table comparing raw vs outlier-removed statistics."""
    
    datasets = sorted(results.keys())
    all_models = set()
    for models in results.values():
        all_models.update(models.keys())
    models = sorted(all_models)
    
    lines = []
    lines.append("=" * 120)
    title = "K-FOLD RESULTS COMPARISON: Raw vs Outlier-Removed (IQR method)"
    if remove_neg:
        title += " [negative values removed]"
    lines.append(title)
    lines.append("=" * 120)
    lines.append("")
    
    for model in models:
        lines.append(f"### {model}")
        lines.append("")
        header = f"{'Dataset':<20} | {'Raw Mean':>10} {'Raw Std':>10} {'N':>4} | {'IQR Mean':>10} {'IQR Std':>10} {'N':>4} {'Removed':>4}"
        lines.append("-" * len(header))
        lines.append(header)
        lines.append("-" * len(header))
        
        for dataset in datasets:
            if model not in results.get(dataset, {}):
                continue
            
            values = results[dataset][model]
            if remove_neg:
                values, _ = remove_negative(values)
            
            raw_mean, raw_std, raw_n = compute_stats(values)
            
            filtered, n_removed = remove_outliers_iqr(values, k=outlier_k)
            iqr_mean, iqr_std, iqr_n = compute_stats(filtered)
            
            lines.append(
                f"{dataset:<20} | {raw_mean:>10.4f} {raw_std:>10.4f} {raw_n:>4} | "
                f"{iqr_mean:>10.4f} {iqr_std:>10.4f} {iqr_n:>4} {n_removed:>4}"
            )
        
        lines.append("-" * len(header))
        lines.append("")
    
    return "\n".join(lines)


def format_with_error(mean: float, std: float, bold: bool = False) -> str:
    """Format mean ± std with 2 significant figures on error, mean matching precision."""
    if std == 0:
        decimals = 2
    else:
        from math import floor, log10
        first_sig_fig = floor(log10(abs(std)))
        decimals = max(0, -first_sig_fig + 1)
    
    mean_str = f"{mean:.{decimals}f}"
    std_str = f"{std:.{decimals}f}"
    
    if bold:
        return f"$\\mathbf{{{mean_str} \\pm {std_str}}}$"
    return f"${mean_str} \\pm {std_str}$"


def generate_latex_table(results: Dict[str, Dict[str, List[float]]], outlier_k: float = 1.5, use_iqr: bool = True, remove_neg: bool = False) -> str:
    """Generate LaTeX table."""
    
    datasets = sorted(results.keys())
    all_models = set()
    for models in results.values():
        all_models.update(models.keys())
    models = sorted(all_models)
    
    col_spec = "l" + "c" * len(models)
    
    caption_parts = ["Test Quality (mean $\\pm$ std)"]
    if remove_neg:
        caption_parts.append("negative removed")
    if use_iqr:
        caption_parts.append("IQR outlier removal")
    caption = caption_parts[0] + (" (" + ", ".join(caption_parts[1:]) + ")" if len(caption_parts) > 1 else "")
    
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Dataset & " + " & ".join(models) + r" \\",
        r"\midrule",
    ]
    
    for dataset in datasets:
        abbrev = DATASET_ABBREV.get(dataset, dataset[:2].upper())
        
        row_data = {}
        for model in models:
            values = results.get(dataset, {}).get(model, [])
            if remove_neg:
                values, _ = remove_negative(values)
            if not values:
                row_data[model] = None
                continue
            
            if use_iqr:
                filtered, _ = remove_outliers_iqr(values, k=outlier_k)
                mean, std, _ = compute_stats(filtered)
            else:
                mean, std, _ = compute_stats(values)
            
            row_data[model] = (mean, std)
        
        valid_means = [v[0] for v in row_data.values() if v is not None]
        best_mean = max(valid_means) if valid_means else None
        
        row_cells = [abbrev]
        for model in models:
            data = row_data[model]
            if data is None:
                row_cells.append("-")
            else:
                mean, std = data
                is_best = (best_mean is not None and abs(mean - best_mean) < 1e-9)
                row_cells.append(format_with_error(mean, std, bold=is_best))
        
        lines.append(" & ".join(row_cells) + r" \\")
    
    legend_items = [f"{abbrev}={name}" for name, abbrev in sorted(DATASET_ABBREV.items(), key=lambda x: x[1])]
    legend = ", ".join(legend_items)
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        f"%\\\\\\footnotesize{{{legend}}}",
        r"\end{table*}",
    ])
    
    return "\n".join(lines)


def generate_markdown_table(results: Dict[str, Dict[str, List[float]]], outlier_k: float = 1.5, use_iqr: bool = True, remove_neg: bool = False) -> str:
    """Generate Markdown table."""
    
    datasets = sorted(results.keys())
    all_models = set()
    for models in results.values():
        all_models.update(models.keys())
    models = sorted(all_models)
    
    title_parts = ["## Test Quality"]
    if remove_neg:
        title_parts.append("negative removed")
    if use_iqr:
        title_parts.append("IQR")
    else:
        title_parts.append("Raw")
    title = title_parts[0] + " (" + ", ".join(title_parts[1:]) + ")"
    
    header = "| Dataset | " + " | ".join(models) + " |"
    separator = "|" + "---|" * (len(models) + 1)
    
    lines = [title, "", header, separator]
    
    for dataset in datasets:
        row_cells = [dataset]
        for model in models:
            values = results.get(dataset, {}).get(model, [])
            if remove_neg:
                values, _ = remove_negative(values)
            if not values:
                row_cells.append("-")
                continue
            
            if use_iqr:
                filtered, _ = remove_outliers_iqr(values, k=outlier_k)
                mean, std, _ = compute_stats(filtered)
            else:
                mean, std, _ = compute_stats(values)
            
            row_cells.append(f"{mean:.4f} ± {std:.4f}")
        
        lines.append("| " + " | ".join(row_cells) + " |")
    
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze K-Fold cross-validation results")
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing {dataset}_{model}/summary.json files",
    )
    parser.add_argument(
        "--format",
        choices=["ascii", "markdown", "latex", "all"],
        default="all",
        help="Output format",
    )
    parser.add_argument(
        "--outlier-k",
        type=float,
        default=1.5,
        help="IQR multiplier for outlier detection (default: 1.5)",
    )
    parser.add_argument(
        "--remove-negative",
        action="store_true",
        help="Remove negative quality values before computing stats",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file (default: stdout)",
    )
    
    args = parser.parse_args()
    
    results = load_kfold_results(args.results_dir)
    
    if not results:
        print(f"No results found in {args.results_dir}")
        return 1
    
    total_runs = sum(len(v) for d in results.values() for v in d.values())
    print(f"Loaded {total_runs} runs across {len(results)} datasets", file=__import__('sys').stderr)
    
    output_lines = []
    
    if args.format in ("ascii", "all"):
        output_lines.append(generate_comparison_table(results, outlier_k=args.outlier_k, remove_neg=args.remove_negative))
    
    if args.format in ("markdown", "all"):
        output_lines.append(generate_markdown_table(results, outlier_k=args.outlier_k, use_iqr=False, remove_neg=args.remove_negative))
        output_lines.append(generate_markdown_table(results, outlier_k=args.outlier_k, use_iqr=True, remove_neg=args.remove_negative))
    
    if args.format in ("latex", "all"):
        output_lines.append(generate_latex_table(results, outlier_k=args.outlier_k, use_iqr=False, remove_neg=args.remove_negative))
        output_lines.append("")
        output_lines.append(generate_latex_table(results, outlier_k=args.outlier_k, use_iqr=True, remove_neg=args.remove_negative))
    
    output = "\n".join(output_lines)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Results written to {args.output}")
    else:
        print(output)
    
    return 0


if __name__ == "__main__":
    exit(main())
