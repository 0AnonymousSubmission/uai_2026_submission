# type: ignore
import os
import sys
import json
import argparse
import time
import math
import torch
import quimb.tensor as qt
from experiments.config_parser import load_config, create_experiment_plan, print_experiment_summary
from experiments.dataset_loader import load_dataset,append_bias 
from experiments.trackers import create_tracker, TrackerError
from experiments.device_utils import DEVICE, move_tn_to_device, move_data_to_device
from tensor.builder import Inputs
from tensor.tn_als import TNALS
from model import MODELS
from model.utils import REGRESSION_METRICS, compute_quality


torch.set_default_dtype(torch.float64)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Grid search experiment runner for TN-ALS (Tensor Network Alternating Least Squares).
Reads JSON configuration and runs all parameter combinations with tracking.
"""


def get_result_filepath(output_dir: str, run_id: str) -> str:
    return os.path.join(output_dir, f"{run_id}.json")


def run_already_completed(output_dir: str, run_id: str) -> tuple:
    """
    Check if a run has already been attempted.

    Returns:
        (was_attempted, was_successful, is_singular, error_message)
        - is_singular: True if failed due to singular matrix (should skip permanently)
    """
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


def create_btn_model(params: dict, n_features: int):
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


def extract_tn_ridge_metrics(tn) -> dict:
    metrics = {
        "bond_dims": {},
    }
    try:
        excluded = set(tn.output_dimensions)
        for bond_tag in tn.q_bonds:
            if bond_tag not in excluded:
                metrics["bond_dims"][bond_tag] = int(tn.mu.ind_size(bond_tag))
    except:
        pass
    return metrics


def run_single_experiment(
    experiment: dict,
    data: dict,
    dataset_info: dict,
    verbose: bool = False,
    tracker=None,
):
    params = experiment["params"]
    seed = experiment["seed"]
    model_name = params.get("model", "MPO2")

    torch.manual_seed(seed)

    n_epochs = params.get("n_epochs", 50)
    batch_size = params.get("batch_size", 64)
    patience = params.get("patience", None)
    min_delta = params.get("min_delta", 0.0)
    bond_prior_alpha = params.get("bond_prior_alpha", 5.0)
    added_bias = params.get("added_bias", False)
    if added_bias:
        data = append_bias(data)

    n_features = data["X_train"].shape[1]

    try:
        model = create_btn_model(params, n_features)
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

        tn = TNALS(
            mu=mu_tn,
            data_stream=train_loader,
            batch_dim="s",
            method="cholesky",
            device=DEVICE,
            bond_prior_alpha=bond_prior_alpha,
        )
        tn.register_data_streams(val_loader, test_loader)

        if tracker:
            hparams = {
                "seed": seed,
                "dataset": experiment["dataset"],
                "n_features": n_features,
                "L": params["L"],
                "bond_dim": params["bond_dim"],
                **params,
            }
            tracker.log_hparams(hparams)

        best_val_quality = float("-inf")
        best_epoch = -1
        stopped_early = False
        patience_counter = 0

        nodes = list(tn.mu.tag_map.keys())

        initial_metrics = extract_tn_ridge_metrics(tn)
        initial_bond_dims = initial_metrics.get("bond_dims", {})

        if tracker:
            init_train_scores = tn.evaluate(REGRESSION_METRICS, data_stream=train_loader)
            init_val_scores = tn.evaluate(REGRESSION_METRICS, data_stream=val_loader)
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

            init_log = {
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
            }
            if "bond_dims" in initial_metrics:
                for bond_tag, dim in initial_metrics["bond_dims"].items():
                    init_log[f"dim_{bond_tag}"] = dim
            tracker.log_metrics(init_log, step=0)

        warmup_epochs = params.get("warmup_epochs", 0)
        bonds = [i for i in tn.mu.ind_map]
        decay = params.get("decay")
        for epoch in range(n_epochs):
            for node_tag in nodes:
                tn.update_mu_node(node_tag)

            if epoch >= warmup_epochs:
                for bond_tag in bonds:
                    c0 = tn.q_bonds[bond_tag].concentration
                    r0 = tn.q_bonds[bond_tag].rate
                    c1 = c0*decay
                    tn.update_bond(bond_tag, c1, r0)
            train_scores = tn.evaluate(REGRESSION_METRICS, data_stream=train_loader)
            val_scores = tn.evaluate(REGRESSION_METRICS, data_stream=val_loader)

            train_quality = compute_quality(train_scores)
            val_quality = compute_quality(val_scores)
            train_loss = train_scores.get("loss", (0, 1))
            if isinstance(train_loss, tuple):
                train_loss = train_loss[0] / train_loss[1] if train_loss[1] != 0 else 0
            val_loss = val_scores.get("loss", (0, 1))
            if isinstance(val_loss, tuple):
                val_loss = val_loss[0] / val_loss[1] if val_loss[1] != 0 else 0

            if val_quality is not None and val_quality > best_val_quality + min_delta:
                best_val_quality = val_quality
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            if tracker:
                metrics = {
                    "train_loss": float(train_loss) if torch.is_tensor(train_loss) else train_loss,
                    "train_quality": float(train_quality) if train_quality is not None else 0.0,
                    "val_loss": float(val_loss) if torch.is_tensor(val_loss) else val_loss,
                    "val_quality": float(val_quality) if val_quality is not None else 0.0,
                    "patience_counter": patience_counter,
                }
                tracker.log_metrics(metrics, step=epoch + 1)

            if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
                print(
                    f"  Epoch {epoch + 1:3d} | Train Quality: {train_quality:.4f} | Val Quality: {val_quality:.4f}"
                )

            if patience is not None and patience_counter >= patience:
                if verbose:
                    print(f"\n  Early stopping at epoch {epoch + 1}")
                stopped_early = True
                break

        test_scores = tn.evaluate(REGRESSION_METRICS, data_stream=test_loader)
        test_quality = compute_quality(test_scores)
        test_loss = test_scores.get("loss", (0, 1))
        if isinstance(test_loss, tuple):
            test_loss = test_loss[0] / test_loss[1] if test_loss[1] != 0 else 0

        final_metrics = extract_tn_ridge_metrics(tn)

        result = {
            "run_id": experiment["run_id"],
            "seed": seed,
            "model": model_name,
            "dataset": experiment["dataset"],
            "params": params,
            "n_features": n_features,
            "train_loss": float(train_loss) if torch.is_tensor(train_loss) else train_loss,
            "train_quality": float(train_quality) if train_quality is not None else 0.0,
            "val_loss": float(val_loss) if torch.is_tensor(val_loss) else val_loss,
            "val_quality": float(best_val_quality) if best_val_quality != float("-inf") else 0.0,
            "test_loss": float(test_loss) if torch.is_tensor(test_loss) else test_loss,
            "test_quality": float(test_quality) if test_quality is not None else 0.0,
            "best_epoch": best_epoch,
            "stopped_early": stopped_early,
            "patience_counter": patience_counter,
            "initial_bond_dims": initial_bond_dims,
            "final_bond_dims": final_metrics["bond_dims"],
            "success": True,
            "singular": False,
        }

        if tracker:
            tracker.log_summary(
                {
                    "test_quality": result["test_quality"],
                    "test_loss": result["test_loss"],
                    "best_val_quality": result["val_quality"],
                    "n_parameters": sum(t.data.numel() for t in tn.mu.tensors),
                }
            )

        return result

    except torch.linalg.LinAlgError as e:
        return {
            "run_id": experiment["run_id"],
            "seed": seed,
            "model": model_name,
            "dataset": experiment["dataset"],
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
                "run_id": experiment["run_id"],
                "seed": seed,
                "model": model_name,
                "dataset": experiment["dataset"],
                "params": params,
                "success": True,
                "singular": True,
                "error": str(e),
            }
        raise

    except Exception as e:
        import traceback

        return {
            "run_id": experiment["run_id"],
            "seed": seed,
            "model": model_name,
            "dataset": experiment["dataset"],
            "params": params,
            "success": False,
            "singular": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def is_grid_complete(output_dir: str) -> bool:
    """Check if grid search was already completed."""
    complete_file = os.path.join(output_dir, ".complete")
    return os.path.exists(complete_file)


def mark_grid_complete(output_dir: str) -> None:
    """Mark grid search as complete by creating .complete file."""
    complete_file = os.path.join(output_dir, ".complete")
    with open(complete_file, "w") as f:
        f.write("")


def main():
    parser = argparse.ArgumentParser(description="Run TNALS grid search experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    parser.add_argument(
        "--tracker",
        type=str,
        default="file",
        choices=["file", "aim", "both", "none"],
        help="Tracking backend",
    )
    parser.add_argument(
        "--tracker-dir", type=str, default="experiment_logs", help="Directory for file tracker logs"
    )
    parser.add_argument(
        "--aim-repo", type=str, default=None, help="AIM repository (local path or aim://host:port)"
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")
    parser.add_argument("--force", action="store_true", help="Remove existing results and re-run")

    args = parser.parse_args()

    config = load_config(args.config)
    experiment_plan, metadata = create_experiment_plan(config, method="ALS")

    if args.tracker == "file" and "tracker" in config:
        args.tracker = config["tracker"].get("backend", "file")
    if args.tracker_dir == "experiment_logs" and "tracker" in config:
        args.tracker_dir = config["tracker"].get("tracker_dir", "experiment_logs")
    if args.aim_repo is None and "tracker" in config:
        args.aim_repo = config["tracker"].get("aim_repo", None)

    args.output_dir = config["output"]["results_dir"]

    if args.force and os.path.exists(args.output_dir):
        import shutil
        shutil.rmtree(args.output_dir)
        print(f"Removed existing results: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    if is_grid_complete(args.output_dir):
        print(f"Grid search already complete. Found .complete file in {args.output_dir}")
        print("Delete .complete file to re-run experiments.")
        return

    dataset_name = config["dataset"]
    print(f"\nLoading dataset: {dataset_name}")
    data, dataset_info = load_dataset(dataset_name)

    data = move_data_to_device(data)

    print(f"  Task: regression (TNALS)")
    print(f"  Device: {DEVICE}")
    print(f"  Train: {dataset_info['n_train']} samples")
    print(f"  Val: {dataset_info['n_val']} samples")
    print(f"  Test: {dataset_info['n_test']} samples")
    print(f"  Features: {dataset_info['n_features']}")

    print_experiment_summary(experiment_plan, metadata)

    results = []
    skipped_count = 0

    start_time = time.time()

    for idx, experiment in enumerate(experiment_plan, 1):
        run_id = experiment["run_id"]

        was_attempted, was_successful, is_singular, error = run_already_completed(
            args.output_dir, run_id
        )
        if was_attempted:
            if was_successful:
                if args.verbose:
                    print(f"[{idx}/{len(experiment_plan)}] {run_id} - SKIPPED (success)")
                skipped_count += 1
                continue
            elif is_singular:
                print(f"[{idx}/{len(experiment_plan)}] {run_id} - SKIPPED (singular matrix)")
                skipped_count += 1
                continue
            else:
                err_short = (error[:80] + "...") if error and len(error) > 80 else error
                print(f"[{idx}/{len(experiment_plan)}] {run_id} - RETRYING ({err_short})")

        print(f"\n[{idx}/{len(experiment_plan)}] Running: {run_id}")

        aim_repo = (
            args.aim_repo
            or config.get("tracker", {}).get("aim_repo")
            or os.getenv("AIM_REPO")
        )

        params = experiment["params"]
        model_name = params.get("model", "MPO2")
        tracker = create_tracker(
            experiment_name=config["experiment_name"],
            config=experiment,
            backend=args.tracker,
            output_dir=args.tracker_dir,
            repo=aim_repo,
            run_name=experiment["run_name"] + model_name,
        )

        result = run_single_experiment(
            experiment=experiment,
            data=data,
            dataset_info=dataset_info,
            verbose=args.verbose,
            tracker=tracker,
        )

        if tracker and hasattr(tracker, "close"):
            tracker.close()

        result_file = get_result_filepath(args.output_dir, run_id)
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)

        if not result["success"]:
            print(f"  ✗ FAILED: {result.get('error', 'Unknown error')}")
            raise RuntimeError(f"Experiment failed: {result.get('error')}")

        results.append(result)

        if result.get("singular"):
            print(f"  ⊘ SINGULAR MATRIX - skipped permanently")
        else:
            print(
                f"  ✓ Test Quality={result['test_quality']:.4f} | Val Quality={result['val_quality']:.4f}"
            )

    elapsed_time = time.time() - start_time

    non_singular = [r for r in results if not r.get("singular")]
    singular_count = len(results) - len(non_singular)

    print("\n" + "=" * 70)
    print("GRID SEARCH COMPLETE")
    print("=" * 70)
    print(f"Total experiments: {len(experiment_plan)}")
    print(f"Skipped (prior): {skipped_count}")
    print(f"Ran: {len(results)}")
    print(f"Successful: {len(non_singular)}")
    print(f"Singular: {singular_count}")
    if len(results) > 0:
        print(f"Time elapsed: {elapsed_time:.1f}s ({elapsed_time / len(results):.1f}s per run)")
    else:
        print(f"Time elapsed: {elapsed_time:.1f}s")
    print()

    if non_singular:
        results_sorted = sorted(non_singular, key=lambda r: r["test_quality"], reverse=True)

        print(f"Top 5 Runs (by test Quality):")
        print()
        for i, r in enumerate(results_sorted[:5], 1):
            print(f"{i}. {r['run_id']}")
            print(f"   Test Quality: {r['test_quality']:.4f}")
            print(f"   Val Quality: {r['val_quality']:.4f}")
            print(f"   L={r['params']['L']}, bond_dim={r['params']['bond_dim']}, seed={r['seed']}")
            print()
    elif len(results) == 0:
        print("All experiments already completed. Check results directory for existing results.")
        print()

    summary = {
        "experiment_name": config["experiment_name"],
        "dataset": config["dataset"],
        "total_experiments": len(results),
        "successful": len(non_singular),
        "singular": singular_count,
        "elapsed_time": elapsed_time,
        "results": results,
        "metadata": metadata,
        "config": config,
    }

    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to: {args.output_dir}")
    print(f"Summary: {summary_file}")

    mark_grid_complete(args.output_dir)
    print("Grid search complete. Marked as .complete")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except TrackerError as e:
        print(f"\n[FATAL] Tracker error - terminating job: {e}", file=sys.stderr)
        sys.stderr.flush()
        sys.stdout.flush()
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Job cancelled by user", file=sys.stderr)
        sys.exit(130)
