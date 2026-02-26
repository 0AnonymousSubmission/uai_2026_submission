# Recovering Results from Run Files

When experiments crash mid-training, results may not be saved properly. However, run files in `runs_kfold_btn/` contain `metrics_log` with training history that can be used to rebuild results.

## Usage

```bash
# Dry run (preview what will be extracted)
python scripts/recover_results_from_runs.py --dry-run

# Recover to a separate folder (recommended)
python scripts/recover_results_from_runs.py --results-dir results_from_runs/kfold_btn

# Recover specific dataset/model only
python scripts/recover_results_from_runs.py --folder energy_efficiency_BTT --results-dir results_from_runs/kfold_btn
```

## How It Works

1. Finds the epoch with best `val_quality` in `metrics_log`
2. Extracts `test_quality` with priority:
   - `metrics_log` at best val epoch (preferred — summary often has only initial value)
   - Last logged `test_quality` in `metrics_log`
   - `summary.test_quality` (fallback if not in metrics_log)
3. Sets `singular=False` so results are included in analysis
4. Generates individual result files + `summary.json` per folder

## Limitations

Runs with **0 extractions** have no `test_quality` data:
- Empty `metrics_log` (crashed before training)
- No `test_quality` in log or summary (test eval never ran)

These experiments must be re-run to get test results.

## Verify with Analysis

```bash
python scripts/analyze_kfold_results.py --results-dir results_from_runs/kfold_btn --format latex
```
