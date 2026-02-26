# type: ignore
"""
Flexible experiment tracking system supporting multiple backends.

Supported backends:
- 'file': Save metrics to JSON file (epoch-by-epoch)
- 'aim': Track with AIM experiment tracking
- 'none': No tracking (useful for quick tests)
"""

import os
import json
import time
from typing import Dict, Any, Optional, List
from pathlib import Path

MAX_RETRIES = 3
RETRY_DELAY = 10


def _with_retry(operation, error_msg: str):
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return operation()
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                print(f"  AIM error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                print(f"  Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                raise TrackerError(f"{error_msg}: {e}") from e
    raise TrackerError(f"{error_msg}: {last_error}") from last_error


class TrackerError(Exception):
    """Exception raised when tracker operations fail.

    This exception is meant to be fatal - when raised, experiments should stop
    rather than continuing with broken tracking.
    """

    pass


class BaseTracker:
    """Base class for all trackers."""

    def __init__(self, experiment_name: str, config: Dict[str, Any]):
        self.experiment_name = experiment_name
        self.config = config

    def log_hparams(self, hparams: Dict[str, Any]):
        """Log hyperparameters."""
        pass

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics for a specific step/epoch."""
        pass

    def log_summary(self, summary: Dict[str, Any]):
        """Log final summary statistics."""
        pass

    def finalize(self):
        """Cleanup and finalize tracking."""
        pass

    def close(self):
        """Alias for finalize."""
        self.finalize()


class FileTracker(BaseTracker):
    """Track experiments to JSON files."""

    def __init__(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        output_dir: str = "experiment_logs",
        run_name: Optional[str] = None,
    ):
        super().__init__(experiment_name, config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name

        self.hparams = {}
        self.metrics_log = []
        self.summary = {}

    def log_hparams(self, hparams: Dict[str, Any]):
        """Log hyperparameters."""
        self.hparams.update(hparams)

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics for a specific step/epoch."""
        entry = {"step": step, **metrics}
        self.metrics_log.append(entry)

    def log_summary(self, summary: Dict[str, Any]):
        """Log final summary statistics."""
        self.summary.update(summary)

    def finalize(self):
        """Save all data to JSON file."""
        if self.run_name:
            output_file = self.output_dir / f"{self.experiment_name}_{self.run_name}.json"
        else:
            output_file = self.output_dir / f"{self.experiment_name}.json"

        data = {
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "config": self.config,
            "hparams": self.hparams,
            "metrics_log": self.metrics_log,
            "summary": self.summary,
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"  Experiment log saved to: {output_file}")


class AIMTracker(BaseTracker):
    """Track experiments with AIM.

    All tracking operations retry up to 3 times on transient errors
    (e.g., "resource temporarily unavailable") with 10s delay between attempts.
    After all retries fail, raises TrackerError to stop the experiment.
    """

    def __init__(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        repo: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        super().__init__(experiment_name, config)

        try:
            from aim_auth import Run
        except ImportError:
            from aim import Run

        if repo is None:
            repo = os.getenv("AIM_REPO", ".aim")

        self._Run = Run
        self._repo = repo
        self._run_name = run_name

        def init_run():
            self.run = Run(repo=repo, experiment=experiment_name, log_system_params=True)
            if run_name:
                self.run.name = run_name
            self.run["config"] = config
            print(f"  AIM run started: {self.run.hash}")

        _with_retry(init_run, "Failed to initialize AIM run")

    def log_hparams(self, hparams: Dict[str, Any]):
        def op():
            self.run["hparams"] = hparams

        _with_retry(op, "Failed to log hparams to AIM")

    def log_metrics(self, metrics: Dict[str, float], step: int):
        def op():
            for key, value in metrics.items():
                self.run.track(value, name=key, step=step)

        _with_retry(op, f"Failed to log metrics to AIM at step {step}")

    def log_summary(self, summary: Dict[str, Any]):
        def op():
            self.run["summary"] = summary

        _with_retry(op, "Failed to log summary to AIM")

    def finalize(self):
        def op():
            self.run.close()
            print(f"  AIM run finalized: {self.run.hash}")

        _with_retry(op, "Failed to finalize AIM run")


class MultiTracker(BaseTracker):
    """Track to multiple backends simultaneously."""

    def __init__(self, trackers: List[BaseTracker]):
        self.trackers = trackers

    def log_hparams(self, hparams: Dict[str, Any]):
        for tracker in self.trackers:
            tracker.log_hparams(hparams)

    def log_metrics(self, metrics: Dict[str, float], step: int):
        for tracker in self.trackers:
            tracker.log_metrics(metrics, step)

    def log_summary(self, summary: Dict[str, Any]):
        for tracker in self.trackers:
            tracker.log_summary(summary)

    def finalize(self):
        for tracker in self.trackers:
            tracker.finalize()


class NoOpTracker(BaseTracker):
    """No-op tracker for when tracking is disabled."""

    pass


def create_tracker(
    experiment_name: str, config: Dict[str, Any], backend: str = "file", **kwargs
) -> BaseTracker:
    """
    Factory function to create appropriate tracker.

    Args:
        experiment_name: Name of experiment
        config: Configuration dict
        backend: 'file', 'aim', 'both', or 'none'
        **kwargs: Additional arguments for specific trackers
            - output_dir: For file tracker
            - repo: For AIM tracker
            - run_name: Name for this specific run (e.g., seed_0, seed_1)

    Returns:
        Tracker instance
    """
    run_name = kwargs.get("run_name", None)

    if backend == "file":
        return FileTracker(
            experiment_name,
            config,
            output_dir=kwargs.get("output_dir", "experiment_logs"),
            run_name=run_name,
        )

    elif backend == "aim":
        return AIMTracker(experiment_name, config, repo=kwargs.get("repo", None), run_name=run_name)

    elif backend == "both":
        aim_tracker = AIMTracker(
            experiment_name, config, repo=kwargs.get("repo", None), run_name=run_name
        )
        file_tracker = FileTracker(
            experiment_name,
            config,
            output_dir=kwargs.get("output_dir", "experiment_logs"),
            run_name=run_name,
        )
        return MultiTracker([aim_tracker, file_tracker])

    elif backend == "none":
        return NoOpTracker(experiment_name, config)

    else:
        raise ValueError(
            f"Unknown tracking backend: {backend}. Choose from: 'file', 'aim', 'both', 'none'"
        )
