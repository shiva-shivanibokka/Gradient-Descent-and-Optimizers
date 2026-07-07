"""
logger.py
=========
MLflow experiment tracking wrapper.

Provides a clean, typed interface over MLflow's API so the rest of the
codebase never has to import mlflow directly.

Key design decisions:
  - Context manager protocol: ``with ExperimentLogger(cfg) as log:``
    automatically starts and ends the MLflow run.
  - ``log_run()`` accepts a ``TrainingResult`` and logs everything in
    one call — params, metrics per epoch, and final summary metrics.
  - All MLflow calls are wrapped in try/except so a tracking server
    outage never crashes a training run.

Usage
-----
    from gdo.experiment.logger import ExperimentLogger
    from gdo.config import MLflowConfig

    cfg = MLflowConfig(experiment_name="adam-vs-sgd", run_name="adam_lr0.001")
    with ExperimentLogger(cfg) as log:
        log.log_params({"optimizer": "adam", "lr": 0.001})
        for epoch, metrics in training_loop():
            log.log_epoch(epoch, metrics.to_dict())
        log.log_summary(result)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gdo.config import MLflowConfig
    from gdo.training.trainer import TrainingResult

logger = logging.getLogger(__name__)

# Lazy import so mlflow is not required just to import the module
try:
    import mlflow
    import mlflow.tracking  # ensure sub-module is available

    _mlflow_available = True
except ImportError:
    mlflow = None  # type: ignore[assignment]
    _mlflow_available = False
    logger.warning("mlflow not installed — experiment logging is disabled.")


class ExperimentLogger:
    """
    MLflow experiment tracking wrapper.

    Parameters
    ----------
    config:
        ``MLflowConfig`` from ``gdo.config``.

    Example
    -------
    >>> with ExperimentLogger(cfg.mlflow) as log:
    ...     log.log_params({"lr": 0.001, "optimizer": "adam"})
    ...     result = trainer.fit()
    ...     log.log_run(result)
    """

    def __init__(self, config: MLflowConfig) -> None:
        self.config = config
        self._run: Any = None
        self._active = False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> ExperimentLogger:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end(status="FAILED" if exc_type else "FINISHED")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start an MLflow run. Idempotent if already started."""
        if not _mlflow_available or not self.config.enabled or self._active:
            return
        try:
            # MLflow 3.14+ raises on the local file store unless this opt-out is set.
            # This project's default workflow (local ./mlruns + `mlflow ui`) relies on
            # it, so keep the file store working across mlflow versions.
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
            mlflow.set_tracking_uri(self.config.tracking_uri)
            mlflow.set_experiment(self.config.experiment_name)
            self._run = mlflow.start_run(
                run_name=self.config.run_name,
                tags=self.config.tags,
            )
            self._active = True
            logger.info(
                "[ExperimentLogger] MLflow run started | experiment='%s' | run_id='%s'",
                self.config.experiment_name,
                self._run.info.run_id,
            )
        except Exception as e:
            logger.warning("[ExperimentLogger] Failed to start MLflow run: %s", e)

    def end(self, status: str = "FINISHED") -> None:
        """End the active MLflow run."""
        if not _mlflow_available or not self._active:
            return
        try:
            mlflow.end_run(status=status)
            self._active = False
            logger.info("[ExperimentLogger] MLflow run ended with status=%s", status)
        except Exception as e:
            logger.warning("[ExperimentLogger] Failed to end MLflow run: %s", e)

    # ------------------------------------------------------------------
    # Logging methods
    # ------------------------------------------------------------------

    def log_params(self, params: dict[str, Any]) -> None:
        """Log a dict of hyperparameters."""
        if not _mlflow_available or not self._active:
            return
        try:
            mlflow.log_params(params)
            logger.debug("[ExperimentLogger] Logged %d params.", len(params))
        except Exception as e:
            logger.warning("[ExperimentLogger] log_params failed: %s", e)

    def log_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        """Log a dict of per-epoch metrics at step ``epoch``."""
        if not _mlflow_available or not self._active:
            return
        try:
            mlflow.log_metrics(metrics, step=epoch)
        except Exception as e:
            logger.warning("[ExperimentLogger] log_epoch failed at epoch %d: %s", epoch, e)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a single metric value."""
        if not _mlflow_available or not self._active:
            return
        try:
            mlflow.log_metric(key, value, step=step)
        except Exception as e:
            logger.warning("[ExperimentLogger] log_metric '%s' failed: %s", key, e)

    def log_summary(self, result: TrainingResult) -> None:
        """
        Log a complete ``TrainingResult`` to MLflow.

        Logs:
        - Final summary metrics (best_val_loss, best_val_acc, etc.)
        - Gradient norm summary
        - Total training time
        """
        if not _mlflow_available or not self._active:
            return
        try:
            summary = {
                "best_val_loss": result.best_val_loss,
                "best_val_acc": result.best_val_acc,
                "best_epoch": float(result.best_epoch),
                "total_train_time_s": result.total_train_time_s,
                "num_epochs": float(len(result.history)),
            }
            if result.convergence_epoch is not None:
                summary["convergence_epoch"] = float(result.convergence_epoch)
            summary.update({f"grad_{k}": v for k, v in result.grad_norm_summary.items()})
            mlflow.log_metrics(summary)
            logger.info("[ExperimentLogger] Summary metrics logged.")
        except Exception as e:
            logger.warning("[ExperimentLogger] log_summary failed: %s", e)

    def log_run(self, result: TrainingResult, experiment_config: Any = None) -> None:
        """
        Log a complete training run in one call.

        Logs params from ExperimentConfig, per-epoch metrics, and summary.

        Parameters
        ----------
        result:
            TrainingResult from Trainer.fit().
        experiment_config:
            Optional ExperimentConfig — its ``to_flat_dict()`` is logged
            as MLflow params.
        """
        if not _mlflow_available or not self._active:
            return

        # Log params
        params: dict[str, Any] = {
            "optimizer": result.optimizer_name,
            "scheduler": result.scheduler_name,
            "epochs": result.config.epochs,
            "batch_size": result.config.batch_size,
            "seed": result.config.seed,
            "grad_clip": str(result.config.grad_clip),
        }
        if experiment_config is not None:
            params.update(experiment_config.to_flat_dict())
        self.log_params(params)

        # Log per-epoch metrics
        for m in result.history:
            self.log_epoch(m.epoch, m.to_dict())

        # Log summary
        self.log_summary(result)

    def log_artifact(self, local_path: str | Path) -> None:
        """Log a file (plot, notebook output, etc.) as an MLflow artifact."""
        if not _mlflow_available or not self._active:
            return
        try:
            mlflow.log_artifact(str(local_path))
            logger.debug("[ExperimentLogger] Artifact logged: %s", local_path)
        except Exception as e:
            logger.warning("[ExperimentLogger] log_artifact failed: %s", e)

    def log_figure(self, fig: Any, filename: str) -> None:
        """
        Log a matplotlib or plotly figure directly.

        Parameters
        ----------
        fig:
            matplotlib Figure or plotly Figure.
        filename:
            Filename under which the figure is saved in MLflow artifacts.
        """
        if not _mlflow_available or not self._active:
            return
        try:
            mlflow.log_figure(fig, filename)
            logger.debug("[ExperimentLogger] Figure logged: %s", filename)
        except Exception as e:
            logger.warning("[ExperimentLogger] log_figure failed: %s", e)

    @property
    def run_id(self) -> str | None:
        """Active MLflow run ID, or None if not started."""
        if self._run is not None:
            return self._run.info.run_id
        return None

    @staticmethod
    def load_run_metrics(run_id: str, tracking_uri: str = "mlruns") -> dict[str, list[float]]:
        """
        Load all logged metrics for a completed run.

        Useful in notebooks for post-training analysis and comparison.

        Parameters
        ----------
        run_id:
            MLflow run ID.
        tracking_uri:
            Path to the local mlruns directory.

        Returns
        -------
        dict[str, list[float]]
            ``{metric_name: [values over epochs]}``.
        """
        if not _mlflow_available:
            return {}
        try:
            mlflow.set_tracking_uri(tracking_uri)
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(run_id)
            metrics: dict[str, list[float]] = {}
            for key in run.data.metrics:
                history = client.get_metric_history(run_id, key)
                metrics[key] = [m.value for m in sorted(history, key=lambda x: x.step)]
            return metrics
        except Exception as e:
            logger.warning("[ExperimentLogger] load_run_metrics failed: %s", e)
            return {}
