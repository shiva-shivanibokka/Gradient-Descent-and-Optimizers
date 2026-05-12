"""
__main__.py
===========
CLI entrypoint for running training experiments.

Usage
-----
    python -m gdo.train --config configs/adam_mnist.yaml
    python -m gdo.train --config configs/adam_mnist.yaml --run-name my-run
    python -m gdo.train --config configs/adam_mnist.yaml --no-mlflow
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stdout,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="gdo",
        description="Gradient Descent and Optimizers — run a training experiment",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to a YAML experiment config file (e.g. configs/adam_mnist.yaml)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        type=str,
        help="Override the MLflow run name",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging for this run",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args()

    _configure_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Import here so logging is configured first
    from gdo.config import ExperimentConfig
    from gdo.experiment.logger import ExperimentLogger
    from gdo.training.trainer import Trainer

    logger.info("Loading config from %s", args.config)
    cfg = ExperimentConfig.from_yaml(args.config)

    if args.run_name:
        cfg.mlflow.run_name = args.run_name
    if args.no_mlflow:
        cfg.mlflow.enabled = False

    logger.info(
        "Starting experiment | optimizer=%s | dataset=%s | epochs=%d",
        cfg.optimizer.name.value,
        cfg.train.dataset.value,
        cfg.train.epochs,
    )

    trainer = Trainer.from_config(cfg)

    if cfg.mlflow.enabled:
        with ExperimentLogger(cfg.mlflow) as exp_logger:
            exp_logger.log_params(cfg.to_flat_dict())
            result = trainer.fit()
            exp_logger.log_run(result, cfg)
        logger.info("MLflow run ID: %s", exp_logger.run_id)
    else:
        result = trainer.fit()

    logger.info(
        "Done | best_val_loss=%.4f | best_val_acc=%.4f | best_epoch=%d",
        result.best_val_loss,
        result.best_val_acc,
        result.best_epoch,
    )


if __name__ == "__main__":
    main()
