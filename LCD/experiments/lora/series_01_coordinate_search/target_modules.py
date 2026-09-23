"""Run or resume the target-module stage of the local LoRA search."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from LCD.experiments.lora.series_01_coordinate_search.engine import (
    ALL_MODELS,
    ALL_TASKS,
    DEFAULT_SEARCH_ID,
    DEFAULT_SWEEP_TASKS,
    run_sweep_stage,
)


def run_lora_target_module_experiments(
    *,
    search_id: str = DEFAULT_SEARCH_ID,
    repo_root: str | Path | None = None,
    models: Sequence[str] = ALL_MODELS,
    tasks: Sequence[str] = DEFAULT_SWEEP_TASKS,
    hyperparameters: Mapping[str, Any] | None = None,
    nruns: int = 4,
    max_retries: int = 1,
    dry_run: bool = False,
):
    """Compare q/v, q/k/v, and all-linear LoRA target modules."""
    return run_sweep_stage(
        "target_modules",
        search_id=search_id,
        repo_root=repo_root,
        models=models,
        tasks=tasks,
        hyperparameters=hyperparameters,
        nruns=nruns,
        max_retries=max_retries,
        dry_run=dry_run,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-id", default=DEFAULT_SEARCH_ID)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS)
    parser.add_argument(
        "--tasks", nargs="+", choices=ALL_TASKS, default=DEFAULT_SWEEP_TASKS
    )
    parser.add_argument("--nruns", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    run_lora_target_module_experiments(
        search_id=arguments.search_id,
        repo_root=arguments.repo_root,
        models=arguments.models,
        tasks=arguments.tasks,
        nruns=arguments.nruns,
        max_retries=arguments.max_retries,
        dry_run=arguments.dry_run,
    )
