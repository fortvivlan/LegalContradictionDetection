"""Run or resume the rank stage of the local LoRA search."""

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


def run_lora_rank_experiments(
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
    """Compare ranks 8, 16, and 32 with alpha fixed to twice rank."""
    return run_sweep_stage(
        "rank",
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
    args = _parse_args()
    run_lora_rank_experiments(
        search_id=args.search_id,
        repo_root=args.repo_root,
        models=args.models,
        tasks=args.tasks,
        nruns=args.nruns,
        max_retries=args.max_retries,
        dry_run=args.dry_run,
    )
