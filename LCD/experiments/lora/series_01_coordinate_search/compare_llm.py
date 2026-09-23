"""Compare final staged-search LoRA winners with matched ready LLMs."""

from __future__ import annotations

import argparse
from pathlib import Path

from LCD.experiments.lora.series_01_coordinate_search.engine import DEFAULT_SEARCH_ID, run_llm_comparison


def run_lora_vs_llm_comparison(
    *, search_id: str = DEFAULT_SEARCH_ID,
    repo_root: str | Path | None = None,
    nruns: int = 4,
    dry_run: bool = False,
):
    """Evaluate matched ready LLMs and report deltas from winning LoRAs."""
    return run_llm_comparison(
        search_id=search_id, repo_root=repo_root,
        nruns=nruns, dry_run=dry_run,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-id", default=DEFAULT_SEARCH_ID)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--nruns", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_lora_vs_llm_comparison(
        search_id=args.search_id, repo_root=args.repo_root,
        nruns=args.nruns, dry_run=args.dry_run,
    )
