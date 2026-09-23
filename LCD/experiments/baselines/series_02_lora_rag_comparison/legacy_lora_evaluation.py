"""Evaluate explicitly selected historical three-class adapters with a local RAG bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from LCD.shared.common import REPOSITORY_ROOT
from LCD.shared.full_pipeline import run_full_pipeline_evaluation


def run_legacy_lora_evaluation(
    *,
    models_source: str | Path,
    rag_source: str | Path = REPOSITORY_ROOT / "local" / "bundles" / "rag-qwen",
    results_dir: str | Path = (
        REPOSITORY_ROOT / "local" / "experiments" / "baselines"
        / "series_02_lora_rag_comparison" / "legacy_adapters" / "results"
    ),
    candidate_top_k: int = 100,
    final_top_k: int = 60,
):
    """Run the shared evaluator on a ternary-only model manifest or artifact tree."""
    return run_full_pipeline_evaluation(
        models_source=models_source,
        rag_source=rag_source,
        results_dir=results_dir,
        reranker_mode="bundle",
        inference_parameters={
            "candidate_top_k": candidate_top_k,
            "final_top_k": final_top_k,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-source", required=True, type=Path)
    parser.add_argument("--rag-source", type=Path, default=REPOSITORY_ROOT / "local" / "bundles" / "rag-qwen")
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--candidate-top-k", type=int, default=100)
    parser.add_argument("--final-top-k", type=int, default=60)
    kwargs = vars(parser.parse_args())
    if kwargs["results_dir"] is None:
        del kwargs["results_dir"]
    run_legacy_lora_evaluation(**kwargs)


if __name__ == "__main__":
    main()
