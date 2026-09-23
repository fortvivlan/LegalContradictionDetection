"""Run the stage-two multi-artifact RAG top-k sweep."""

from __future__ import annotations

import argparse
from pathlib import Path
from LCD.shared.common import REPOSITORY_ROOT, DEFAULT_RAG_DIR

from LCD.shared.rag.depth_sweep import (
    DEFAULT_STAGE_TWO_DEPTH_RUNS,
    RagDepthRun,
    run_rag_depth_sweep,
)


def _parse_depth_run(value: str) -> RagDepthRun:
    try:
        artifact, candidate, final = value.rsplit(":", maxsplit=2)
        return RagDepthRun(artifact, int(candidate), int(final))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "depth run must use ARTIFACT:CANDIDATE:FINAL"
        ) from error


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--depth-run",
        action="append",
        type=_parse_depth_run,
        help="Repeatable ARTIFACT:CANDIDATE:FINAL; defaults to stage-two matrix",
    )
    parser.add_argument("--artifact-root", type=Path, default=REPOSITORY_ROOT / "local/experiments/lawbook/series_01_embedding_reranking/artifacts")
    parser.add_argument("--rag-dir", type=Path, default=DEFAULT_RAG_DIR)
    parser.add_argument("--rag-tests", type=Path, default=REPOSITORY_ROOT / "local/data/benchmarks/rag_tests")
    parser.add_argument("--dialogue-workbook", type=Path)
    parser.add_argument("--full-workbook", type=Path)
    parser.add_argument("--full-additional-workbook", type=Path)
    parser.add_argument("--test-docx", type=Path, default=REPOSITORY_ROOT / "local/data/benchmarks/test_docx")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPOSITORY_ROOT / "local/experiments/lawbook/series_01_embedding_reranking/results/sbert_legal_v1/top_k_stage2",
    )
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--reranker-device", default="cuda")
    parser.add_argument(
        "--reranker-precision",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    report = run_rag_depth_sweep(
        depth_runs=arguments.depth_run or DEFAULT_STAGE_TWO_DEPTH_RUNS,
        artifact_root=arguments.artifact_root,
        rag_dir=arguments.rag_dir,
        rag_test_dir=arguments.rag_tests,
        dialogue_workbook=arguments.dialogue_workbook,
        full_workbook=arguments.full_workbook,
        full_additional_workbook=arguments.full_additional_workbook,
        test_docx_dir=arguments.test_docx,
        results_dir=arguments.results_dir,
        embedding_device=arguments.embedding_device,
        reranker_device=arguments.reranker_device,
        reranker_precision=arguments.reranker_precision,
        reranker_batch_size=arguments.reranker_batch_size,
    )
    print(report)

