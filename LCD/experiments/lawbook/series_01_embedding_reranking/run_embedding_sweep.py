"""Run the stage-three pretrained embedding-model RAG comparison locally."""

from __future__ import annotations

import argparse
from pathlib import Path
from LCD.shared.common import REPOSITORY_ROOT, DEFAULT_RAG_DIR

from LCD.shared.rag.embedding_sweep import run_rag_embedding_sweep
from LCD.shared.rag.embeddings import (
    DEFAULT_STAGE_THREE_MODELS,
    EmbeddingModelSpec,
)


def _parse_model(value: str) -> EmbeddingModelSpec:
    known = {
        spec.alias.casefold(): spec for spec in DEFAULT_STAGE_THREE_MODELS
    } | {spec.model_id.casefold(): spec for spec in DEFAULT_STAGE_THREE_MODELS}
    if value.casefold() in known:
        return known[value.casefold()]
    try:
        alias, model_id = value.split("=", maxsplit=1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "model must be a default alias/ID or use ALIAS=MODEL_ID"
        ) from error
    if not alias.strip() or not model_id.strip():
        raise argparse.ArgumentTypeError("model alias and ID cannot be blank")
    return EmbeddingModelSpec(alias.strip(), model_id.strip())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        type=_parse_model,
        help="Repeatable default alias/model ID or ALIAS=MODEL_ID",
    )
    parser.add_argument(
        "--winner-manifest",
        type=Path,
        default=REPOSITORY_ROOT / "local/experiments/lawbook/series_01_embedding_reranking/artifacts/sbert_legal_v1/rag_manifest.json",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=REPOSITORY_ROOT / "local/experiments/lawbook/series_01_embedding_reranking/artifacts/embedding_stage3",
    )
    parser.add_argument("--rag-dir", type=Path, default=DEFAULT_RAG_DIR)
    parser.add_argument("--rag-tests", type=Path, default=REPOSITORY_ROOT / "local/data/benchmarks/rag_tests")
    parser.add_argument("--dialogue-workbook", type=Path)
    parser.add_argument("--full-workbook", type=Path)
    parser.add_argument("--full-additional-workbook", type=Path)
    parser.add_argument("--test-docx", type=Path, default=REPOSITORY_ROOT / "local/data/benchmarks/test_docx")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPOSITORY_ROOT / "local/experiments/lawbook/series_01_embedding_reranking/results/embedding_stage3",
    )
    parser.add_argument("--candidate-top-k", type=int, default=100)
    parser.add_argument("--final-top-k", type=int, default=60)
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument(
        "--embedding-precision",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--reranker-device", default="cuda")
    parser.add_argument(
        "--reranker-precision",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    report = run_rag_embedding_sweep(
        embedding_models=arguments.model or DEFAULT_STAGE_THREE_MODELS,
        winner_manifest=arguments.winner_manifest,
        artifact_root=arguments.artifact_root,
        rag_dir=arguments.rag_dir,
        rag_test_dir=arguments.rag_tests,
        dialogue_workbook=arguments.dialogue_workbook,
        full_workbook=arguments.full_workbook,
        full_additional_workbook=arguments.full_additional_workbook,
        test_docx_dir=arguments.test_docx,
        results_dir=arguments.results_dir,
        candidate_top_k=arguments.candidate_top_k,
        final_top_k=arguments.final_top_k,
        embedding_device=arguments.embedding_device,
        embedding_precision=arguments.embedding_precision,
        embedding_batch_size=arguments.embedding_batch_size,
        chunk_size=arguments.chunk_size,
        chunk_overlap=arguments.chunk_overlap,
        reranker_device=arguments.reranker_device,
        reranker_precision=arguments.reranker_precision,
        reranker_batch_size=arguments.reranker_batch_size,
        seed=arguments.seed,
    )
    print(report)
