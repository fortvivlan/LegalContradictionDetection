"""Train and evaluate the legal sentence-embedding RAG experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
from LCD.shared.common import REPOSITORY_ROOT, DEFAULT_RAG_DIR

from LCD.shared.rag.evaluation import DEFAULT_RETRIEVAL_DEPTHS
from LCD.shared.rag.training import run_rag_experiment


def _parse_retrieval_depth(value: str) -> tuple[int, int]:
    try:
        candidate, final = (int(part) for part in value.split(":", maxsplit=1))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "retrieval depth must use CANDIDATE:FINAL, for example 20:10"
        ) from error
    if candidate <= 0 or final <= 0 or final > candidate:
        raise argparse.ArgumentTypeError(
            "retrieval depth must be positive with FINAL <= CANDIDATE"
        )
    return candidate, final


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", default="sbert_legal_v1")
    parser.add_argument("--train", type=Path, default=REPOSITORY_ROOT / "local/data/classification/train.xlsx")
    parser.add_argument("--val", type=Path, default=REPOSITORY_ROOT / "local/data/classification/val.xlsx")
    parser.add_argument("--rag-dir", type=Path, default=DEFAULT_RAG_DIR)
    parser.add_argument("--rag-tests", type=Path, default=REPOSITORY_ROOT / "local/data/benchmarks/rag_tests")
    parser.add_argument("--dialogue-workbook", type=Path)
    parser.add_argument("--full-workbook", type=Path)
    parser.add_argument("--full-additional-workbook", type=Path)
    parser.add_argument("--test-docx", type=Path, default=REPOSITORY_ROOT / "local/data/benchmarks/test_docx")
    parser.add_argument("--model", default="ai-forever/sbert_large_nlu_ru")
    parser.add_argument(
        "--retrieval-depth",
        type=_parse_retrieval_depth,
        action="append",
        help="Repeatable CANDIDATE:FINAL pair; defaults to 20:10 and 40:20",
    )
    parser.add_argument(
        "--reranker-mode", choices=("pretrained", "finetuned"), default="pretrained"
    )
    parser.add_argument(
        "--reranker-model",
        default="Alibaba-NLP/gte-multilingual-reranker-base",
    )
    parser.add_argument("--reranker-revision")
    parser.add_argument("--reranker-trust-remote-code", action="store_true")
    parser.add_argument("--epochs", type=float, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--index-device", default="cpu")
    parser.add_argument("--reranker-device", default="cuda")
    parser.add_argument("--reranker-batch-size", type=int, default=8)
    parser.add_argument("--reranker-eval-batch-size", type=int, default=16)
    parser.add_argument("--reranker-gradient-accumulation", type=int, default=4)
    parser.add_argument("--reranker-max-length", type=int, default=1024)
    parser.add_argument("--reranker-epochs", type=float, default=3)
    parser.add_argument("--reranker-learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--reranker-precision",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument(
        "--no-reranker-gradient-checkpointing",
        action="store_true",
        help="Disable reranker gradient checkpointing to favor speed over memory",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    run_rag_experiment(
        experiment_id=arguments.experiment_id,
        train_path=arguments.train,
        val_path=arguments.val,
        rag_dir=arguments.rag_dir,
        rag_test_dir=arguments.rag_tests,
        dialogue_workbook=arguments.dialogue_workbook,
        full_workbook=arguments.full_workbook,
        full_additional_workbook=arguments.full_additional_workbook,
        test_docx_dir=arguments.test_docx,
        model_id=arguments.model,
        retrieval_depths=(
            arguments.retrieval_depth or DEFAULT_RETRIEVAL_DEPTHS
        ),
        reranker_mode=arguments.reranker_mode,
        reranker_model_id=arguments.reranker_model,
        reranker_revision=arguments.reranker_revision,
        reranker_trust_remote_code=(
            True if arguments.reranker_trust_remote_code else None
        ),
        hyperparameters={
            "epochs": arguments.epochs,
            "batch_size": arguments.batch_size,
            "gradient_accumulation_steps": arguments.gradient_accumulation,
            "learning_rate": arguments.learning_rate,
            "embedding_device": arguments.embedding_device,
            "index_device": arguments.index_device,
            "reranker_device": arguments.reranker_device,
            "reranker_batch_size": arguments.reranker_batch_size,
            "reranker_eval_batch_size": arguments.reranker_eval_batch_size,
            "reranker_gradient_accumulation_steps": (
                arguments.reranker_gradient_accumulation
            ),
            "reranker_max_length": arguments.reranker_max_length,
            "reranker_epochs": arguments.reranker_epochs,
            "reranker_learning_rate": arguments.reranker_learning_rate,
            "reranker_precision": arguments.reranker_precision,
            "reranker_gradient_checkpointing": (
                not arguments.no_reranker_gradient_checkpointing
            ),
        },
    )
