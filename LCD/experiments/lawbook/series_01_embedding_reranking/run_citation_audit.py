"""Create a CPU-only audit of rule-based legal citation extraction."""

from __future__ import annotations

import argparse
from pathlib import Path
from LCD.shared.common import REPOSITORY_ROOT, DEFAULT_RAG_DIR

from LCD.shared.rag import run_citation_audit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codex", type=Path, default=DEFAULT_RAG_DIR / "codex.csv")
    parser.add_argument("--rag-tests", type=Path, default=REPOSITORY_ROOT / "local/data/benchmarks/rag_tests")
    parser.add_argument("--dialogue-workbook", type=Path)
    parser.add_argument("--full-workbook", type=Path)
    parser.add_argument("--full-additional-workbook", type=Path)
    parser.add_argument("--test-docx", type=Path, default=REPOSITORY_ROOT / "local/data/benchmarks/test_docx")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPOSITORY_ROOT / "local/experiments/lawbook/series_01_embedding_reranking/results/citation_audit",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--routing-scope",
        choices=("all", "rules", "faiss"),
        default="all",
        help="Audit all hypotheses or only those routed to rules/FAISS",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    output = run_citation_audit(
        codex_path=arguments.codex,
        rag_test_dir=arguments.rag_tests,
        dialogue_workbook=arguments.dialogue_workbook,
        full_workbook=arguments.full_workbook,
        full_additional_workbook=arguments.full_additional_workbook,
        test_docx_dir=arguments.test_docx,
        results_dir=arguments.results_dir,
        output_path=arguments.output,
        routing_scope=arguments.routing_scope,
    )
    print(output)
