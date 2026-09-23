"""Focused multi-artifact top-k sweep for baseline retrieval and local rerankers."""

from __future__ import annotations

import gc
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from LCD.shared.common import REPOSITORY_ROOT, DEFAULT_RAG_DIR
from typing import Any

from ..common import file_sha256
from ..retrieval import ensure_rag_repository
from .artifacts import load_rag_bundle
from .evaluation import run_rag_evaluation
from .reranking import CrossEncoderReranker

FOCUSED_VARIANT = "baseline_embeddings__finetuned_reranker"


@dataclass(frozen=True)
class RagDepthRun:
    """One artifact and candidate/final retrieval depth assignment."""

    artifact_run: str
    candidate_top_k: int
    final_top_k: int


DEFAULT_STAGE_TWO_DEPTH_RUNS: tuple[RagDepthRun, ...] = (
    RagDepthRun("sbert_legal_v1", 100, 80),
    RagDepthRun("sbert_legal_60", 80, 60),
    RagDepthRun("sbert_legal_40", 60, 40),
    RagDepthRun("sbert_legal_v1", 40, 20),
    RagDepthRun("sbert_legal_v1", 20, 10),
    RagDepthRun("sbert_legal_v1", 100, 60),
    RagDepthRun("sbert_legal_v1", 100, 40),
    RagDepthRun("sbert_legal_v1", 100, 20),
    RagDepthRun("sbert_legal_v1", 100, 10),
)


def _normalize_depth_runs(
    values: Sequence[RagDepthRun | tuple[str, int, int]],
) -> tuple[RagDepthRun, ...]:
    runs = tuple(
        value if isinstance(value, RagDepthRun) else RagDepthRun(*value)
        for value in values
    )
    if not runs:
        raise ValueError("depth_runs cannot be empty")
    depth_pairs = []
    for run in runs:
        if not run.artifact_run.strip():
            raise ValueError("artifact_run cannot be blank")
        if run.candidate_top_k <= 0 or run.final_top_k <= 0:
            raise ValueError("retrieval depths must be positive")
        if run.final_top_k > run.candidate_top_k:
            raise ValueError("final_top_k cannot exceed candidate_top_k")
        depth_pairs.append((run.candidate_top_k, run.final_top_k))
    if len(set(depth_pairs)) != len(depth_pairs):
        raise ValueError("depth_runs cannot repeat a candidate/final pair")
    return runs


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"RAG manifest is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError(f"Unsupported RAG manifest schema: {path}")
    reranker = value.get("reranker") or {}
    if reranker.get("mode") != "finetuned" or not reranker.get("local"):
        raise ValueError(f"{path} does not contain a local fine-tuned reranker")
    return value


def _compatibility_signature(value: dict[str, Any]) -> tuple[Any, ...]:
    reranker = value["reranker"]
    return (
        value.get("codex_sha256"),
        value.get("rag_commit"),
        value.get("model_id"),
        value.get("resolved_revision"),
        reranker.get("base_model_id"),
        reranker.get("base_model_revision"),
    )


def _validate_manifests(
    runs: Sequence[RagDepthRun], artifact_root: Path
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    paths = {
        run.artifact_run: artifact_root / run.artifact_run / "rag_manifest.json"
        for run in runs
    }
    values = {name: _read_manifest(path) for name, path in paths.items()}
    first_name = next(iter(values))
    expected = _compatibility_signature(values[first_name])
    for name, value in values.items():
        if str(value.get("name")) != name:
            raise ValueError(f"Manifest name does not match artifact run {name!r}")
        if _compatibility_signature(value) != expected:
            raise ValueError(
                f"Artifact {name!r} is incompatible with {first_name!r}"
            )
    if not expected[0] or not expected[1]:
        raise ValueError("Sweep manifests must record codex_sha256 and rag_commit")
    return paths, values


def _release_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ModuleNotFoundError:
        pass


def run_rag_depth_sweep(
    *,
    depth_runs: Sequence[
        RagDepthRun | tuple[str, int, int]
    ] = DEFAULT_STAGE_TWO_DEPTH_RUNS,
    artifact_root: str | Path = REPOSITORY_ROOT / "local/experiments/lawbook/series_01_embedding_reranking/artifacts",
    rag_dir: str | Path = DEFAULT_RAG_DIR,
    rag_test_dir: str | Path = REPOSITORY_ROOT / "local/data/benchmarks/rag_tests",
    dialogue_workbook: str | Path | None = None,
    full_workbook: str | Path | None = None,
    full_additional_workbook: str | Path | None = None,
    test_docx_dir: str | Path = REPOSITORY_ROOT / "local/data/benchmarks/test_docx",
    results_dir: str | Path = REPOSITORY_ROOT / "local/experiments/lawbook/series_01_embedding_reranking/results/sbert_legal_v1/top_k_stage2",
    embedding_device: str = "cuda",
    reranker_device: str = "cuda",
    reranker_precision: str = "auto",
    reranker_batch_size: int = 16,
) -> Path:
    """Run the ordered baseline-embedding/fine-tuned-reranker depth sweep."""
    import pandas as pd

    runs = _normalize_depth_runs(depth_runs)
    artifacts = Path(artifact_root)
    manifest_paths, manifest_values = _validate_manifests(runs, artifacts)
    first_manifest = manifest_values[runs[0].artifact_run]
    rag_path, resolved_rag_commit = ensure_rag_repository(
        rag_dir, revision=str(first_manifest["rag_commit"])
    )
    if resolved_rag_commit != str(first_manifest["rag_commit"]):
        raise RuntimeError("Resolved RAG commit differs from the sweep manifests")
    baseline_bundle = load_rag_bundle(rag_path, name="baseline")
    codex_path = rag_path / "codex.csv"

    grouped: dict[str, list[RagDepthRun]] = defaultdict(list)
    for run in runs:
        grouped[run.artifact_run].append(run)

    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    missing = None
    order = {
        (run.artifact_run, run.candidate_top_k, run.final_top_k): index
        for index, run in enumerate(runs)
    }
    for artifact_run, artifact_depths in grouped.items():
        bundle = load_rag_bundle(
            manifest_paths[artifact_run],
            codex_override=codex_path,
        )
        if bundle.reranker is None or bundle.reranker.mode != "finetuned":
            raise ValueError(f"Artifact {artifact_run!r} lacks a fine-tuned reranker")
        reranker = CrossEncoderReranker(
            bundle.reranker.model,
            revision=None,
            trust_remote_code=bundle.reranker.trust_remote_code,
            device=reranker_device,
            precision=reranker_precision,
            batch_size=reranker_batch_size,
            max_length=bundle.reranker.max_length,
        )
        group_output = output / "by_artifact" / artifact_run
        scores = run_rag_evaluation(
            [baseline_bundle, bundle],
            rag_test_dir=rag_test_dir,
            dialogue_workbook=dialogue_workbook,
            full_workbook=full_workbook,
            full_additional_workbook=full_additional_workbook,
            test_docx_dir=test_docx_dir,
            results_dir=group_output,
            embedding_device=embedding_device,
            retrieval_depths=tuple(
                (run.candidate_top_k, run.final_top_k)
                for run in artifact_depths
            ),
            finetuned_reranker=reranker,
            evaluation_variants=(FOCUSED_VARIANT,),
        )
        scores.insert(0, "artifact_run", artifact_run)
        frames.append(scores)
        if missing is None:
            missing = pd.read_excel(
                group_output / "rag_recall.xlsx",
                sheet_name="missing_hypotheses",
            )
        del reranker
        _release_gpu()

    combined = pd.concat(frames, ignore_index=True)
    combined["_requested_order"] = [
        order[(row.artifact_run, row.candidate_top_k, row.final_top_k)]
        for row in combined.itertuples(index=False)
    ]
    combined.sort_values("_requested_order", inplace=True)
    combined.drop(columns="_requested_order", inplace=True)
    combined["full_total_recall_rank"] = combined["full_total_recall"].rank(
        method="dense", ascending=False
    ).astype("Int64")
    best = combined["full_total_recall"].max()
    combined["is_best_full"] = combined["full_total_recall"].eq(best)

    combined.to_csv(output / "rag_recall.csv", index=False, encoding="utf-8")
    with pd.ExcelWriter(output / "rag_recall.xlsx", engine="openpyxl") as writer:
        combined.to_excel(writer, sheet_name="recall", index=False)
        (missing if missing is not None else pd.DataFrame()).to_excel(
            writer, sheet_name="missing_hypotheses", index=False
        )
    config = {
        "workflow": "baseline_embeddings_finetuned_reranker_depth_sweep",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "focused_variant": FOCUSED_VARIANT,
        "ranking": "full_total_recall_dense_descending",
        "rag_commit": resolved_rag_commit,
        "depth_runs": [asdict(run) for run in runs],
        "manifests": {
            name: {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
            }
            for name, path in manifest_paths.items()
        },
        "scoring": {
            "faiss_recall": "FAISS-routed hypotheses only",
            "rules_recall": "rule-routed hypotheses only",
            "total_recall": "all annotated hypotheses with production routing",
        },
    }
    (output / "evaluation_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return (output / "rag_recall.xlsx").resolve()

