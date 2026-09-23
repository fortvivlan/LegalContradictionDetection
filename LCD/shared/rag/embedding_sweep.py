"""Stage-three comparison of pretrained embedding models at the winning depth."""

from __future__ import annotations

import gc
import json
import re
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from LCD.shared.common import REPOSITORY_ROOT, DEFAULT_RAG_DIR
from typing import Any

from ..local_support import get_huggingface_token
from ..common import (
    announce_stage,
    configure_reproducibility,
    file_sha256,
    resolve_huggingface_revision,
    reproducibility_metadata,
)
from ..retrieval import ensure_rag_repository
from .artifacts import RagBundle, load_rag_bundle, write_rag_manifest
from .embeddings import (
    DEFAULT_STAGE_THREE_MODELS,
    EmbeddingModelSpec,
    SentenceTransformerEmbeddings,
)
from .evaluation import run_rag_evaluation
from .indexing import build_faiss_index
from .reranking import CrossEncoderReranker

BASELINE_VARIANT = "baseline_embeddings__finetuned_reranker"
CANDIDATE_VARIANT = "tuned_embeddings__finetuned_reranker"


def _normalize_models(
    values: Sequence[EmbeddingModelSpec],
) -> tuple[EmbeddingModelSpec, ...]:
    models = tuple(values)
    if not models:
        raise ValueError("embedding_models cannot be empty")
    aliases = []
    for spec in models:
        alias = spec.alias.strip()
        if not alias or not spec.model_id.strip():
            raise ValueError("embedding aliases and model IDs cannot be blank")
        if alias != spec.alias or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]*", alias
        ):
            raise ValueError(
                "embedding aliases must be trimmed directory-safe names"
            )
        aliases.append(alias)
    if len(set(aliases)) != len(aliases):
        raise ValueError("embedding aliases must be unique")
    return models


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_winner_manifest(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if value.get("schema_version") != 1:
        raise ValueError(f"Unsupported winner manifest schema: {path}")
    reranker = value.get("reranker") or {}
    if reranker.get("mode") != "finetuned" or not reranker.get("local"):
        raise ValueError("winner_manifest must contain a local fine-tuned reranker")
    if not value.get("codex_sha256") or not value.get("rag_commit"):
        raise ValueError("winner_manifest must record codex_sha256 and rag_commit")
    return value


def _candidate_signature(
    spec: EmbeddingModelSpec,
    *,
    resolved_revision: str,
    codex_sha256: str,
    rag_commit: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_precision: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "model": asdict(spec),
        "resolved_revision": resolved_revision,
        "codex_sha256": codex_sha256,
        "rag_commit": rag_commit,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_precision": embedding_precision,
        "seed": seed,
        "normalize_embeddings": True,
    }


def _completed_candidate(
    candidate_dir: Path,
    expected_signature: dict[str, Any],
    *,
    codex_path: Path,
) -> RagBundle | None:
    manifest = candidate_dir / "rag_manifest.json"
    config = candidate_dir / "run_config.json"
    if not manifest.is_file() and not config.is_file():
        return None
    if not manifest.is_file() or not config.is_file():
        raise ValueError(f"Incomplete cached embedding artifact: {candidate_dir}")
    stored = _read_json(config)
    if stored.get("signature") != expected_signature:
        raise ValueError(
            f"Cached embedding artifact does not match the request: {candidate_dir}"
        )
    return load_rag_bundle(manifest, codex_override=codex_path)


def _cached_revision(candidate_dir: Path, spec: EmbeddingModelSpec) -> str | None:
    """Reuse the immutable revision of a matching completed or partial run."""
    config = candidate_dir / "run_config.json"
    if not config.is_file():
        return None
    signature = _read_json(config).get("signature") or {}
    if signature.get("model") != asdict(spec):
        return None
    revision = signature.get("resolved_revision")
    return str(revision) if revision else None


def _release_embedding_model(value: Any | None = None) -> None:
    if value is not None:
        del value
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ModuleNotFoundError:
        pass


def _build_candidate(
    spec: EmbeddingModelSpec,
    *,
    resolved_revision: str,
    signature: dict[str, Any],
    candidate_dir: Path,
    codex_path: Path,
    embedding_device: str,
    embedding_precision: str,
    embedding_batch_size: int,
    chunk_size: int,
    chunk_overlap: int,
    token: str | None,
) -> RagBundle:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    model_dir = candidate_dir / "embedding_model"
    embeddings = SentenceTransformerEmbeddings(
        spec.model_id,
        revision=resolved_revision,
        token=token,
        trust_remote_code=spec.trust_remote_code,
        device=embedding_device,
        precision=embedding_precision,
        batch_size=embedding_batch_size,
        normalize_embeddings=True,
        query_prefix=spec.query_prefix,
        document_prefix=spec.document_prefix,
        show_progress=True,
    )
    embeddings.save_pretrained(model_dir)
    index_metadata = build_faiss_index(
        codex_path,
        model_dir,
        candidate_dir / "faiss_index",
        device=embedding_device,
        batch_size=embedding_batch_size,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embeddings=embeddings,
    )
    resolved_dtype = embeddings.resolved_dtype
    del embeddings
    _release_embedding_model()

    run_config = {
        "workflow": "pretrained_embedding_model_sweep",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "signature": signature,
        "embedding_precision": embedding_precision,
        "resolved_dtype": resolved_dtype,
        "embedding_batch_size": embedding_batch_size,
        "index": index_metadata,
        "reproducibility": reproducibility_metadata(
            int(signature["seed"]), deterministic=True
        ),
    }
    (candidate_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    manifest = write_rag_manifest(
        candidate_dir,
        experiment_id=spec.alias,
        codex_path=codex_path,
        embedding_model_dir=model_dir,
        index_dir=candidate_dir / "faiss_index",
        metadata={
            "model_id": spec.model_id,
            "resolved_revision": resolved_revision,
            "rag_commit": signature["rag_commit"],
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "stage": 3,
        },
        embedding_options={
            "query_prefix": spec.query_prefix,
            "document_prefix": spec.document_prefix,
            "trust_remote_code": spec.trust_remote_code,
            "precision": resolved_dtype,
            "batch_size": embedding_batch_size,
        },
    )
    return load_rag_bundle(manifest, codex_override=codex_path)


def run_rag_embedding_sweep(
    *,
    embedding_models: Sequence[EmbeddingModelSpec] = DEFAULT_STAGE_THREE_MODELS,
    winner_manifest: str | Path = (
        REPOSITORY_ROOT / "local/experiments/lawbook/series_01_embedding_reranking/artifacts/sbert_legal_v1/rag_manifest.json"
    ),
    artifact_root: str | Path = REPOSITORY_ROOT / "local/experiments/lawbook/series_01_embedding_reranking/artifacts/embedding_stage3",
    rag_dir: str | Path = DEFAULT_RAG_DIR,
    rag_test_dir: str | Path = REPOSITORY_ROOT / "local/data/benchmarks/rag_tests",
    dialogue_workbook: str | Path | None = None,
    full_workbook: str | Path | None = None,
    full_additional_workbook: str | Path | None = None,
    test_docx_dir: str | Path = REPOSITORY_ROOT / "local/data/benchmarks/test_docx",
    results_dir: str | Path = REPOSITORY_ROOT / "local/experiments/lawbook/series_01_embedding_reranking/results/embedding_stage3",
    candidate_top_k: int = 100,
    final_top_k: int = 60,
    embedding_device: str = "cuda",
    embedding_precision: str = "auto",
    embedding_batch_size: int = 16,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    reranker_device: str = "cuda",
    reranker_precision: str = "auto",
    reranker_batch_size: int = 16,
    seed: int = 42,
) -> Path:
    """Build and evaluate pretrained embedding indexes with the winner reranker."""
    import pandas as pd

    models = _normalize_models(embedding_models)
    configure_reproducibility(seed, deterministic=True)
    if candidate_top_k <= 0 or final_top_k <= 0:
        raise ValueError("retrieval depths must be positive")
    if final_top_k > candidate_top_k:
        raise ValueError("final_top_k cannot exceed candidate_top_k")
    if embedding_batch_size <= 0 or reranker_batch_size <= 0:
        raise ValueError("embedding and reranker batch sizes must be positive")
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("invalid chunk size or overlap")

    winner_path = Path(winner_manifest).expanduser().resolve()
    winner_value = _validate_winner_manifest(winner_path)
    rag_path, rag_commit = ensure_rag_repository(
        rag_dir, revision=str(winner_value["rag_commit"])
    )
    if rag_commit != str(winner_value["rag_commit"]):
        raise RuntimeError("Resolved RAG commit differs from the winner manifest")
    codex_path = rag_path / "codex.csv"
    if file_sha256(codex_path) != str(winner_value["codex_sha256"]):
        raise ValueError("Current codex.csv differs from the winner manifest")

    baseline_bundle = load_rag_bundle(rag_path, name="sbert_baseline")
    winner_bundle = load_rag_bundle(winner_path, codex_override=codex_path)
    if winner_bundle.reranker is None:
        raise ValueError("Winner bundle has no reranker")

    token = get_huggingface_token()
    artifacts = Path(artifact_root)
    candidate_bundles: list[tuple[EmbeddingModelSpec, str, RagBundle]] = []
    for spec in models:
        candidate_dir = artifacts / spec.alias
        resolved_revision = _cached_revision(candidate_dir, spec) or (
            resolve_huggingface_revision(
                spec.model_id, spec.revision, token=token
            )
        )
        signature = _candidate_signature(
            spec,
            resolved_revision=resolved_revision,
            codex_sha256=str(winner_value["codex_sha256"]),
            rag_commit=rag_commit,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_precision=embedding_precision,
            seed=seed,
        )
        bundle = _completed_candidate(
            candidate_dir, signature, codex_path=codex_path
        )
        if bundle is None:
            announce_stage(
                "rag-embedding-sweep",
                spec.alias,
                f"building {spec.model_id} at {resolved_revision}",
            )
            bundle = _build_candidate(
                spec,
                resolved_revision=resolved_revision,
                signature=signature,
                candidate_dir=candidate_dir,
                codex_path=codex_path,
                embedding_device=embedding_device,
                embedding_precision=embedding_precision,
                embedding_batch_size=embedding_batch_size,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                token=token,
            )
        else:
            announce_stage(
                "rag-embedding-sweep",
                spec.alias,
                "reusing the matching checkpoint and FAISS index",
            )
        candidate_bundles.append((spec, resolved_revision, bundle))

    reranker_config = winner_bundle.reranker
    reranker = CrossEncoderReranker(
        reranker_config.model,
        revision=None,
        trust_remote_code=reranker_config.trust_remote_code,
        device=reranker_device,
        precision=reranker_precision,
        batch_size=reranker_batch_size,
        max_length=reranker_config.max_length,
    )
    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    missing = None
    baseline_score = None
    for index, (spec, resolved_revision, bundle) in enumerate(candidate_bundles):
        announce_stage(
            "rag-embedding-sweep",
            spec.alias,
            f"evaluating candidate/final depth {candidate_top_k}:{final_top_k}",
        )
        variants = (
            (BASELINE_VARIANT, CANDIDATE_VARIANT)
            if index == 0
            else (CANDIDATE_VARIANT,)
        )
        model_output = output / "by_model" / spec.alias
        scores = run_rag_evaluation(
            [baseline_bundle, bundle],
            rag_test_dir=rag_test_dir,
            dialogue_workbook=dialogue_workbook,
            full_workbook=full_workbook,
            full_additional_workbook=full_additional_workbook,
            test_docx_dir=test_docx_dir,
            results_dir=model_output,
            embedding_device=embedding_device,
            retrieval_depths=((candidate_top_k, final_top_k),),
            finetuned_reranker=reranker,
            evaluation_variants=variants,
        )
        for row in scores.itertuples(index=False):
            baseline = row.variant == BASELINE_VARIANT
            record = row._asdict()
            record.update(
                {
                    "model_alias": "sbert_baseline" if baseline else spec.alias,
                    "model_id": (
                        str(winner_value["model_id"])
                        if baseline
                        else spec.model_id
                    ),
                    "resolved_revision": (
                        str(winner_value["resolved_revision"])
                        if baseline
                        else resolved_revision
                    ),
                }
            )
            frames.append(record)
            if baseline:
                baseline_score = row.full_total_recall
        if missing is None:
            missing = pd.read_excel(
                model_output / "rag_recall.xlsx",
                sheet_name="missing_hypotheses",
            )
        _release_embedding_model()

    combined = pd.DataFrame(frames)
    leading = ["model_alias", "model_id", "resolved_revision"]
    combined = combined[leading + [c for c in combined.columns if c not in leading]]
    combined["full_total_recall_rank"] = combined["full_total_recall"].rank(
        method="dense", ascending=False
    ).astype("Int64")
    combined["full_total_recall_delta_vs_sbert"] = (
        combined["full_total_recall"] - baseline_score
        if baseline_score is not None
        else None
    )
    best = combined["full_total_recall"].max()
    combined["is_best_full"] = combined["full_total_recall"].eq(best)
    combined.to_csv(output / "rag_recall.csv", index=False, encoding="utf-8")
    with pd.ExcelWriter(output / "rag_recall.xlsx", engine="openpyxl") as writer:
        combined.to_excel(writer, sheet_name="recall", index=False)
        (missing if missing is not None else pd.DataFrame()).to_excel(
            writer, sheet_name="missing_hypotheses", index=False
        )

    evaluation_config = {
        "workflow": "pretrained_embedding_model_sweep",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "winner_manifest": {
            "path": str(winner_path),
            "sha256": file_sha256(winner_path),
        },
        "rag_commit": rag_commit,
        "codex_sha256": file_sha256(codex_path),
        "retrieval_depth": {
            "candidate_top_k": candidate_top_k,
            "final_top_k": final_top_k,
        },
        "models": [
            {
                **asdict(spec),
                "resolved_revision": revision,
                "manifest": str(bundle.manifest_path),
                "manifest_sha256": file_sha256(bundle.manifest_path),
            }
            for spec, revision, bundle in candidate_bundles
            if bundle.manifest_path is not None
        ],
        "embedding": {
            "device": embedding_device,
            "precision": embedding_precision,
            "batch_size": embedding_batch_size,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "normalize_embeddings": True,
        },
        "reranker": {
            "model": reranker_config.model,
            "base_model_id": (winner_value.get("reranker") or {}).get(
                "base_model_id"
            ),
            "base_model_revision": (winner_value.get("reranker") or {}).get(
                "base_model_revision"
            ),
            "device": reranker_device,
            "precision": reranker_precision,
            "batch_size": reranker_batch_size,
        },
        "reproducibility": reproducibility_metadata(seed, deterministic=True),
    }
    (output / "evaluation_config.json").write_text(
        json.dumps(evaluation_config, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return (output / "rag_recall.xlsx").resolve()
