"""Versioned RAG bundle manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from LCD.shared.common import REPOSITORY_ROOT, DEFAULT_RAG_DIR
from typing import Any

from ..common import file_sha256
from ..common import DEFAULT_EMBEDDING_REVISION


@dataclass(frozen=True)
class RerankerBundle:
    """Resolved optional reranker stored in a RAG manifest."""

    mode: str
    model: str
    revision: str | None
    trust_remote_code: bool
    max_length: int


@dataclass(frozen=True)
class RagBundle:
    """Resolved encoder, corpus, and FAISS paths for one RAG version."""

    name: str
    codex_path: Path
    index_dir: Path
    embedding_model: str
    embedding_revision: str | None
    normalize_embeddings: bool
    manifest_path: Path | None = None
    reranker: RerankerBundle | None = None
    embedding_query_prefix: str = ""
    embedding_document_prefix: str = ""
    embedding_trust_remote_code: bool = False
    embedding_precision: str = "float32"
    embedding_batch_size: int = 32


def write_rag_manifest(
    bundle_dir: str | Path,
    *,
    experiment_id: str,
    codex_path: str | Path,
    embedding_model_dir: str | Path,
    index_dir: str | Path,
    metadata: dict[str, Any],
    reranker: dict[str, Any] | None = None,
    embedding_options: dict[str, Any] | None = None,
) -> Path:
    """Write a versioned manifest for a trained encoder/index pair."""
    root = Path(bundle_dir).resolve()
    codex = Path(codex_path).resolve()
    model = Path(embedding_model_dir).resolve()
    index = Path(index_dir).resolve()
    try:
        stored_codex_path = str(codex.relative_to(root))
    except ValueError:
        stored_codex_path = str(codex)
    manifest = {
        "schema_version": 1,
        "name": experiment_id,
        "codex_path": stored_codex_path,
        "codex_sha256": file_sha256(codex),
        "embedding_model": str(model.relative_to(root)),
        "faiss_index": str(index.relative_to(root)),
        "index_sha256": {
            name: file_sha256(index / name) for name in ("index.faiss", "index.pkl")
        },
        "normalize_embeddings": True,
        "embedding_options": embedding_options or {},
        "reranker": reranker,
        **metadata,
    }
    target = root / "rag_manifest.json"
    target.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return target


def load_rag_bundle(
    source: str | Path,
    *,
    name: str | None = None,
    codex_override: str | Path | None = None,
) -> RagBundle:
    """Resolve either a legacy dms-rag directory or a RAG manifest."""
    path = Path(source).expanduser().resolve()
    if path.is_dir() and (path / "rag_manifest.json").is_file():
        path = path / "rag_manifest.json"
    if path.is_dir():
        codex = path / "codex.csv"
        index = path / "faiss_index"
        for required in (codex, index / "index.faiss", index / "index.pkl"):
            if not required.is_file():
                raise FileNotFoundError(f"Incomplete legacy RAG directory: {required}")
        return RagBundle(
            name=name or "baseline",
            codex_path=codex,
            index_dir=index,
            embedding_model="ai-forever/sbert_large_nlu_ru",
            embedding_revision=DEFAULT_EMBEDDING_REVISION,
            normalize_embeddings=False,
        )
    if not path.is_file():
        raise FileNotFoundError(f"RAG source does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError(f"Unsupported RAG manifest schema: {path}")
    root = path.parent
    if codex_override is not None:
        codex = Path(codex_override).expanduser().resolve()
    else:
        stored_codex = Path(value["codex_path"])
        codex = stored_codex if stored_codex.is_absolute() else root / stored_codex
    model = root / value["embedding_model"]
    index = root / value["faiss_index"]
    if file_sha256(codex) != value["codex_sha256"]:
        raise ValueError(f"Corpus hash does not match RAG manifest: {codex}")
    for filename, expected in value["index_sha256"].items():
        if file_sha256(index / filename) != expected:
            raise ValueError(f"Index hash does not match RAG manifest: {index / filename}")
    if not model.is_dir():
        raise FileNotFoundError(f"Embedding model directory is missing: {model}")
    reranker_value = value.get("reranker")
    reranker = None
    if reranker_value:
        reranker_model = str(reranker_value["model"])
        if bool(reranker_value.get("local", False)):
            reranker_path = root / reranker_model
            if not reranker_path.is_dir():
                raise FileNotFoundError(
                    f"Reranker model directory is missing: {reranker_path}"
                )
            reranker_model = str(reranker_path)
        reranker = RerankerBundle(
            mode=str(reranker_value["mode"]),
            model=reranker_model,
            revision=reranker_value.get("revision"),
            trust_remote_code=bool(reranker_value.get("trust_remote_code", False)),
            max_length=int(reranker_value.get("max_length", 1024)),
        )
    embedding_options = value.get("embedding_options") or {}
    return RagBundle(
        name=name or str(value["name"]),
        codex_path=codex,
        index_dir=index,
        embedding_model=str(model),
        embedding_revision=None,
        normalize_embeddings=bool(value.get("normalize_embeddings", True)),
        manifest_path=path,
        reranker=reranker,
        embedding_query_prefix=str(embedding_options.get("query_prefix", "")),
        embedding_document_prefix=str(
            embedding_options.get("document_prefix", "")
        ),
        embedding_trust_remote_code=bool(
            embedding_options.get("trust_remote_code", False)
        ),
        embedding_precision=str(embedding_options.get("precision", "float32")),
        embedding_batch_size=int(embedding_options.get("batch_size", 32)),
    )
