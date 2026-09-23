import json
from pathlib import Path

from LCD.shared.rag.artifacts import load_rag_bundle, write_rag_manifest


def test_manifest_round_trip_resolves_local_reranker(tmp_path: Path) -> None:
    codex = tmp_path / "codex.csv"
    codex.write_text("text,source\ntext,source\n", encoding="utf-8")
    embedding = tmp_path / "embedding_model"
    embedding.mkdir()
    index = tmp_path / "faiss_index"
    index.mkdir()
    (index / "index.faiss").write_bytes(b"faiss")
    (index / "index.pkl").write_bytes(b"metadata")
    reranker = tmp_path / "reranker_model"
    reranker.mkdir()

    manifest = write_rag_manifest(
        tmp_path,
        experiment_id="legal-rag",
        codex_path=codex,
        embedding_model_dir=embedding,
        index_dir=index,
        metadata={},
        reranker={
            "mode": "finetuned",
            "model": "reranker_model",
            "local": True,
            "revision": None,
            "trust_remote_code": True,
            "max_length": 768,
        },
    )

    bundle = load_rag_bundle(manifest)

    assert bundle.reranker is not None
    assert bundle.reranker.mode == "finetuned"
    assert bundle.reranker.model == str(reranker)
    assert bundle.reranker.max_length == 768


def test_legacy_manifest_without_reranker_remains_loadable(tmp_path: Path) -> None:
    codex = tmp_path / "codex.csv"
    codex.write_text("text,source\ntext,source\n", encoding="utf-8")
    embedding = tmp_path / "embedding_model"
    embedding.mkdir()
    index = tmp_path / "faiss_index"
    index.mkdir()
    (index / "index.faiss").write_bytes(b"faiss")
    (index / "index.pkl").write_bytes(b"metadata")

    manifest = write_rag_manifest(
        tmp_path,
        experiment_id="legal-rag",
        codex_path=codex,
        embedding_model_dir=embedding,
        index_dir=index,
        metadata={},
    )

    assert load_rag_bundle(manifest).reranker is None


def test_manifest_accepts_hash_validated_codex_override(tmp_path: Path) -> None:
    original = tmp_path / "codex.csv"
    original.write_text("text,source\ntext,source\n", encoding="utf-8")
    override = tmp_path / "portable" / "codex.csv"
    override.parent.mkdir()
    override.write_text(original.read_text(encoding="utf-8"), encoding="utf-8")
    embedding = tmp_path / "embedding_model"
    embedding.mkdir()
    index = tmp_path / "faiss_index"
    index.mkdir()
    (index / "index.faiss").write_bytes(b"faiss")
    (index / "index.pkl").write_bytes(b"metadata")
    manifest = write_rag_manifest(
        tmp_path,
        experiment_id="portable",
        codex_path=original,
        embedding_model_dir=embedding,
        index_dir=index,
        metadata={},
    )

    bundle = load_rag_bundle(manifest, codex_override=override)

    assert bundle.codex_path == override.resolve()


def test_manifest_round_trip_preserves_asymmetric_embedding_options(
    tmp_path: Path,
) -> None:
    codex = tmp_path / "codex.csv"
    codex.write_text("text,source\ntext,source\n", encoding="utf-8")
    embedding = tmp_path / "embedding_model"
    embedding.mkdir()
    index = tmp_path / "faiss_index"
    index.mkdir()
    (index / "index.faiss").write_bytes(b"faiss")
    (index / "index.pkl").write_bytes(b"metadata")
    manifest = write_rag_manifest(
        tmp_path,
        experiment_id="asymmetric",
        codex_path=codex,
        embedding_model_dir=embedding,
        index_dir=index,
        metadata={},
        embedding_options={
            "query_prefix": "query:",
            "document_prefix": "passage:",
            "trust_remote_code": True,
            "precision": "bfloat16",
            "batch_size": 7,
        },
    )

    bundle = load_rag_bundle(manifest)

    assert bundle.embedding_query_prefix == "query:"
    assert bundle.embedding_document_prefix == "passage:"
    assert bundle.embedding_trust_remote_code is True
    assert bundle.embedding_precision == "bfloat16"
    assert bundle.embedding_batch_size == 7


def test_self_contained_bundle_uses_relative_codex_and_loads_from_directory(
    tmp_path: Path,
) -> None:
    codex = tmp_path / "codex.csv"
    codex.write_text("text,source\ntext,source\n", encoding="utf-8")
    embedding = tmp_path / "embedding_model"
    embedding.mkdir()
    index = tmp_path / "faiss_index"
    index.mkdir()
    (index / "index.faiss").write_bytes(b"faiss")
    (index / "index.pkl").write_bytes(b"metadata")

    manifest = write_rag_manifest(
        tmp_path,
        experiment_id="portable",
        codex_path=codex,
        embedding_model_dir=embedding,
        index_dir=index,
        metadata={},
    )

    value = json.loads(manifest.read_text(encoding="utf-8"))
    bundle = load_rag_bundle(tmp_path)

    assert value["codex_path"] == "codex.csv"
    assert bundle.manifest_path == manifest
    assert bundle.codex_path == codex
