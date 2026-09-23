import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from LCD.shared.common import file_sha256
from LCD.shared.rag.artifacts import RagBundle, RerankerBundle
from LCD.shared.rag.embedding_sweep import (
    BASELINE_VARIANT,
    CANDIDATE_VARIANT,
    _normalize_models,
    run_rag_embedding_sweep,
)
from LCD.shared.rag.embeddings import (
    DEFAULT_STAGE_THREE_MODELS,
    EmbeddingModelSpec,
    SentenceTransformerEmbeddings,
)
from LCD.shared.rag.evaluation import SUMMARY_COLUMNS
from LCD.experiments.lawbook.series_01_embedding_reranking.run_embedding_sweep import _parse_model


class _FakeSentenceTransformer:
    def __init__(self) -> None:
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return np.asarray([[float(index), 1.0] for index, _ in enumerate(texts)])


def test_default_models_use_their_required_asymmetric_query_inputs() -> None:
    by_alias = {spec.alias: spec for spec in DEFAULT_STAGE_THREE_MODELS}

    assert by_alias["bge_m3"].query_prefix == ""
    assert by_alias["bge_m3"].document_prefix == ""
    assert by_alias["qwen3_embedding_0_6b"].query_prefix.endswith("\nQuery:")
    assert by_alias["multilingual_e5_large_instruct"].query_prefix.endswith(
        "\nQuery: "
    )
    assert all(spec.document_prefix == "" for spec in DEFAULT_STAGE_THREE_MODELS)


def test_embedding_adapter_keeps_query_and_document_formatting_separate() -> None:
    adapter = SentenceTransformerEmbeddings.__new__(SentenceTransformerEmbeddings)
    adapter.model = _FakeSentenceTransformer()
    adapter.batch_size = 4
    adapter.normalize_embeddings = True
    adapter.query_prefix = "query:"
    adapter.document_prefix = "passage:"
    adapter.show_progress = False

    documents = adapter.embed_documents(["law one", "law two"])
    query = adapter.embed_query("decision")

    assert documents == [[0.0, 1.0], [1.0, 1.0]]
    assert query == [0.0, 1.0]
    assert adapter.model.calls[0][0] == ["passage:law one", "passage:law two"]
    assert adapter.model.calls[1][0] == ["query:decision"]
    assert all(call[1]["normalize_embeddings"] for call in adapter.model.calls)


def test_embedding_specs_and_cli_models_are_validated() -> None:
    assert _parse_model("bge_m3") == DEFAULT_STAGE_THREE_MODELS[0]
    assert _parse_model("custom=owner/model") == EmbeddingModelSpec(
        "custom", "owner/model"
    )
    with pytest.raises(ValueError, match="unique"):
        _normalize_models(
            [EmbeddingModelSpec("same", "one"), EmbeddingModelSpec("same", "two")]
        )
    with pytest.raises(ValueError, match="directory-safe"):
        _normalize_models([EmbeddingModelSpec("../escape", "owner/model")])


def test_stage_three_sweep_reports_baseline_once_and_ranks_candidates(
    monkeypatch, tmp_path: Path
) -> None:
    rag_dir = tmp_path / "dms-rag"
    rag_dir.mkdir()
    codex = rag_dir / "codex.csv"
    codex.write_text("text,source\nlegal text,КоАП РФ: Статья 1.\n", encoding="utf-8")
    winner_dir = tmp_path / "winner"
    winner_dir.mkdir()
    winner_manifest = winner_dir / "rag_manifest.json"
    winner_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "sbert_legal_v1",
                "codex_sha256": file_sha256(codex),
                "rag_commit": "rag-commit",
                "model_id": "ai-forever/sbert_large_nlu_ru",
                "resolved_revision": "sbert-revision",
                "reranker": {
                    "mode": "finetuned",
                    "local": True,
                    "model": "reranker_model",
                    "base_model_id": "reranker/base",
                    "base_model_revision": "reranker-revision",
                },
            }
        ),
        encoding="utf-8",
    )
    models = (
        EmbeddingModelSpec("first", "owner/first", query_prefix="query:"),
        EmbeddingModelSpec("second", "owner/second"),
    )
    artifact_root = tmp_path / "artifacts"
    results = tmp_path / "results"

    monkeypatch.setattr(
        "LCD.shared.rag.embedding_sweep.ensure_rag_repository",
        lambda path, revision: (rag_dir, revision),
    )
    monkeypatch.setattr(
        "LCD.shared.rag.embedding_sweep.resolve_huggingface_revision",
        lambda model_id, revision, token=None: f"revision-{model_id.rsplit('/', 1)[-1]}",
    )
    monkeypatch.setattr(
        "LCD.shared.rag.embedding_sweep.get_huggingface_token", lambda: None
    )

    reranker_bundle = RerankerBundle(
        "finetuned", str(winner_dir / "reranker_model"), None, True, 1024
    )
    baseline_bundle = RagBundle(
        "sbert_baseline", codex, rag_dir / "index", "sbert", None, False
    )
    winner_bundle = RagBundle(
        "winner",
        codex,
        winner_dir / "index",
        str(winner_dir / "embedding"),
        None,
        True,
        reranker=reranker_bundle,
    )

    def fake_load(source, *, name=None, codex_override=None):
        return baseline_bundle if Path(source) == rag_dir else winner_bundle

    monkeypatch.setattr(
        "LCD.shared.rag.embedding_sweep.load_rag_bundle", fake_load
    )
    monkeypatch.setattr(
        "LCD.shared.rag.embedding_sweep._completed_candidate",
        lambda *args, **kwargs: None,
    )

    def fake_build(spec, *, resolved_revision, candidate_dir, codex_path, **kwargs):
        candidate_dir.mkdir(parents=True)
        manifest = candidate_dir / "rag_manifest.json"
        manifest.write_text(spec.alias, encoding="utf-8")
        return RagBundle(
            spec.alias,
            codex_path,
            candidate_dir / "index",
            str(candidate_dir / "embedding"),
            None,
            True,
            manifest_path=manifest,
            embedding_query_prefix=spec.query_prefix,
        )

    monkeypatch.setattr(
        "LCD.shared.rag.embedding_sweep._build_candidate", fake_build
    )

    class FakeReranker:
        def __init__(self, model_id, **kwargs):
            self.model_id = model_id
            self.kwargs = kwargs

    monkeypatch.setattr(
        "LCD.shared.rag.embedding_sweep.CrossEncoderReranker", FakeReranker
    )
    calls = []

    def fake_evaluation(
        sources, *, evaluation_variants, results_dir, retrieval_depths, **kwargs
    ):
        calls.append((sources[1].name, evaluation_variants, retrieval_depths))
        values = {BASELINE_VARIANT: 0.80, CANDIDATE_VARIANT: 0.85}
        if sources[1].name == "second":
            values[CANDIDATE_VARIANT] = 0.75
        rows = []
        for variant in evaluation_variants:
            row = {column: 0.5 for column in SUMMARY_COLUMNS}
            row.update(
                {
                    "variant": variant,
                    "candidate_top_k": 100,
                    "final_top_k": 60,
                    "full_total_recall": values[variant],
                }
            )
            rows.append(row)
        frame = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
        target = Path(results_dir)
        target.mkdir(parents=True)
        with pd.ExcelWriter(target / "rag_recall.xlsx", engine="openpyxl") as writer:
            frame.to_excel(writer, sheet_name="recall", index=False)
            pd.DataFrame(columns=["dataset"]).to_excel(
                writer, sheet_name="missing_hypotheses", index=False
            )
        return frame

    monkeypatch.setattr(
        "LCD.shared.rag.embedding_sweep.run_rag_evaluation", fake_evaluation
    )
    monkeypatch.setattr(
        "LCD.shared.rag.embedding_sweep._release_embedding_model",
        lambda value=None: None,
    )

    report = run_rag_embedding_sweep(
        embedding_models=models,
        winner_manifest=winner_manifest,
        artifact_root=artifact_root,
        rag_dir=rag_dir,
        results_dir=results,
    )

    combined = pd.read_excel(report, sheet_name="recall")
    assert combined["model_alias"].tolist() == [
        "sbert_baseline",
        "first",
        "second",
    ]
    assert combined["full_total_recall_rank"].tolist() == [2, 1, 3]
    assert combined["full_total_recall_delta_vs_sbert"].tolist() == pytest.approx(
        [0.0, 0.05, -0.05]
    )
    assert combined["is_best_full"].tolist() == [False, True, False]
    assert calls == [
        (
            "first",
            (BASELINE_VARIANT, CANDIDATE_VARIANT),
            ((100, 60),),
        ),
        ("second", (CANDIDATE_VARIANT,), ((100, 60),)),
    ]
    config = json.loads((results / "evaluation_config.json").read_text("utf-8"))
    assert [model["alias"] for model in config["models"]] == ["first", "second"]
    assert config["retrieval_depth"] == {
        "candidate_top_k": 100,
        "final_top_k": 60,
    }
