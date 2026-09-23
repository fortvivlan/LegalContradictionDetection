import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from LCD.shared.rag.artifacts import RagBundle
from LCD.shared.rag.data import convert_embedding_dataset
from LCD.shared.rag import training as rag_training
from LCD.shared.rag.evaluation import (
    SUMMARY_COLUMNS,
    _merge_full_annotations,
    _validate_retrieval_depths,
    read_rag_workbook,
    run_rag_evaluation,
)
from LCD.shared.retrieval import (
    Citation,
    RetrievalOutcome,
    RetrievalRecord,
)
from LCD.experiments.lawbook.series_01_embedding_reranking.run_experiment import _parse_retrieval_depth


def test_embedding_conversion_merges_only_requested_labels(tmp_path: Path) -> None:
    source = tmp_path / "train.xlsx"
    pd.DataFrame(
        {
            "premise": ["p1", "p2", "p3"],
            "hypothesis": ["h1", "h2", "h3"],
            "source": ["s1", "s2", "s3"],
            "tag": ["contradiction", "entailment", "not mentioned"],
        }
    ).to_excel(source, index=False)

    converted = convert_embedding_dataset(source, "train")

    assert converted["embedding_tag"].tolist() == [
        "similar",
        "similar",
        "not mentioned",
    ]
    assert converted["label"].tolist() == [1, 1, 0]
    assert converted["example_id"].tolist() == [
        "train:000000",
        "train:000001",
        "train:000002",
    ]


def test_headerless_rag_workbook_groups_multiple_and_zero_golds(
    tmp_path: Path,
) -> None:
    workbook = tmp_path / "rag.xlsx"
    pd.DataFrame(
        [
            ["Sentence one", "КоАП РФ Статья 18.8 Часть 3.1", "text 1"],
            ["Sentence one", "КоАП РФ Статья 3.10 Часть 5", "text 2"],
            ["Sentence two", "", ""],
        ]
    ).to_excel(workbook, header=False, index=False)

    rows = read_rag_workbook(workbook)

    assert len(rows) == 2
    assert len(rows[0]["gold_citations"]) == 2
    assert rows[0]["gold_citations"][0].part == "3.1"
    assert rows[1]["gold_citations"] == ()


@pytest.mark.parametrize(
    ("depths", "message"),
    [
        ((), "cannot be empty"),
        (((20, 10), (20, 10)), "duplicates"),
        (((0, 1),), "positive"),
        (((10, 20),), "cannot exceed"),
    ],
)
def test_retrieval_depth_validation(depths, message) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_retrieval_depths(depths)


def test_cli_retrieval_depth_parser() -> None:
    assert _parse_retrieval_depth("20:10") == (20, 10)
    with pytest.raises(argparse.ArgumentTypeError, match="FINAL <= CANDIDATE"):
        _parse_retrieval_depth("10:20")


def _annotation(normalized: str, articles: tuple[str, ...]) -> dict:
    return {
        "normalized_hypothesis": normalized,
        "hypothesis": normalized,
        "gold_citations": tuple(Citation("КоАП РФ", value) for value in articles),
        "gold_references": articles,
        "gold_texts": (),
    }


def test_full_additional_annotations_are_fallback_only() -> None:
    primary = [_annotation("shared", ("1",)), _annotation("primary", ("2",))]
    additional = [
        _annotation("shared", ("1",)),
        _annotation("additional", ("3",)),
    ]

    merged = _merge_full_annotations(primary, additional)

    assert set(merged) == {"shared", "primary", "additional"}
    assert merged["shared"] is primary[0]


def test_conflicting_full_annotations_fail() -> None:
    with pytest.raises(ValueError, match="Conflicting Full"):
        _merge_full_annotations(
            [_annotation("shared", ("1",))],
            [_annotation("shared", ("2",))],
        )


class _FakeReranker:
    revision = "test"

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.calls = []

    def score(self, query, documents):
        self.calls.append((query, len(documents)))
        wanted = {"article-2" if query == "Full fallback" else "article-1"}
        return [1.0 if document in wanted else 0.0 for document in documents]


class _EvaluationRetriever:
    def retrieve_rules_with_details(self, hypothesis: str) -> RetrievalOutcome:
        records = ()
        if hypothesis in {"Dialogue positive", "Full primary"}:
            records = (self._record(1, "1", "exact"),)
        return RetrievalOutcome(records, records, False)

    def retrieve_semantic_with_details(
        self, hypothesis: str, *, top_k: int, final_top_k: int
    ) -> RetrievalOutcome:
        records = []
        for rank in range(1, top_k + 1):
            article = str(100 + rank)
            if rank == 15:
                article = "1"
            if rank == 30:
                article = "2"
            records.append(self._record(rank, article, "faiss"))
        values = tuple(records)
        return RetrievalOutcome(values, values[:final_top_k], False)

    @staticmethod
    def _record(rank: int, article: str, method: str) -> RetrievalRecord:
        return RetrievalRecord(
            premise=f"article-{article}",
            source=f"КоАП РФ: Статья {article}.",
            method=method,
            rank=rank,
            score=float(rank) if method == "faiss" else None,
            citation=Citation("КоАП РФ", article),
            initial_rank=rank,
        )


def _write_workbooks(root: Path) -> None:
    pd.DataFrame(
        [
            ["Dialogue positive", "КоАП РФ Статья 1", "gold"],
            ["Dialogue positive", "КоАП РФ Статья 2", "gold"],
            ["Dialogue fallback", "КоАП РФ Статья 1", "gold"],
            ["Dialogue zero", "", ""],
            ["Workbook only", "КоАП РФ Статья 9", "gold"],
        ]
    ).to_excel(root / "RAG_DIALOGUE_test.xlsx", header=False, index=False)
    pd.DataFrame(
        [
            ["Full primary", "КоАП РФ Статья 1", "gold"],
            ["Full zero", "", ""],
        ]
    ).to_excel(root / "RAG_FULL_test.xlsx", header=False, index=False)
    pd.DataFrame(
        [
            ["Full primary", "КоАП РФ Статья 1", "gold"],
            ["Full fallback", "КоАП РФ Статья 2", "gold"],
        ]
    ).to_excel(root / "RAG_FULL_additional_test.xlsx", header=False, index=False)


def _document_rows(directory: Path) -> list[dict]:
    hypotheses = (
        [
            "Dialogue positive",
            "Dialogue fallback",
            "Dialogue zero",
            "Dialogue missing",
        ]
        if directory.name == "Dialogue"
        else ["Full primary", "Full fallback", "Full zero", "Full missing"]
    )
    return [
        {
            "document": f"{directory.name}.docx",
            "sentence_index": index,
            "hypothesis": hypothesis,
            "normalized_hypothesis": hypothesis.casefold(),
        }
        for index, hypothesis in enumerate(hypotheses)
    ]


@pytest.mark.parametrize(("with_finetuned", "expected_rows"), [(False, 8), (True, 12)])
def test_compact_rag_matrix_and_depth_boundaries(
    monkeypatch, tmp_path: Path, with_finetuned: bool, expected_rows: int
) -> None:
    _write_workbooks(tmp_path)
    monkeypatch.setattr(
        "LCD.shared.rag.evaluation._document_hypotheses", _document_rows
    )
    monkeypatch.setattr(
        "LCD.shared.rag.evaluation.load_rag_bundle",
        lambda source: RagBundle(
            str(source), Path("codex"), Path("index"), "encoder", None, False
        ),
    )
    pretrained = _FakeReranker("pretrained")
    finetuned = _FakeReranker("finetuned") if with_finetuned else None
    output = tmp_path / "results"

    with pytest.warns(UserWarning, match="missing from rag_tests"):
        scores = run_rag_evaluation(
            ["baseline", "tuned"],
            rag_test_dir=tmp_path,
            test_docx_dir=tmp_path,
            results_dir=output,
            selected_datasets=("Dialogue", "Full"),
            retriever_factory=lambda bundle, device: _EvaluationRetriever(),
            pretrained_reranker=pretrained,
            finetuned_reranker=finetuned,
        )

    assert list(scores.columns) == list(SUMMARY_COLUMNS)
    assert len(scores) == expected_rows
    assert (
        "baseline_embeddings__finetuned_reranker" in set(scores.variant)
    ) is with_finetuned
    assert set(zip(scores.candidate_top_k, scores.final_top_k)) == {
        (20, 10),
        (40, 20),
    }
    baseline_no = scores[scores.variant == "baseline_embeddings__no_reranker"].set_index(
        "candidate_top_k"
    )
    assert baseline_no.loc[20, "dialogue_faiss_recall"] == 0
    assert baseline_no.loc[40, "dialogue_faiss_recall"] == 1
    assert baseline_no.loc[20, "dialogue_rules_recall"] == pytest.approx(0.5)
    assert baseline_no.loc[20, "dialogue_total_recall"] == pytest.approx(1 / 3)
    assert baseline_no.loc[40, "dialogue_total_recall"] == pytest.approx(2 / 3)
    assert baseline_no.loc[20, "full_rules_recall"] == 1
    pretrained_rows = scores[
        scores.variant == "baseline_embeddings__pretrained_reranker"
    ].set_index("candidate_top_k")
    assert pretrained_rows.loc[20, "dialogue_faiss_recall"] == 1
    assert pretrained_rows.loc[40, "dialogue_faiss_recall"] == 1
    assert pretrained_rows.loc[20, "full_faiss_recall"] == 0
    assert pretrained_rows.loc[40, "full_faiss_recall"] == 1
    assert pretrained_rows.loc[20, "full_total_recall"] == pytest.approx(0.5)
    assert pretrained_rows.loc[40, "full_total_recall"] == 1
    assert pretrained.calls
    assert all(candidate_count == 40 for _, candidate_count in pretrained.calls)
    assert {query for query, _ in pretrained.calls} == {
        "Dialogue fallback",
        "Full fallback",
    }

    workbook = pd.ExcelFile(output / "rag_recall.xlsx")
    assert workbook.sheet_names == ["recall", "missing_hypotheses"]
    missing = pd.read_excel(workbook, sheet_name="missing_hypotheses")
    assert set(missing.dataset) == {"DIALOGUE", "FULL"}
    assert len(missing) == 2
    assert pd.read_csv(output / "rag_recall.csv").shape == scores.shape


def test_focused_variant_skips_unused_bundle_and_pretrained_reranker(
    monkeypatch, tmp_path: Path
) -> None:
    _write_workbooks(tmp_path)
    monkeypatch.setattr(
        "LCD.shared.rag.evaluation._document_hypotheses", _document_rows
    )
    monkeypatch.setattr(
        "LCD.shared.rag.evaluation.load_rag_bundle",
        lambda source: RagBundle(
            str(source), Path("codex"), Path("index"), "encoder", None, False
        ),
    )
    created = []
    finetuned = _FakeReranker("finetuned")

    with pytest.warns(UserWarning, match="missing from rag_tests"):
        scores = run_rag_evaluation(
            ["baseline", "tuned"],
            rag_test_dir=tmp_path,
            test_docx_dir=tmp_path,
            results_dir=tmp_path / "focused",
            selected_datasets=("Dialogue", "Full"),
            retrieval_depths=((40, 20),),
            retriever_factory=lambda bundle, device: (
                created.append(bundle.name) or _EvaluationRetriever()
            ),
            finetuned_reranker=finetuned,
            evaluation_variants=(
                "baseline_embeddings__finetuned_reranker",
            ),
        )

    assert scores["variant"].tolist() == [
        "baseline_embeddings__finetuned_reranker"
    ]
    assert created == ["baseline"]
    assert {query for query, _ in finetuned.calls} == {
        "Dialogue fallback",
        "Full fallback",
    }


def test_completed_experiment_only_recalculates_scores(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    experiment_id = "complete"
    artifacts = tmp_path / "artifacts" / experiment_id
    artifacts.mkdir(parents=True)
    (artifacts / "run_config.json").write_text(
        json.dumps(
            {
                "model_id": "encoder",
                "rag_commit": "stored-rag-commit",
                "reranker": {"base_model_id": "reranker"},
            }
        ),
        encoding="utf-8",
    )
    (artifacts / "rag_manifest.json").write_text(
        json.dumps(
            {"reranker": {"mode": "finetuned", "local": True}}
        ),
        encoding="utf-8",
    )
    calls = {}

    monkeypatch.setattr(rag_training, "configure_reproducibility", lambda *a, **k: None)
    monkeypatch.setattr(
        rag_training,
        "ensure_rag_repository",
        lambda path, revision: calls.setdefault("rag", (Path(path), revision)),
    )
    monkeypatch.setattr(
        rag_training,
        "load_rag_bundle",
        lambda path: SimpleNamespace(
            reranker=SimpleNamespace(model=str(artifacts / "reranker_model"))
        ),
    )
    monkeypatch.setattr(
        rag_training,
        "_make_evaluation_rerankers",
        lambda **kwargs: ("pretrained", "finetuned", None, True, False, None),
    )
    monkeypatch.setattr(
        rag_training,
        "run_rag_evaluation",
        lambda sources, **kwargs: calls.setdefault(
            "evaluation", (sources, kwargs)
        ),
    )

    def unexpected_training(*args, **kwargs):
        pytest.fail("completed artifacts must bypass data conversion and training")

    monkeypatch.setattr(rag_training, "convert_embedding_dataset", unexpected_training)
    monkeypatch.setattr(rag_training, "_train_encoder", unexpected_training)
    monkeypatch.setattr(rag_training, "build_faiss_index", unexpected_training)
    monkeypatch.setattr(rag_training, "train_reranker", unexpected_training)

    result = rag_training.run_rag_experiment(
        experiment_id=experiment_id,
        model_id="encoder",
        reranker_mode="finetuned",
        reranker_model_id="reranker",
        rag_dir=tmp_path / "rag",
        artifact_root=tmp_path / "artifacts",
        results_root=tmp_path / "results",
    )

    assert result == calls["evaluation"]
    assert calls["rag"] == (tmp_path / "rag", "stored-rag-commit")
    sources, evaluation = calls["evaluation"]
    assert sources == [tmp_path / "rag", artifacts / "rag_manifest.json"]
    assert evaluation["pretrained_reranker"] == "pretrained"
    assert evaluation["finetuned_reranker"] == "finetuned"
    assert "skipping training" in capsys.readouterr().out
