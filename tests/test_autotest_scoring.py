from pathlib import Path
from inspect import signature

import pandas as pd
import pytest

from LCD.shared.autotest_scoring import (
    discover_autotest_cases,
    discover_autotest_datasets,
    normalize_subject_key,
    run_autotest_scoring,
    score_autotest_predictions,
)
from LCD.shared.bert import run_bert_ternary
from LCD.shared.llm_evaluation import run_llm_evaluation
from LCD.shared.lora import run as run_lora


def _write_review(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "hypothesis": "matched hypothesis",
                "premise": "matched premise",
                "article_number": "КоАП РФ Статья 1 Часть 1",
                "model_prediction": "not mentioned",
                "expert_label": " contradiction ",
                "expert_comment": "",
            },
            {
                "hypothesis": "missed entailment",
                "premise": "entailing premise",
                "article_number": "КоАП РФ: Статья 2.",
                "model_prediction": "",
                "expert_label": "entailment",
                "expert_comment": "раг",
            },
            {
                "hypothesis": "irrelevant hypothesis",
                "premise": "irrelevant premise",
                "article_number": "КоАП РФ Статья 3",
                "model_prediction": "not mentioned",
                "expert_label": "not mentioned",
                "expert_comment": "",
            },
            {
                "hypothesis": "blank hypothesis",
                "premise": "blank premise",
                "article_number": "КоАП РФ Статья 4",
                "model_prediction": "not mentioned",
                "expert_label": "",
                "expert_comment": "",
            },
        ]
    ).to_excel(path, sheet_name="model_predictions", index=False)


def _setup_benchmark(tmp_path: Path) -> tuple[Path, Path, Path]:
    autotest = tmp_path / "autotest"
    documents = tmp_path / "test_docx"
    autotest.mkdir()
    documents.mkdir()
    workbook = autotest / "Тест_Иванов_ternary_model_predictions.xlsx"
    document = documents / "Тест_Иванов решение.docx"
    _write_review(workbook)
    document.write_bytes(b"not parsed by scoring")
    return autotest, documents, document


def _pairs(task: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "model",
                "task": task,
                "document": "Тест_Иванов решение.docx",
                "hypothesis_id": "matched",
                "hypothesis": " matched   hypothesis ",
                "premise": "matched premise",
                "source": "КоАП РФ: Статья 1. ч. 1.",
                "prediction": "contradiction",
            },
            {
                "model": "model",
                "task": task,
                "document": "Тест_Иванов решение.docx",
                "hypothesis_id": "new-correct",
                "hypothesis": "new hypothesis",
                "premise": "new premise",
                "source": "КоАП РФ: Статья 5.",
                "prediction": "not mentioned" if task == "ternary" else "no",
            },
            {
                "model": "model",
                "task": task,
                "document": "Тест_Иванов решение.docx",
                "hypothesis_id": "new-wrong",
                "hypothesis": "another new hypothesis",
                "premise": "another new premise",
                "source": "КоАП РФ: Статья 6.",
                "prediction": "entailment" if task == "ternary" else "contradiction",
            },
        ]
    )


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("Тест_Иванов решение.docx", "иванов"),
        ("Иванов_новый_predictions.xlsx", "иванов"),
        ("Тест_ООО Молоток решение.docx", "молоток"),
        ("ООО_ПроектСтрой results.xlsx", "проектстрой"),
    ],
)
def test_normalize_subject_key(filename: str, expected: str) -> None:
    assert normalize_subject_key(filename) == expected


def test_discover_cases_reports_extra_docx(tmp_path: Path) -> None:
    autotest, documents, expected_document = _setup_benchmark(tmp_path)
    (documents / "Петров решение.docx").write_bytes(b"extra")

    matched, table = discover_autotest_cases(autotest, documents)

    assert matched == [expected_document]
    assert set(table["status"]) == {"matched", "docx_without_xlsx"}


def test_discover_cases_rejects_xlsx_without_docx(tmp_path: Path) -> None:
    autotest = tmp_path / "autotest"
    documents = tmp_path / "test_docx"
    autotest.mkdir()
    documents.mkdir()
    _write_review(autotest / "Иванов_predictions.xlsx")

    with pytest.raises(ValueError, match="0 DOCX"):
        discover_autotest_cases(autotest, documents)


def test_discover_multiple_datasets_orders_dialogue_then_full(tmp_path: Path) -> None:
    autotest = tmp_path / "autotest"
    documents = tmp_path / "test_docx"
    for dataset_name in ("Extra", "Full", "Dialogue"):
        review_dir = autotest / dataset_name
        docx_dir = documents / dataset_name
        review_dir.mkdir(parents=True)
        docx_dir.mkdir(parents=True)
        _write_review(review_dir / "Тест_Иванов_results.xlsx")
        (docx_dir / "Иванов decision.docx").write_bytes(b"test")

    datasets = discover_autotest_datasets(
        autotest, documents, multiple_test=True
    )

    assert [dataset.name for dataset in datasets] == ["Full"]
    assert [len(dataset.documents) for dataset in datasets] == [1]


def test_discover_multiple_datasets_rejects_one_sided_folder(tmp_path: Path) -> None:
    autotest = tmp_path / "autotest"
    documents = tmp_path / "test_docx"
    (autotest / "Full").mkdir(parents=True)
    documents.mkdir()

    with pytest.raises(ValueError, match="Unpaired benchmark dataset folders"):
        discover_autotest_datasets(autotest, documents, multiple_test=True)


def test_ternary_scores_new_pairs_and_relevant_rag_misses(tmp_path: Path) -> None:
    autotest, documents, document = _setup_benchmark(tmp_path)

    tables = score_autotest_predictions(
        _pairs("ternary"),
        [document],
        model_id="model",
        task="ternary",
        autotest_dir=autotest,
        docx_dir=documents,
    )

    model = tables.scores.set_index("evaluation_scope").loc["autotest_model"]
    total = tables.scores.set_index("evaluation_scope").loc["autotest_total"]
    assert model["support"] == 3
    assert model["accuracy"] == pytest.approx(2 / 3)
    assert total["support"] == 4
    assert total["rag_misses"] == 1
    assert tables.rag_summary.query(
        "scope == 'total' and original_gold_label == 'entailment'"
    )["missed_pairs"].item() == 1
    assert set(tables.inferred_gold["gold_label"]) == {"not mentioned"}
    assert len(tables.excluded) == 1
    for table in (
        tables.scores,
        tables.per_class,
        tables.confusion_matrix,
        tables.rag_summary,
        tables.alignment,
        tables.inferred_gold,
        tables.excluded,
        tables.file_matching,
    ):
        assert set(table["test_dataset"]) == {"default"}


def test_dialogue_legacy_sheet_and_prediction_column_are_supported(
    tmp_path: Path,
) -> None:
    autotest = tmp_path / "autotest"
    documents = tmp_path / "test_docx"
    autotest.mkdir()
    documents.mkdir()
    premise = "reviewed semantic chunk without an embedded article heading"
    pd.DataFrame(
        [
            {
                "hypothesis": "reviewed hypothesis",
                "premise": premise,
                "prediction": "contradiction",
            }
        ]
    ).to_excel(autotest / "Иванов_results.xlsx", sheet_name="Sheet1", index=False)
    document = documents / "Иванов decision.docx"
    document.write_bytes(b"test")
    pairs = pd.DataFrame(
        [
            {
                "model": "model",
                "task": "ternary",
                "document": document.name,
                "hypothesis_id": "matched",
                "hypothesis": "reviewed hypothesis",
                "premise": premise,
                "source": "КоАП РФ: Статья 1. ч. 1.",
                "prediction": "contradiction",
            }
        ]
    )

    tables = score_autotest_predictions(
        pairs,
        [document],
        model_id="model",
        task="ternary",
        autotest_dir=autotest,
        docx_dir=documents,
        test_dataset="Dialogue",
    )

    model_score = tables.scores.set_index("evaluation_scope").loc["autotest_model"]
    assert model_score["support"] == 1
    assert model_score["accuracy"] == 1
    assert model_score["test_dataset"] == "Dialogue"


def test_raw_legacy_export_and_result_filename_are_supported(tmp_path: Path) -> None:
    autotest = tmp_path / "autotest"
    documents = tmp_path / "test_docx"
    autotest.mkdir()
    documents.mkdir()
    article = "КоАП РФ: Статья 20.20. п. 3. "
    pd.DataFrame(
        [
            {
                "sentence": "reviewed hypothesis",
                "article": article,
                "premise": article + "reviewed premise",
                "answer": "contradiction",
            }
        ]
    ).to_excel(
        autotest / "result_Тест_Иванов_2025.xlsx", sheet_name="Sheet1", index=False
    )
    document = documents / "Тест_Иванов решение.docx"
    document.write_bytes(b"test")
    pairs = pd.DataFrame(
        [
            {
                "model": "model",
                "task": "ternary",
                "document": document.name,
                "hypothesis_id": "matched",
                "sentence_index": 0,
                "hypothesis": "reviewed hypothesis",
                "premise": "reviewed premise",
                "source": "КоАП РФ: Статья 20.20. ч. 3.",
                "retrieval_method": "exact",
                "prediction": "contradiction",
                "raw_output": "contradiction",
            }
        ]
    )

    matched, _ = discover_autotest_cases(autotest, documents)
    tables = score_autotest_predictions(
        pairs,
        matched,
        model_id="model",
        task="ternary",
        autotest_dir=autotest,
        docx_dir=documents,
    )

    row = tables.alignment.iloc[0]
    assert row["gold_source"] == "expert"
    assert row["gold_label"] == "contradiction"
    assert row["hypothesis"] == "reviewed hypothesis"
    assert row["premise"] == "reviewed premise"
    assert row["raw_output"] == "contradiction"
    assert row["expert_workbook"].startswith("result_Тест_Иванов")


def test_model_only_scope_omits_total_score(tmp_path: Path) -> None:
    autotest, documents, document = _setup_benchmark(tmp_path)
    tables = score_autotest_predictions(
        _pairs("ternary"),
        [document],
        model_id="model",
        task="ternary",
        autotest_dir=autotest,
        docx_dir=documents,
        benchmark_scopes=("autotest_model",),
    )
    assert tables.scores["evaluation_scope"].tolist() == ["autotest_model"]


def test_previously_missed_pair_is_model_scored_when_retrieved(tmp_path: Path) -> None:
    autotest, documents, document = _setup_benchmark(tmp_path)
    pairs = _pairs("ternary")
    pairs.loc[len(pairs)] = {
        "model": "model",
        "task": "ternary",
        "document": document.name,
        "hypothesis_id": "now-retrieved",
        "hypothesis": "missed entailment",
        "premise": "entailing premise",
        "source": "КоАП РФ: Статья 2.",
        "prediction": "entailment",
    }

    tables = score_autotest_predictions(
        pairs,
        [document],
        model_id="model",
        task="ternary",
        autotest_dir=autotest,
        docx_dir=documents,
    )

    assert tables.scores.set_index("evaluation_scope").loc[
        "autotest_model", "support"
    ] == 4
    assert tables.scores.set_index("evaluation_scope").loc[
        "autotest_total", "rag_misses"
    ] == 0


@pytest.mark.parametrize(
    "workflow", [run_bert_ternary, run_llm_evaluation, run_lora]
)
def test_full_pipeline_workflows_expose_autotest_controls(workflow) -> None:
    parameters = signature(workflow).parameters
    assert parameters["document_paths"].default is None
    assert parameters["score_autotest"].default is True
    assert parameters["multiple_test"].default is False
    assert "autotest_dir" in parameters
    assert "test_docx_dir" in parameters


def test_standalone_scoring_keeps_duplicate_names_isolated_by_dataset(
    tmp_path: Path,
) -> None:
    autotest = tmp_path / "autotest"
    documents = tmp_path / "test_docx"
    frames = []
    for dataset_name in ("Full",):
        review_dir = autotest / dataset_name
        docx_dir = documents / dataset_name
        review_dir.mkdir(parents=True)
        docx_dir.mkdir(parents=True)
        _write_review(review_dir / "Тест_Иванов_results.xlsx")
        (docx_dir / "Тест_Иванов решение.docx").write_bytes(dataset_name.encode())
        frame = _pairs("ternary")
        frame.insert(0, "test_dataset", dataset_name)
        frames.append(frame)

    scores = run_autotest_scoring(
        pd.concat(frames, ignore_index=True),
        task="ternary",
        autotest_dir=autotest,
        docx_dir=documents,
        multiple_test=True,
        output_dir=tmp_path,
    )

    assert list(scores["test_dataset"]) == ["Full", "Full"]
