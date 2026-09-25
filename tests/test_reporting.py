from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from LCD.shared.reporting import (
    write_document_review_package,
    write_document_review_workbooks,
)


def test_document_review_package_contains_only_ternary_workbooks_with_articles(
    tmp_path: Path,
) -> None:
    pairs = pd.DataFrame(
        [
            {
                "document": "decision.docx", "task": "ternary",
                "hypothesis_id": "decision.docx:00000", "sentence_index": 0,
                "hypothesis": "Sentence one.", "premise": "second-ranked text",
                "source": "КоАП РФ: Статья 18.8. ч. 1 п. 2.",
                "retrieval_rank": 2, "prediction": "not mentioned", "citation_article": "18.8",
            },
            {
                "document": "decision.docx", "task": "ternary",
                "hypothesis_id": "decision.docx:00000", "sentence_index": 0,
                "hypothesis": "Sentence one.", "premise": "top-ranked text",
                "source": "КоАП РФ: Статья 18.8. ч. 1 п. 1.",
                "retrieval_rank": 1, "prediction": "contradiction", "citation_article": "18.8",
            },
            {
                "document": "decision.docx", "task": "ternary",
                "hypothesis_id": "decision.docx:00001", "sentence_index": 1,
                "hypothesis": "Sentence two.", "premise": "article text 30.3",
                "source": "КоАП РФ: Статья 30.3. ч. 1.",
                "retrieval_rank": 1, "prediction": "entailment", "citation_article": "30.3",
            },
        ]
    )
    archive_path = write_document_review_package("ready_llm", pairs, output_dir=tmp_path)
    assert archive_path is not None
    with ZipFile(archive_path) as archive:
        assert archive.namelist() == ["decision_ternary_model_predictions.xlsx"]
        model = pd.read_excel(BytesIO(archive.read(archive.namelist()[0])))
        assert list(model.columns) == [
            "hypothesis", "premise", "article_number", "model_prediction",
            "expert_label", "expert_comment",
        ]
        assert len(model) == 3
        assert model.loc[:, ["hypothesis", "premise", "article_number"]].to_dict(
            orient="records"
        ) == [
            {"hypothesis": "Sentence one.", "premise": "top-ranked text", "article_number": "КоАП РФ Статья 18.8 Часть 1 Пункт 1"},
            {"hypothesis": "Sentence one.", "premise": "second-ranked text", "article_number": "КоАП РФ Статья 18.8 Часть 1 Пункт 2"},
            {"hypothesis": "Sentence two.", "premise": "article text 30.3", "article_number": "КоАП РФ Статья 30.3 Часть 1"},
        ]


def test_document_review_package_is_skipped_without_pairs(tmp_path: Path) -> None:
    assert (
        write_document_review_package(
            "lora", pd.DataFrame(), output_dir=tmp_path
        )
        is None
    )


def test_document_review_package_separates_multi_dataset_duplicate_names(
    tmp_path: Path,
) -> None:
    pairs = pd.DataFrame(
        [
            {
                "test_dataset": dataset_name,
                "document": "decision.docx",
                "task": "ternary",
                "sentence_index": 0,
                "hypothesis": "Sentence.",
                "premise": f"{dataset_name} premise",
                "source": "КоАП РФ: Статья 1.",
                "retrieval_rank": 1,
                "prediction": "not mentioned",
            }
            for dataset_name in ("Dialogue", "Full")
        ]
    )

    archive_path = write_document_review_package(
        "multi", pairs, output_dir=tmp_path
    )

    assert archive_path is not None
    with ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == {
            "Dialogue/decision_ternary_model_predictions.xlsx",
            "Full/decision_ternary_model_predictions.xlsx",
        }


def test_document_review_package_avoids_long_temporary_workbook_paths(
    tmp_path: Path, monkeypatch
) -> None:
    padding = max(2, 195 - len(str(tmp_path)) - 2)
    first = min(padding // 2, 80)
    second = min(padding - first, 80)
    output_dir = tmp_path / ("a" * first) / ("b" * second)
    pairs = pd.DataFrame(
        [
            {
                "test_dataset": "Full",
                "document": "Абдырахманов часть 5 статья 18.8_Готово.docx",
                "task": "ternary",
                "sentence_index": 0,
                "hypothesis": "Sentence.",
                "premise": "Premise.",
                "source": "КоАП РФ: Статья 18.8.",
                "retrieval_rank": 1,
                "prediction": "contradiction",
            }
        ]
    )
    original_to_excel = pd.DataFrame.to_excel

    def windows_limited_to_excel(self, excel_writer, *args, **kwargs):
        if isinstance(excel_writer, (str, Path)) and len(str(excel_writer)) >= 260:
            raise FileNotFoundError(excel_writer)
        return original_to_excel(self, excel_writer, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_excel", windows_limited_to_excel)
    archive_path = write_document_review_package("lora", pairs, output_dir=output_dir)

    assert archive_path is not None
    with ZipFile(archive_path) as archive:
        assert archive.namelist() == [
            "Full/Абдырахманов часть 5 статья 18.8_Готово_ternary_model_predictions.xlsx"
        ]


def test_document_review_package_shortens_windows_length_archive_path(
    tmp_path: Path,
) -> None:
    padding = 174 - len(str(tmp_path)) - 2
    output_dir = tmp_path / ("a" * min(padding, 80)) / ("b" * max(padding - 80, 1))
    workflow = "lora_mistralai_Ministral-8B-Instruct-2410_ternary"
    pairs = pd.DataFrame(
        [{
            "document": "decision.docx", "task": "ternary",
            "sentence_index": 0, "hypothesis": "Sentence.",
            "premise": "Premise.", "source": "КоАП РФ: Статья 1.",
            "retrieval_rank": 1, "prediction": "contradiction",
        }]
    )

    archive_path = write_document_review_package(workflow, pairs, output_dir=output_dir)

    assert archive_path is not None
    assert len(str(output_dir / f"{workflow}_document_review_20260924T180225Z.zip")) > 259
    assert len(str(archive_path.resolve())) <= 259
    with ZipFile(archive_path) as archive:
        assert archive.namelist() == ["decision_ternary_model_predictions.xlsx"]


def test_document_review_workbooks_remain_available_as_xlsx_files(
    tmp_path: Path,
) -> None:
    pairs = pd.DataFrame(
        [
            {
                "test_dataset": "Full",
                "document": "decision.docx",
                "task": "ternary",
                "sentence_index": 0,
                "hypothesis": "Sentence.",
                "premise": "Premise.",
                "source": "КоАП РФ: Статья 1.",
                "retrieval_rank": 1,
                "prediction": "not mentioned",
            }
        ]
    )

    directory = write_document_review_workbooks(
        "lora", pairs, output_dir=tmp_path
    )

    assert directory is not None
    workbook = directory / "Full" / "decision_ternary_model_predictions.xlsx"
    assert workbook.is_file()
    assert pd.read_excel(workbook).loc[0, "model_prediction"] == "not mentioned"
