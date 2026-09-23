from pathlib import Path

import pandas as pd

from LCD.shared.full_pipeline_analysis import run_full_pipeline_analysis


def test_analysis_places_expert_labels_beside_saved_predictions(tmp_path: Path) -> None:
    current = tmp_path / "autotest"
    legacy = tmp_path / "legacy"
    documents = tmp_path / "test_docx"
    output = tmp_path / "results"
    current.mkdir()
    legacy.mkdir()
    documents.mkdir()
    document = documents / "Тест_Иванов решение.docx"
    document.write_bytes(b"not parsed")
    pd.DataFrame(
        [
            {
                "hypothesis": "A hypothesis",
                "premise": "A premise",
                "article_number": "КоАП РФ Статья 1 Часть 1",
                "model_prediction": "",
                "expert_label": "not mentioned",
                "expert_comment": "check this pair",
            }
        ]
    ).to_excel(
        current / "Тест_Иванов_predictions.xlsx",
        sheet_name="model_predictions",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "sentence": "A hypothesis",
                "article": "КоАП РФ: Статья 1. п. 1. ",
                "premise": "КоАП РФ: Статья 1. п. 1. A premise",
                "answer": "contradiction",
            }
        ]
    ).to_excel(legacy / "result_Тест_Иванов_old.xlsx", index=False)
    source = tmp_path / "saved_results.xlsx"
    pairs = pd.DataFrame(
        [
            {
                "test_dataset": "Dialogue",
                "model": "ministral-lora",
                "task": "ternary",
                "document": document.name,
                "hypothesis_id": f"{document.name}:00000",
                "sentence_index": 0,
                "hypothesis": "A hypothesis",
                "premise": "A premise",
                "source": "КоАП РФ: Статья 1. ч. 1.",
                "retrieval_method": "exact",
                "prediction": "contradiction",
                "raw_output": "contradiction",
            }
        ]
    )
    saved_scores = pd.DataFrame(
        [
            {
                "test_dataset": "Dialogue",
                "evaluation_scope": "autotest_model",
                "contradiction_precision": 0.5,
            },
            {
                "test_dataset": "",
                "evaluation_scope": "validation",
                "contradiction_precision": 0.97,
            },
        ]
    )
    with pd.ExcelWriter(source, engine="openpyxl") as writer:
        pairs.to_excel(writer, sheet_name="document_pairs", index=False)
        saved_scores.to_excel(writer, sheet_name="scores", index=False)

    result = run_full_pipeline_analysis(
        source,
        previous_results_workbook=source,
        autotest_dir=current,
        docx_dir=documents,
        legacy_autotest_dir=legacy,
        output_dir=output,
    )

    workbook = pd.ExcelFile(result, engine="openpyxl")
    assert {
        "current_comparison",
        "false_positives",
        "legacy_comparison",
        "expert_snapshot_diff",
        "saved_metric_context",
        "prediction_run_summary",
    }.issubset(workbook.sheet_names)
    comparison = pd.read_excel(result, sheet_name="current_comparison")
    assert comparison.loc[0, "prediction"] == "contradiction"
    assert comparison.loc[0, "gold_label"] == "not mentioned"
    assert comparison.loc[0, "expert_comment"] == "check this pair"
    assert comparison.loc[0, "error_type"].startswith("false_positive")
    legacy_comparison = pd.read_excel(result, sheet_name="legacy_comparison")
    assert legacy_comparison.loc[0, "gold_label"] == "contradiction"
    notes = pd.read_excel(result, sheet_name="analysis_notes")
    assert "Dialogue contradiction precision denominator" in set(notes["finding"])
