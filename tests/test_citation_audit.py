from pathlib import Path

import pandas as pd
import pytest

from LCD.shared.rag.citation_audit import run_citation_audit


def _write_annotations(root: Path) -> None:
    pd.DataFrame(
        [
            ["Exact", "КоАП РФ Статья 10 Часть 1", "gold"],
            ["Part differs", "КоАП РФ Статья 20 Часть 1", "gold"],
            ["Missed", "КоАП РФ Статья 30", "gold"],
            ["Extra", "КоАП РФ Статья 50", "gold"],
            ["Unresolved", "КоАП РФ Статья 60", "gold"],
        ]
    ).to_excel(root / "RAG_DIALOGUE_test.xlsx", header=False, index=False)
    full = pd.DataFrame([["Full exact", "КоАП РФ Статья 70", "gold"]])
    full.to_excel(root / "RAG_FULL_test.xlsx", header=False, index=False)
    full.to_excel(root / "RAG_FULL_additional_test.xlsx", header=False, index=False)


def _document_hypotheses(directory: Path) -> list[dict]:
    rows = (
        [
            ("Exact", "ч. 1 ст. 10 КоАП РФ"),
            ("Part differs", "ч. 2 ст. 20 КоАП РФ"),
            ("Missed", "Ссылка подразумевается экспертом"),
            ("Extra", "применена ст. 40 КоАП РФ"),
            ("Unresolved", "применена ст. 60 КоАП РФ"),
        ]
        if directory.name == "Dialogue"
        else [
            ("Full exact", "применена ст. 70 КоАП РФ"),
            ("Missing annotation", "применена ст. 80 КоАП РФ"),
        ]
    )
    return [
        {
            "document": f"{directory.name}.docx",
            "sentence_index": index,
            "hypothesis": text,
            "normalized_hypothesis": annotation.casefold(),
        }
        for index, (annotation, text) in enumerate(rows)
    ]


def test_citation_audit_separates_extraction_and_lookup_failures(
    monkeypatch, tmp_path: Path
) -> None:
    _write_annotations(tmp_path)
    pd.DataFrame(
        {
            "text": ["ten", "twenty-two", "forty", "seventy", "eighty"],
            "source": [
                "КоАП РФ: Статья 10. ч. 1.",
                "КоАП РФ: Статья 20. ч. 2.",
                "КоАП РФ: Статья 40.",
                "КоАП РФ: Статья 70.",
                "КоАП РФ: Статья 80.",
            ],
        }
    ).to_csv(tmp_path / "codex.csv", index=False)
    monkeypatch.setattr(
        "LCD.shared.rag.citation_audit._document_hypotheses",
        _document_hypotheses,
    )
    output = tmp_path / "audit.xlsx"

    with pytest.warns(UserWarning, match="missing from rag_tests"):
        result = run_citation_audit(
            codex_path=tmp_path / "codex.csv",
            rag_test_dir=tmp_path,
            test_docx_dir=tmp_path,
            output_path=output,
            selected_datasets=("Dialogue", "Full"),
        )

    assert result == output.resolve()
    workbook = pd.ExcelFile(output)
    assert workbook.sheet_names == [
        "summary",
        "hypotheses",
        "citation_comparison",
        "missing_hypotheses",
    ]

    summary = pd.read_excel(output, sheet_name="summary").set_index("dataset")
    assert summary.loc["ALL", "pipeline_hypotheses"] == 7
    assert summary.loc["ALL", "annotated_hypotheses"] == 6
    assert summary.loc["ALL", "expert_articles"] == 6
    assert summary.loc["ALL", "matched_articles"] == 4
    assert summary.loc["ALL", "missed_expert_articles"] == 2
    assert summary.loc["ALL", "article_extraction_recall"] == pytest.approx(4 / 6)
    assert summary.loc["ALL", "matched_full_references"] == 3
    assert summary.loc["ALL", "full_reference_recall"] == pytest.approx(3 / 6)
    assert summary.loc["ALL", "resolved_expert_articles"] == 3
    assert summary.loc["ALL", "rule_retrieval_recall"] == pytest.approx(3 / 6)
    assert summary.loc["ALL", "unresolved_detected_articles"] == 1
    assert summary.loc["ALL", "detected_with_missing_annotation"] == 1

    comparisons = pd.read_excel(output, sheet_name="citation_comparison")
    assert set(comparisons["comparison_status"]) == {
        "matched_full",
        "matched_article_subprovision_diff",
        "missed_expert",
        "extracted_not_in_expert",
        "unannotated_extraction",
    }
    unresolved = comparisons[comparisons["article"] == 60].iloc[0]
    assert unresolved["comparison_status"] == "matched_full"
    assert unresolved["exact_lookup_status"] == "unresolved"

    missing = pd.read_excel(output, sheet_name="missing_hypotheses")
    assert missing[["dataset", "document", "sentence_index"]].to_dict(
        orient="records"
    ) == [{"dataset": "FULL", "document": "Full.docx", "sentence_index": 1}]

    rules_output = tmp_path / "rules_audit.xlsx"
    with pytest.warns(UserWarning, match="missing from rag_tests"):
        run_citation_audit(
            codex_path=tmp_path / "codex.csv",
            rag_test_dir=tmp_path,
            test_docx_dir=tmp_path,
            output_path=rules_output,
            routing_scope="rules",
            selected_datasets=("Dialogue", "Full"),
        )
    rules_summary = pd.read_excel(
        rules_output, sheet_name="summary"
    ).set_index("dataset")
    assert rules_summary.loc["ALL", "routing_scope"] == "rules"
    assert rules_summary.loc["ALL", "pipeline_hypotheses"] == 5
    assert rules_summary.loc["ALL", "annotated_hypotheses"] == 4
    assert rules_summary.loc["ALL", "expert_articles"] == 4
    assert rules_summary.loc["ALL", "matched_articles"] == 3
    assert rules_summary.loc["ALL", "missed_expert_articles"] == 1
    assert rules_summary.loc["ALL", "article_extraction_recall"] == 0.75
    rules_hypotheses = pd.read_excel(rules_output, sheet_name="hypotheses")
    assert set(rules_hypotheses["retrieval_route"]) == {"rules"}
    rules_comparisons = pd.read_excel(
        rules_output, sheet_name="citation_comparison"
    )
    partial_failure = rules_comparisons[
        rules_comparisons["hypothesis"].str.contains("ст. 40")
    ]
    assert set(partial_failure["comparison_status"]) == {
        "missed_expert",
        "extracted_not_in_expert",
    }


def test_citation_audit_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "existing.xlsx"
    output.touch()

    with pytest.raises(FileExistsError, match="already exists"):
        run_citation_audit(output_path=output, codex_path=tmp_path / "missing.csv")


def test_citation_audit_rejects_unknown_routing_scope(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="all, rules, or faiss"):
        run_citation_audit(
            routing_scope="unknown",
            codex_path=tmp_path / "missing.csv",
        )
