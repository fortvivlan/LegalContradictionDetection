"""Build an offline, row-level audit of saved full-pipeline predictions."""

from __future__ import annotations

import argparse
from copy import copy
from pathlib import Path
from typing import Any

from .autotest_scoring import (
    _normalized_text,
    _read_review_workbook,
    discover_autotest_cases,
    normalize_subject_key,
    score_autotest_predictions,
)
from .common import DEFAULT_RESULTS_DIR, file_sha256, validate_task
from .reporting import write_results_workbook


def _error_type(row: Any) -> str:
    prediction = str(row.prediction)
    gold = str(row.gold_label)
    status = str(row.alignment_status)
    if status == "irrelevant_not_retrieved":
        return "irrelevant_not_retrieved"
    if status == "rag_miss":
        return f"rag_miss_{gold.replace(' ', '_')}"
    if prediction == "contradiction" and gold == "contradiction":
        return "true_positive_contradiction"
    if prediction == "contradiction":
        return f"false_positive_contradiction_gold_{gold.replace(' ', '_')}"
    if gold == "contradiction":
        return f"false_negative_contradiction_pred_{prediction.replace(' ', '_')}"
    if prediction == gold:
        return "other_correct"
    return f"other_error_{gold.replace(' ', '_')}_as_{prediction.replace(' ', '_')}"


def _annotate_alignment(alignment):
    result = alignment.copy()
    if result.empty:
        result["is_correct"] = []
        result["error_type"] = []
        return result
    result["is_correct"] = [
        row.alignment_status == "retrieved" and row.prediction == row.gold_label
        for row in result.itertuples(index=False)
    ]
    result["error_type"] = [_error_type(row) for row in result.itertuples(index=False)]
    preferred = [
        "test_dataset",
        "subject_key",
        "document",
        "hypothesis_id",
        "sentence_index",
        "hypothesis",
        "premise",
        "source",
        "article_number",
        "prediction",
        "gold_label",
        "original_gold_label",
        "is_correct",
        "error_type",
        "alignment_status",
        "gold_source",
        "expert_comment",
        "expert_workbook",
        "excel_row",
        "retrieval_method",
        "retrieval_rank",
        "retrieval_initial_rank",
        "retrieval_score",
        "reranker_score",
        "raw_output",
        "model",
        "task",
    ]
    return result.loc[:, [column for column in preferred if column in result.columns]]


def _snapshot_table(autotest_dir: Path, snapshot: str):
    import pandas as pd

    rows: list[dict[str, object]] = []
    for workbook in sorted(
        path
        for path in autotest_dir.glob("*.xlsx")
        if not path.name.startswith("~$")
    ):
        reviewed = _read_review_workbook(workbook)
        for row_index, row in reviewed.iterrows():
            label = _normalized_text(row["expert_label"])
            rows.append(
                {
                    "snapshot": snapshot,
                    "subject_key": normalize_subject_key(workbook),
                    "expert_workbook": workbook.name,
                    "excel_row": int(row_index) + 2,
                    "hypothesis": str(row["hypothesis"]),
                    "premise": str(row["premise"]),
                    "article_number": str(row["article_number"]),
                    "expert_label": label,
                    "pair_key": "\u241f".join(
                        (
                            normalize_subject_key(workbook),
                            _normalized_text(row["hypothesis"]),
                            _normalized_text(row["premise"]),
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def _compare_snapshots(current, legacy):
    import pandas as pd

    def deduplicate(frame, prefix: str):
        if frame.empty:
            return frame
        grouped = frame.groupby("pair_key", sort=False, dropna=False)
        result = grouped.first().reset_index()
        result[f"{prefix}_duplicate_rows"] = grouped.size().to_numpy()
        return result.rename(
            columns={
                column: f"{prefix}_{column}"
                for column in result.columns
                if column != "pair_key" and not column.startswith(f"{prefix}_")
            }
        )

    current_unique = deduplicate(current, "current")
    legacy_unique = deduplicate(legacy, "legacy")
    comparison = legacy_unique.merge(
        current_unique, on="pair_key", how="outer", indicator=True
    )
    comparison["pair_status"] = comparison["_merge"].map(
        {"left_only": "legacy_only", "right_only": "current_only", "both": "shared"}
    )
    comparison["label_changed"] = (
        comparison["pair_status"].eq("shared")
        & comparison["legacy_expert_label"].ne(comparison["current_expert_label"])
    )
    comparison = comparison.drop(columns=["_merge", "pair_key"])
    leading = ["pair_status", "label_changed"]
    return comparison.loc[:, leading + [c for c in comparison.columns if c not in leading]]


def _alignment_summary(alignment, snapshot: str):
    import pandas as pd

    if alignment.empty:
        return pd.DataFrame()
    return (
        alignment.groupby(
            ["alignment_status", "gold_source", "error_type"],
            dropna=False,
        )
        .size()
        .rename("rows")
        .reset_index()
        .assign(annotation_snapshot=snapshot)
    )


def _snapshot_summary(*snapshots):
    import pandas as pd

    rows: list[dict[str, object]] = []
    for snapshot in snapshots:
        if snapshot.empty:
            continue
        name = str(snapshot["snapshot"].iloc[0])
        for label, count in snapshot["expert_label"].value_counts(dropna=False).items():
            rows.append(
                {
                    "annotation_snapshot": name,
                    "expert_label": label,
                    "rows": int(count),
                    "unique_pair_keys": int(
                        snapshot.loc[
                            snapshot["expert_label"].eq(label), "pair_key"
                        ].nunique()
                    ),
                    "workbooks": int(snapshot["expert_workbook"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def _analysis_notes(current_alignment):
    import pandas as pd

    retrieved = current_alignment[current_alignment["alignment_status"].eq("retrieved")]
    true_positives = int(
        retrieved["error_type"].eq("true_positive_contradiction").sum()
    )
    false_positives = int(
        retrieved["error_type"].str.startswith(
            "false_positive_contradiction", na=False
        ).sum()
    )
    exact_false_positives = int(
        (
            retrieved["retrieval_method"].eq("exact")
            & retrieved["error_type"].str.startswith(
                "false_positive_contradiction", na=False
            )
        ).sum()
    )
    faiss_false_positives = false_positives - exact_false_positives
    return pd.DataFrame(
        [
            {
                "finding": "Metric scope",
                "evidence": (
                    "The historical notebook classification reports use the "
                    "labeled validation pairs; this audit uses retrieved Dialogue pairs."
                ),
            },
            {
                "finding": "Dialogue contradiction precision denominator",
                "evidence": (
                    f"{true_positives} true-positive contradiction predictions and "
                    f"{false_positives} false positives: "
                    f"{true_positives}/({true_positives}+{false_positives})."
                ),
            },
            {
                "finding": "Retrieval path of false positives",
                "evidence": (
                    f"exact={exact_false_positives}; faiss={faiss_false_positives}."
                ),
            },
            {
                "finding": "How to inspect",
                "evidence": (
                    "Start with false_positives, then false_negatives and "
                    "expert_snapshot_diff; all text and expert labels are side by side."
                ),
            },
        ]
    )


def _compare_prediction_runs(current_pairs, previous_pairs):
    import pandas as pd

    key_columns = ["document", "hypothesis", "premise", "source"]

    def prepare(frame, prefix: str):
        result = frame.copy()
        result["pair_key"] = [
            "\u241f".join(_normalized_text(value) for value in values)
            for values in zip(*(result[column] for column in key_columns))
        ]
        result["pair_occurrence"] = result.groupby("pair_key", sort=False).cumcount()
        selected = [
            "pair_key",
            "pair_occurrence",
            "document",
            "hypothesis_id",
            "hypothesis",
            "premise",
            "source",
            "retrieval_method",
            "retrieval_rank",
            "prediction",
            "raw_output",
        ]
        selected = [column for column in selected if column in result.columns]
        return result.loc[:, selected].rename(
            columns={
                column: f"{prefix}_{column}"
                for column in selected
                if column not in {"pair_key", "pair_occurrence"}
            }
        )

    comparison = prepare(previous_pairs, "previous").merge(
        prepare(current_pairs, "current"),
        on=["pair_key", "pair_occurrence"],
        how="outer",
        indicator=True,
    )
    comparison["pair_status"] = comparison["_merge"].map(
        {"left_only": "previous_only", "right_only": "current_only", "both": "shared"}
    )
    comparison["prediction_changed"] = (
        comparison["pair_status"].eq("shared")
        & comparison["previous_prediction"].ne(comparison["current_prediction"])
    )
    comparison = comparison.drop(columns=["_merge", "pair_key"])
    leading = ["pair_status", "prediction_changed", "pair_occurrence"]
    return comparison.loc[:, leading + [c for c in comparison.columns if c not in leading]]


def _prediction_run_summary(comparison):
    import pandas as pd

    rows = [
        {
            "measure": "shared_pairs",
            "value": int(comparison["pair_status"].eq("shared").sum()),
        },
        {
            "measure": "previous_only_pairs",
            "value": int(comparison["pair_status"].eq("previous_only").sum()),
        },
        {
            "measure": "current_only_pairs",
            "value": int(comparison["pair_status"].eq("current_only").sum()),
        },
        {
            "measure": "shared_prediction_changes",
            "value": int(comparison["prediction_changed"].sum()),
        },
    ]
    previous_only = comparison[comparison["pair_status"].eq("previous_only")]
    for label, count in previous_only["previous_prediction"].value_counts().items():
        rows.append(
            {
                "measure": f"previous_only_prediction_{label}",
                "value": int(count),
            }
        )
    return pd.DataFrame(rows)


def _format_audit_workbook(path: Path) -> None:
    from openpyxl import load_workbook

    workbook = load_workbook(path)
    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            font = copy(cell.font)
            font.bold = True
            cell.font = font
        for column in sheet.columns:
            letter = column[0].column_letter
            header = str(column[0].value or "")
            if header in {"hypothesis", "premise", "source", "expert_comment"}:
                width = 70
            elif header in {"document", "expert_workbook", "error_type"}:
                width = 40
            else:
                width = min(max(len(header) + 2, 12), 28)
            sheet.column_dimensions[letter].width = width
    workbook.save(path)


def run_full_pipeline_analysis(
    results_workbook: str | Path,
    *,
    previous_results_workbook: str | Path | None = None,
    task: str = "ternary",
    test_dataset: str = "Dialogue",
    autotest_dir: str | Path = "autotest/Dialogue",
    docx_dir: str | Path = "test_docx/Dialogue",
    legacy_autotest_dir: str | Path | None = ".legacy",
    output_dir: str | Path = DEFAULT_RESULTS_DIR / "full_pipeline_analysis",
) -> Path:
    """Audit saved predictions against current and optional legacy expert labels.

    This workflow is entirely offline: it reads the ``document_pairs`` sheet,
    rescoring the saved outputs without loading model weights or rerunning RAG.
    The returned XLSX places prediction, expert label, hypothesis, premise, and
    retrieval metadata on the same row and includes focused error worksheets.
    """
    import pandas as pd

    results_path = Path(results_workbook)
    current_reviews = Path(autotest_dir)
    documents_dir = Path(docx_dir)
    task_name = validate_task(task)
    pairs = pd.read_excel(
        results_path, sheet_name="document_pairs", engine="openpyxl"
    )
    if "test_dataset" in pairs.columns:
        pairs = pairs[pairs["test_dataset"].astype(str) == test_dataset].copy()
    pairs = pairs[pairs["task"].astype(str) == task_name].copy()
    if pairs.empty:
        raise ValueError(
            f"No {task_name!r} document pairs for test_dataset={test_dataset!r}"
        )
    model_id = str(pairs["model"].iloc[0])
    documents, current_file_matching = discover_autotest_cases(
        current_reviews, documents_dir
    )
    current_tables = score_autotest_predictions(
        pairs,
        documents,
        model_id=model_id,
        task=task_name,
        autotest_dir=current_reviews,
        docx_dir=documents_dir,
        test_dataset=f"{test_dataset}_current",
    )
    current_alignment = _annotate_alignment(current_tables.alignment)

    tables: dict[str, Any] = {
        "analysis_notes": _analysis_notes(current_alignment),
        "metric_context": current_tables.scores.assign(
            annotation_snapshot="current", prediction_run="current"
        ),
        "current_comparison": current_alignment,
        "contradiction_review": current_alignment[
            current_alignment["prediction"].eq("contradiction")
            | current_alignment["gold_label"].eq("contradiction")
        ].copy(),
        "false_positives": current_alignment[
            current_alignment["error_type"].str.startswith(
                "false_positive_contradiction", na=False
            )
        ].copy(),
        "false_negatives": current_alignment[
            current_alignment["error_type"].str.startswith(
                "false_negative_contradiction", na=False
            )
            | current_alignment["error_type"].eq("rag_miss_contradiction")
        ].copy(),
        "current_per_class": current_tables.per_class,
        "current_confusion": current_tables.confusion_matrix,
        "current_alignment_summary": _alignment_summary(
            current_alignment, "current"
        ),
        "current_excluded": current_tables.excluded,
        "current_file_matching": current_file_matching,
    }

    current_snapshot = _snapshot_table(current_reviews, "current")
    legacy_path = Path(legacy_autotest_dir) if legacy_autotest_dir else None
    metadata: dict[str, Any] = {
        "workflow": "full_pipeline_analysis",
        "results_workbook": results_path,
        "results_sha256": file_sha256(results_path),
        "model": model_id,
        "task": task_name,
        "test_dataset": test_dataset,
        "autotest_dir": current_reviews,
        "docx_dir": documents_dir,
        "offline_only": True,
        "note": (
            "Validation and full-pipeline Dialogue metrics have different sample "
            "populations and must not be compared as if only parsing changed."
        ),
    }
    if legacy_path is not None:
        legacy_documents, legacy_file_matching = discover_autotest_cases(
            legacy_path, documents_dir
        )
        legacy_tables = score_autotest_predictions(
            pairs,
            legacy_documents,
            model_id=model_id,
            task=task_name,
            autotest_dir=legacy_path,
            docx_dir=documents_dir,
            test_dataset=f"{test_dataset}_legacy",
        )
        legacy_alignment = _annotate_alignment(legacy_tables.alignment)
        tables["metric_context"] = pd.concat(
            [
                tables["metric_context"],
                legacy_tables.scores.assign(
                    annotation_snapshot="legacy", prediction_run="current"
                ),
            ],
            ignore_index=True,
        )
        tables.update(
            {
                "legacy_comparison": legacy_alignment,
                "legacy_contradiction_review": legacy_alignment[
                    legacy_alignment["prediction"].eq("contradiction")
                    | legacy_alignment["gold_label"].eq("contradiction")
                ].copy(),
                "legacy_alignment_summary": _alignment_summary(
                    legacy_alignment, "legacy"
                ),
                "legacy_file_matching": legacy_file_matching,
            }
        )
        metadata["legacy_autotest_dir"] = legacy_path
        legacy_snapshot = _snapshot_table(legacy_path, "legacy")
        tables["expert_snapshot_diff"] = _compare_snapshots(
            current_snapshot, legacy_snapshot
        )
        tables["annotation_summary"] = _snapshot_summary(
            current_snapshot, legacy_snapshot
        )
    else:
        tables["annotation_summary"] = _snapshot_summary(current_snapshot)

    if previous_results_workbook is not None:
        previous_path = Path(previous_results_workbook)
        previous_pairs = pd.read_excel(
            previous_path, sheet_name="document_pairs", engine="openpyxl"
        )
        if "test_dataset" in previous_pairs.columns:
            previous_pairs = previous_pairs[
                previous_pairs["test_dataset"].astype(str).eq(test_dataset)
            ].copy()
        previous_pairs = previous_pairs[
            previous_pairs["task"].astype(str).eq(task_name)
        ].copy()
        run_comparison = _compare_prediction_runs(pairs, previous_pairs)
        tables["prediction_run_summary"] = _prediction_run_summary(run_comparison)
        tables["prediction_run_diff"] = run_comparison
        previous_current_tables = score_autotest_predictions(
            previous_pairs,
            documents,
            model_id=str(previous_pairs["model"].iloc[0]),
            task=task_name,
            autotest_dir=current_reviews,
            docx_dir=documents_dir,
            test_dataset=f"{test_dataset}_current",
        )
        metric_frames = [
            tables["metric_context"],
            previous_current_tables.scores.assign(
                annotation_snapshot="current", prediction_run="previous"
            ),
        ]
        previous_current_alignment = _annotate_alignment(
            previous_current_tables.alignment
        )
        tables["previous_current_summary"] = _alignment_summary(
            previous_current_alignment, "current"
        )
        if legacy_path is not None:
            previous_legacy_tables = score_autotest_predictions(
                previous_pairs,
                legacy_documents,
                model_id=str(previous_pairs["model"].iloc[0]),
                task=task_name,
                autotest_dir=legacy_path,
                docx_dir=documents_dir,
                test_dataset=f"{test_dataset}_legacy",
            )
            metric_frames.append(
                previous_legacy_tables.scores.assign(
                    annotation_snapshot="legacy", prediction_run="previous"
                )
            )
            tables["previous_legacy_summary"] = _alignment_summary(
                _annotate_alignment(previous_legacy_tables.alignment), "legacy"
            )
        tables["metric_context"] = pd.concat(metric_frames, ignore_index=True)
        metadata["previous_results_workbook"] = previous_path
        metadata["previous_results_sha256"] = file_sha256(previous_path)

    try:
        saved_scores = pd.read_excel(
            results_path, sheet_name="scores", engine="openpyxl"
        )
    except ValueError:
        saved_scores = pd.DataFrame()
    if not saved_scores.empty:
        validation_mask = saved_scores["evaluation_scope"].astype(str).eq(
            "validation"
        )
        dataset_mask = (
            saved_scores["test_dataset"].astype(str).eq(test_dataset)
            if "test_dataset" in saved_scores.columns
            else pd.Series(False, index=saved_scores.index)
        )
        selected = saved_scores[
            validation_mask
            | (
                dataset_mask
                & saved_scores["evaluation_scope"].astype(str).isin(
                    ["autotest_model", "autotest_total"]
                )
            )
        ].copy()
        selected.insert(0, "metric_origin", "saved_results")
        tables["saved_metric_context"] = selected

    output_path = write_results_workbook(
        "full_pipeline_analysis", tables, metadata, output_dir=output_dir
    )
    _format_audit_workbook(output_path)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_workbook", type=Path)
    parser.add_argument("--previous-results-workbook", type=Path)
    parser.add_argument("--task", choices=("ternary",), default="ternary")
    parser.add_argument("--test-dataset", default="Dialogue")
    parser.add_argument("--autotest-dir", type=Path, default=Path("autotest/Dialogue"))
    parser.add_argument("--docx-dir", type=Path, default=Path("test_docx/Dialogue"))
    parser.add_argument("--legacy-autotest-dir", type=Path, default=Path(".legacy"))
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_RESULTS_DIR / "full_pipeline_analysis"
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    print(
        run_full_pipeline_analysis(
            arguments.results_workbook,
            previous_results_workbook=arguments.previous_results_workbook,
            task=arguments.task,
            test_dataset=arguments.test_dataset,
            autotest_dir=arguments.autotest_dir,
            docx_dir=arguments.docx_dir,
            legacy_autotest_dir=arguments.legacy_autotest_dir,
            output_dir=arguments.output_dir,
        )
    )
