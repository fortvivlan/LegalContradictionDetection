"""Excel result serialization shared by all workflows."""

from __future__ import annotations

import hashlib
import re
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

from .common import DEFAULT_RESULTS_DIR, json_value, validate_task


def concatenate_tables(tables: list[Any]):
    """Concatenate DataFrames, returning an empty DataFrame for no inputs."""
    import pandas as pd

    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def metadata_table(metadata: Mapping[str, Any]):
    """Convert run metadata to a two-column DataFrame."""
    import pandas as pd

    return pd.DataFrame(
        [{"key": key, "value": json_value(value)} for key, value in metadata.items()]
    )


def write_results_workbook(
    workflow_name: str,
    tables: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    output_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> Path:
    """Write one timestamped multi-sheet XLSX result workbook."""
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"{workflow_name}_{timestamp}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, table in tables.items():
            if table is None:
                table = pd.DataFrame()
            table.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        metadata_table(metadata).to_excel(writer, sheet_name="run_metadata", index=False)
    return path


def display_scores(scores) -> None:
    """Display a score table in notebooks, with a plain terminal fallback."""
    try:
        from IPython.display import display

        display(scores)
    except ModuleNotFoundError:
        print(scores.to_string(index=False))


def _safe_artifact_name(value: str, *, limit: int = 80) -> str:
    """Return a readable filename component safe on common operating systems."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip(" ._")
    return (cleaned or "document")[:limit]


def _review_archive_path(output_dir: Path, workflow_name: str, timestamp: str) -> Path:
    """Fit a review ZIP below the traditional Windows 260-character path limit."""
    suffix = f"_document_review_{timestamp}.zip"
    available = 259 - len(str(output_dir.resolve())) - 1 - len(suffix)
    if available < 9:
        raise ValueError(
            f"Review output directory is too long for a Windows ZIP path: {output_dir}"
        )
    safe_name = _safe_artifact_name(workflow_name)
    if len(safe_name) > available:
        digest = hashlib.sha256(workflow_name.encode("utf-8")).hexdigest()[:8]
        safe_name = f"{safe_name[:available - 9]}_{digest}"
    return output_dir / f"{safe_name}{suffix}"


def _metadata_text(value: object) -> str:
    if value is None or str(value) == "nan":
        return ""
    return str(value).strip()


def format_article_reference(
    source: object,
    citation_code: object = None,
    citation_article: object = None,
    citation_part: object = None,
    citation_point: object = None,
) -> str:
    """Format a retrieved codex source like a review-workbook article label."""
    source_text = _metadata_text(source)
    article_match = re.search(
        r"(?i)\bСтатья\s+([0-9]+(?:\.[0-9]+)*)", source_text
    )
    point = ""
    if article_match:
        code = source_text[: article_match.start()].strip().rstrip(":.;").strip()
        article = article_match.group(1)
        remainder = source_text[article_match.end() :]
        part_match = re.search(
            r"(?i)(?:\bч\.|\bчасть)\s*([0-9]+(?:\.[0-9]+)*)",
            remainder,
        )
        part = part_match.group(1) if part_match else ""
        point_match = re.search(
            r"(?i)(?:\bп\.|\bпункт)\s*([0-9]+(?:\.[0-9]+)*)",
            remainder,
        )
        point = point_match.group(1) if point_match else ""
    else:
        code = _metadata_text(citation_code)
        article = _metadata_text(citation_article)
        part = _metadata_text(citation_part)
        point = _metadata_text(citation_point)
    if not article:
        return ""
    components = [code, f"Статья {article}"] if code else [f"Статья {article}"]
    if part:
        components.append(f"Часть {part}")
    if point:
        components.append(f"Пункт {point}")
    return " ".join(components)


# Kept for compatibility with older internal imports and tests.
_article_reference = format_article_reference


def write_document_review_package(
    workflow_name: str,
    document_pairs,
    *,
    output_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> Path | None:
    """Create a ZIP containing one model-review workbook per document/task.

    Every document/task model workbook contains all classified premise pairs
    with formatted codex/article references and blank specialist fields for
    later scoring. ``None`` is returned when no document pairs were produced.
    """
    if document_pairs is None or document_pairs.empty:
        return None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = _review_archive_path(output_dir, workflow_name, timestamp)
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative_path, model_review in _document_review_tables(document_pairs):
            workbook = BytesIO()
            model_review.to_excel(
                workbook, sheet_name="model_predictions", index=False
            )
            archive.writestr(relative_path.as_posix(), workbook.getvalue())
    return archive_path


def _document_review_tables(document_pairs):
    """Yield archive-relative workbook paths and review tables."""
    required = {
        "document",
        "task",
        "sentence_index",
        "hypothesis",
        "premise",
        "source",
        "retrieval_rank",
        "prediction",
    }
    missing = sorted(required - set(document_pairs.columns))
    if missing:
        raise ValueError(
            "Document pair table lacks review-package columns: " + ", ".join(missing)
        )
    for task in document_pairs["task"].dropna().unique():
        validate_task(str(task))
    dataset_aware = "test_dataset" in document_pairs.columns and any(
        str(value) != "default"
        for value in document_pairs["test_dataset"].dropna().unique()
    )
    grouping = ["document"]
    sort_columns = ["document", "task", "sentence_index", "retrieval_rank"]
    if dataset_aware:
        grouping.insert(0, "test_dataset")
        sort_columns.insert(0, "test_dataset")
    ordered = document_pairs.sort_values(sort_columns, kind="stable")
    for group_key, document_rows in ordered.groupby(grouping, sort=False):
        if dataset_aware:
            dataset_name, document_name = group_key
            dataset_directory = Path(_safe_artifact_name(str(dataset_name)))
        else:
            document_name = group_key[0] if isinstance(group_key, tuple) else group_key
            dataset_directory = Path()
        document_slug = _safe_artifact_name(Path(str(document_name)).stem)
        for task, task_rows in document_rows.groupby("task", sort=False):
            model_review = task_rows.loc[
                :, ["hypothesis", "premise", "prediction"]
            ].rename(columns={"prediction": "model_prediction"})
            model_review.insert(
                2,
                "article_number",
                [
                    format_article_reference(
                        row.source,
                        getattr(row, "citation_code", None),
                        getattr(row, "citation_article", None),
                        getattr(row, "citation_part", None),
                        getattr(row, "citation_point", None),
                    )
                    for row in task_rows.itertuples(index=False)
                ],
            )
            model_review["expert_label"] = ""
            model_review["expert_comment"] = ""
            relative_path = dataset_directory / (
                f"{document_slug}_{_safe_artifact_name(str(task))}_model_predictions.xlsx"
            )
            yield relative_path, model_review


def _write_document_review_workbooks(document_pairs, target_dir: Path) -> list[Path]:
    """Write review workbooks below an existing target directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    workbooks: list[Path] = []
    for relative_path, model_review in _document_review_tables(document_pairs):
        model_path = target_dir / relative_path
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_review.to_excel(
            model_path, sheet_name="model_predictions", index=False
        )
        workbooks.append(model_path)
    return workbooks


def write_document_review_workbooks(
    workflow_name: str,
    document_pairs,
    *,
    output_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> Path | None:
    """Write persistent per-document prediction workbooks and return their directory."""
    if document_pairs is None or document_pairs.empty:
        return None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = output_dir / (
        f"{_safe_artifact_name(workflow_name)}_document_predictions_{timestamp}"
    )
    _write_document_review_workbooks(document_pairs, target)
    return target
