"""Score fresh full-pipeline predictions against reviewed autotest workbooks."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .common import (
    DEFAULT_AUTOTEST_DIR,
    DEFAULT_RESULTS_DIR,
    DEFAULT_TEST_DOCX_DIR,
    LABELS_BY_TASK,
    Task,
    file_sha256,
    validate_task,
)
from .reporting import format_article_reference, write_results_workbook

REVIEW_COLUMNS = (
    "hypothesis",
    "premise",
    "article_number",
    "model_prediction",
    "expert_label",
    "expert_comment",
)
LEGACY_REVIEW_COLUMNS = ("hypothesis", "premise", "prediction")
RAW_LEGACY_REVIEW_COLUMNS = ("sentence", "article", "premise", "answer")
@dataclass
class AutotestScoringTables:
    """Tables produced by one task's full-pipeline benchmark scoring."""

    scores: Any
    per_class: Any
    confusion_matrix: Any
    rag_summary: Any
    alignment: Any
    inferred_gold: Any
    excluded: Any
    file_matching: Any


@dataclass(frozen=True)
class AutotestDataset:
    """One paired benchmark dataset discovered below the configured roots."""

    name: str
    autotest_dir: Path
    docx_dir: Path
    documents: tuple[Path, ...]
    file_matching: Any


def normalize_subject_key(path_or_name: str | Path) -> str:
    """Extract the benchmark subject key from a DOCX or review filename."""
    stem = Path(path_or_name).stem.replace("_", " ")
    words = re.findall(r"[^\s,]+", stem, flags=re.UNICODE)
    while words and words[0].casefold() in {"result", "тест", "ооо"}:
        words = words[1:]
    if not words:
        raise ValueError(f"Cannot determine a subject from {path_or_name!s}")
    return words[0].casefold()


def discover_autotest_cases(
    autotest_dir: str | Path = DEFAULT_AUTOTEST_DIR,
    docx_dir: str | Path = DEFAULT_TEST_DOCX_DIR,
) -> tuple[list[Path], Any]:
    """Validate XLSX-to-DOCX correspondence and return matched DOCX paths."""
    import pandas as pd

    autotest_dir = Path(autotest_dir)
    docx_dir = Path(docx_dir)
    if not autotest_dir.is_dir():
        raise FileNotFoundError(f"Autotest directory does not exist: {autotest_dir}")
    if not docx_dir.is_dir():
        raise FileNotFoundError(f"Benchmark DOCX directory does not exist: {docx_dir}")

    workbooks = sorted(
        (path for path in autotest_dir.glob("*.xlsx") if not path.name.startswith("~$")),
        key=lambda path: path.name.casefold(),
    )
    documents = sorted(docx_dir.glob("*.docx"), key=lambda path: path.name.casefold())
    workbook_by_key: dict[str, list[Path]] = {}
    document_by_key: dict[str, list[Path]] = {}
    for path in workbooks:
        workbook_by_key.setdefault(normalize_subject_key(path), []).append(path)
    for path in documents:
        document_by_key.setdefault(normalize_subject_key(path), []).append(path)

    rows: list[dict[str, object]] = []
    errors: list[str] = []
    matched: list[Path] = []
    for key, paths in sorted(workbook_by_key.items()):
        docs = document_by_key.get(key, [])
        if len(paths) != 1:
            errors.append(f"subject {key!r} has {len(paths)} XLSX files")
        if len(docs) != 1:
            errors.append(f"subject {key!r} has {len(docs)} DOCX files")
        status = "matched" if len(paths) == 1 and len(docs) == 1 else "ambiguous_or_missing"
        rows.append(
            {
                "subject_key": key,
                "xlsx": " | ".join(path.name for path in paths),
                "docx": " | ".join(path.name for path in docs),
                "status": status,
            }
        )
        if status == "matched":
            matched.append(docs[0])
    for key, docs in sorted(document_by_key.items()):
        if key not in workbook_by_key:
            rows.append(
                {
                    "subject_key": key,
                    "xlsx": "",
                    "docx": " | ".join(path.name for path in docs),
                    "status": "docx_without_xlsx",
                }
            )
    if errors:
        raise ValueError("Invalid autotest file correspondence: " + "; ".join(errors))
    return matched, pd.DataFrame(rows)


def discover_autotest_datasets(
    autotest_dir: str | Path = DEFAULT_AUTOTEST_DIR,
    docx_dir: str | Path = DEFAULT_TEST_DOCX_DIR,
    *,
    multiple_test: bool = False,
    selected_datasets: Sequence[str] = ("Full",),
) -> list[AutotestDataset]:
    """Discover the root benchmark or paired immediate child datasets."""
    autotest_root = Path(autotest_dir)
    docx_root = Path(docx_dir)
    if not multiple_test:
        documents, matching = discover_autotest_cases(autotest_root, docx_root)
        return [
            AutotestDataset(
                name="default",
                autotest_dir=autotest_root,
                docx_dir=docx_root,
                documents=tuple(documents),
                file_matching=matching,
            )
        ]

    if not autotest_root.is_dir():
        raise FileNotFoundError(f"Autotest directory does not exist: {autotest_root}")
    if not docx_root.is_dir():
        raise FileNotFoundError(f"Benchmark DOCX directory does not exist: {docx_root}")

    def child_directories(root: Path) -> dict[str, Path]:
        children: dict[str, Path] = {}
        for child in root.iterdir():
            if not child.is_dir() or child.name.startswith("."):
                continue
            key = child.name.casefold()
            if key in children:
                raise ValueError(
                    f"Ambiguous dataset folder names under {root}: "
                    f"{children[key].name!r} and {child.name!r}"
                )
            children[key] = child
        return children

    review_children = child_directories(autotest_root)
    docx_children = child_directories(docx_root)
    selected = {name.casefold() for name in selected_datasets}
    if not selected or len(selected) != len(selected_datasets):
        raise ValueError("selected_datasets must contain unique dataset names")
    review_children = {key: value for key, value in review_children.items() if key in selected}
    docx_children = {key: value for key, value in docx_children.items() if key in selected}
    missing_selected = selected - review_children.keys()
    if missing_selected:
        raise FileNotFoundError("Missing selected benchmark dataset(s): " + ", ".join(sorted(missing_selected)))
    missing_docx = sorted(
        review_children[key].name for key in review_children.keys() - docx_children.keys()
    )
    missing_reviews = sorted(
        docx_children[key].name for key in docx_children.keys() - review_children.keys()
    )
    if missing_docx or missing_reviews:
        details = []
        if missing_docx:
            details.append("missing under test_docx: " + ", ".join(missing_docx))
        if missing_reviews:
            details.append("missing under autotest: " + ", ".join(missing_reviews))
        raise ValueError("Unpaired benchmark dataset folders: " + "; ".join(details))
    if not review_children:
        raise ValueError(
            "multiple_test=True requires paired immediate child folders under "
            f"{autotest_root} and {docx_root}"
        )

    priority = {"dialogue": 0, "full": 1}
    ordered_keys = sorted(
        review_children,
        key=lambda key: (priority.get(key, 2), review_children[key].name.casefold()),
    )
    datasets: list[AutotestDataset] = []
    for key in ordered_keys:
        review_path = review_children[key]
        document_path = docx_children[key]
        documents, matching = discover_autotest_cases(review_path, document_path)
        datasets.append(
            AutotestDataset(
                name=review_path.name,
                autotest_dir=review_path,
                docx_dir=document_path,
                documents=tuple(documents),
                file_matching=matching,
            )
        )
    return datasets


def add_test_dataset(table, dataset_name: str):
    """Return a table copy labeled with its benchmark dataset."""
    labeled = table.copy()
    labeled.insert(0, "test_dataset", dataset_name)
    return labeled


def _normalized_text(value: object) -> str:
    return " ".join(str(value or "").replace("\xa0", " ").split()).casefold()


def _normalized_label(value: object) -> str:
    return _normalized_text(value)


def _normalized_article(value: object) -> str:
    text = _normalized_text(value).replace(":", " ")
    text = re.sub(r"(?<=\d)\.(?=\s|$)", "", text)
    return " ".join(text.split())


def _pair_key(hypothesis: object, premise: object, article: object) -> tuple[str, str, str]:
    return (
        _normalized_text(hypothesis),
        _normalized_text(premise),
        _normalized_article(article),
    )


def _strip_legacy_article_prefix(premise: object, article: object) -> str:
    """Remove the duplicated source heading used in raw 2025 review exports."""
    premise_text = str(premise or "").strip()
    article_text = str(article or "").strip()
    if article_text and premise_text.casefold().startswith(article_text.casefold()):
        return premise_text[len(article_text) :].strip()
    return premise_text


def _gold_for_task(label: str, task: Task) -> str:
    return label


def _current_article(row: object) -> str:
    return format_article_reference(
        getattr(row, "source", ""),
        getattr(row, "citation_code", None),
        getattr(row, "citation_article", None),
        getattr(row, "citation_part", None),
        getattr(row, "citation_point", None),
    )


def _read_review_workbook(path: Path):
    """Read current review workbooks and the legacy Dialogue workbook shape."""
    import pandas as pd

    workbook = pd.ExcelFile(path, engine="openpyxl")
    if "model_predictions" in workbook.sheet_names:
        sheet_name = "model_predictions"
    elif len(workbook.sheet_names) == 1:
        sheet_name = workbook.sheet_names[0]
    else:
        raise ValueError(
            f"{path} has no model_predictions sheet and multiple fallback sheets"
        )
    reviewed = pd.read_excel(
        workbook,
        sheet_name=sheet_name,
        dtype=str,
        keep_default_na=False,
    )
    if set(REVIEW_COLUMNS).issubset(reviewed.columns):
        reviewed.attrs["legacy_review"] = False
        return reviewed
    if set(LEGACY_REVIEW_COLUMNS).issubset(reviewed.columns):
        legacy = reviewed.loc[:, list(LEGACY_REVIEW_COLUMNS)].copy()
        legacy["article_number"] = [
            format_article_reference(premise) for premise in legacy["premise"]
        ]
        legacy["model_prediction"] = ""
        legacy["expert_label"] = legacy["prediction"]
        legacy["expert_comment"] = ""
        result = legacy.loc[:, list(REVIEW_COLUMNS)]
        result.attrs["legacy_review"] = True
        return result
    if set(RAW_LEGACY_REVIEW_COLUMNS).issubset(reviewed.columns):
        legacy = reviewed.loc[:, list(RAW_LEGACY_REVIEW_COLUMNS)].copy()
        legacy["hypothesis"] = legacy["sentence"]
        legacy["premise"] = [
            _strip_legacy_article_prefix(premise, article)
            for premise, article in zip(legacy["premise"], legacy["article"])
        ]
        legacy["article_number"] = legacy["article"]
        legacy["model_prediction"] = ""
        legacy["expert_label"] = legacy["answer"]
        legacy["expert_comment"] = ""
        result = legacy.loc[:, list(REVIEW_COLUMNS)]
        result.attrs["legacy_review"] = True
        result.attrs["raw_legacy_review"] = True
        return result
    missing_legacy = sorted(set(LEGACY_REVIEW_COLUMNS) - set(reviewed.columns))
    missing_raw_legacy = sorted(
        set(RAW_LEGACY_REVIEW_COLUMNS) - set(reviewed.columns)
    )
    missing_current = sorted(set(REVIEW_COLUMNS) - set(reviewed.columns))
    raise ValueError(
        f"{path} is missing current review columns "
        f"({', '.join(missing_current)}) and legacy columns "
        f"({', '.join(missing_legacy)}); it also lacks raw legacy columns "
        f"({', '.join(missing_raw_legacy)})"
    )


def _evaluate(rows, *, model_id: str, task: Task, scope: str):
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    labels = list(LABELS_BY_TASK[task])
    gold = rows["gold_label"].tolist() if not rows.empty else []
    predicted = rows["prediction"].tolist() if not rows.empty else []
    macro = precision_recall_fscore_support(
        gold, predicted, labels=labels, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        gold, predicted, labels=labels, average="weighted", zero_division=0
    )
    per_values = precision_recall_fscore_support(
        gold, predicted, labels=labels, average=None, zero_division=0
    )
    contradiction_index = labels.index("contradiction")
    scores = pd.DataFrame(
        [
            {
                "model": model_id,
                "task": task,
                "evaluation_scope": scope,
                "support": len(gold),
                "accuracy": accuracy_score(gold, predicted) if gold else 0.0,
                "macro_precision": macro[0],
                "macro_recall": macro[1],
                "macro_f1": macro[2],
                "weighted_f1": weighted[2],
                "contradiction_precision": per_values[0][contradiction_index],
                "contradiction_recall": per_values[1][contradiction_index],
                "contradiction_f1": per_values[2][contradiction_index],
                "invalid_predictions": sum(
                    prediction not in labels and prediction != "rag_miss"
                    for prediction in predicted
                ),
                "rag_misses": predicted.count("rag_miss"),
            }
        ]
    )
    per_class = pd.DataFrame(
        [
            {
                "model": model_id,
                "task": task,
                "evaluation_scope": scope,
                "label": label,
                "precision": per_values[0][index],
                "recall": per_values[1][index],
                "f1": per_values[2][index],
                "support": int(per_values[3][index]),
            }
            for index, label in enumerate(labels)
        ]
    )
    extras = sorted(set(predicted) - set(labels))
    matrix_labels = labels + extras
    matrix = confusion_matrix(gold, predicted, labels=matrix_labels)
    confusion = pd.DataFrame(
        [
            {
                "model": model_id,
                "task": task,
                "evaluation_scope": scope,
                "gold_label": gold_label,
                "predicted_label": predicted_label,
                "count": int(matrix[row_index, column_index]),
            }
            for row_index, gold_label in enumerate(matrix_labels)
            for column_index, predicted_label in enumerate(matrix_labels)
        ]
    )
    return scores, per_class, confusion


def score_autotest_predictions(
    document_pairs,
    evaluated_documents: Sequence[str | Path],
    *,
    model_id: str,
    task: Task,
    autotest_dir: str | Path = DEFAULT_AUTOTEST_DIR,
    docx_dir: str | Path = DEFAULT_TEST_DOCX_DIR,
    test_dataset: str = "default",
    benchmark_scopes: Sequence[str] = ("autotest_model", "autotest_total"),
) -> AutotestScoringTables:
    """Score fresh retrieved pairs against reviewed expert labels for one task."""
    import pandas as pd

    task = validate_task(task)
    if not benchmark_scopes or set(benchmark_scopes) - {"autotest_model", "autotest_total"} or len(set(benchmark_scopes)) != len(benchmark_scopes):
        raise ValueError("benchmark_scopes must contain unique supported scopes")
    _, file_matching = discover_autotest_cases(autotest_dir, docx_dir)
    workbook_by_key = {
        normalize_subject_key(path): path
        for path in Path(autotest_dir).glob("*.xlsx")
        if not path.name.startswith("~$")
    }
    document_names = {Path(path).name for path in evaluated_documents}
    evaluated_keys = {normalize_subject_key(name): name for name in document_names}

    pairs = document_pairs.copy()
    if not pairs.empty:
        required = {"document", "task", "hypothesis", "premise", "prediction", "source"}
        missing = sorted(required - set(pairs.columns))
        if missing:
            raise ValueError("Document pair table lacks scoring columns: " + ", ".join(missing))
        pairs = pairs[pairs["task"] == task].copy()

    model_rows: list[dict[str, object]] = []
    miss_rows: list[dict[str, object]] = []
    alignment_rows: list[dict[str, object]] = []
    inferred_rows: list[dict[str, object]] = []
    excluded_rows: list[dict[str, object]] = []
    manual_rag_rows: list[dict[str, str]] = []

    for subject_key, document_name in sorted(evaluated_keys.items()):
        workbook_path = workbook_by_key.get(subject_key)
        current = pairs[pairs["document"] == document_name] if not pairs.empty else pairs
        if workbook_path is None:
            for row in current.itertuples(index=False):
                excluded_rows.append(
                    {
                        "subject_key": subject_key,
                        "document": document_name,
                        "xlsx": "",
                        "excel_row": "",
                        "reason": "document has no reviewed workbook",
                    }
                )
            continue

        reviewed = _read_review_workbook(workbook_path)
        missing_columns = sorted(set(REVIEW_COLUMNS) - set(reviewed.columns))
        if missing_columns:
            raise ValueError(
                f"{workbook_path} is missing columns: {', '.join(missing_columns)}"
            )
        gold_by_key: dict[tuple[str, str, str], dict[str, object]] = {}
        gold_by_text_key: dict[tuple[str, str], dict[str, object]] = {}
        legacy_review = bool(reviewed.attrs.get("legacy_review"))
        for row_index, row in reviewed.iterrows():
            excel_row = int(row_index) + 2
            label = _normalized_label(row["expert_label"])
            if label not in LABELS_BY_TASK["ternary"]:
                excluded_rows.append(
                    {
                        "subject_key": subject_key,
                        "document": document_name,
                        "xlsx": workbook_path.name,
                        "excel_row": excel_row,
                        "reason": "blank expert_label" if not label else f"invalid expert_label: {label}",
                    }
                )
                continue
            key = _pair_key(row["hypothesis"], row["premise"], row["article_number"])
            if key in gold_by_key:
                if (
                    legacy_review
                    and gold_by_key[key]["original_gold_label"] == label
                ):
                    continue
                raise ValueError(
                    f"Ambiguous duplicate reviewed pair in {workbook_path.name}, row {excel_row}"
                )
            reviewed_data = {
                "original_gold_label": label,
                "gold_label": _gold_for_task(label, task),
                "excel_row": excel_row,
                "expert_comment": str(row["expert_comment"]).strip(),
                "expert_workbook": workbook_path.name,
                "hypothesis": str(row["hypothesis"]),
                "premise": str(row["premise"]),
                "article_number": str(row["article_number"]),
                "matched": False,
            }
            gold_by_key[key] = reviewed_data
            if legacy_review or not key[2]:
                text_key = key[:2]
                if text_key in gold_by_text_key:
                    previous = gold_by_text_key[text_key]
                    if previous["original_gold_label"] != label:
                        raise ValueError(
                            f"Ambiguous duplicate reviewed text pair in "
                            f"{workbook_path.name}, row {excel_row}"
                        )
                    continue
                gold_by_text_key[text_key] = reviewed_data
            if _normalized_text(row["expert_comment"]) == "раг":
                manual_rag_rows.append(
                    {
                        "document": document_name,
                        "original_gold_label": label,
                    }
                )

        seen_current: set[tuple[str, str, str]] = set()
        for row in current.itertuples(index=False):
            article = _current_article(row)
            key = _pair_key(row.hypothesis, row.premise, article)
            if key in seen_current:
                raise ValueError(
                    f"Ambiguous duplicate current pair for {document_name}: {article}"
                )
            seen_current.add(key)
            reviewed_row = gold_by_key.get(key)
            if reviewed_row is None:
                fallback_row = gold_by_text_key.get(key[:2])
                if fallback_row is not None and not fallback_row["matched"]:
                    reviewed_row = fallback_row
            prediction = _normalized_label(row.prediction)
            if prediction not in LABELS_BY_TASK[task]:
                prediction = "invalid"
            base = {
                "model": model_id,
                "task": task,
                "subject_key": subject_key,
                "document": document_name,
                "hypothesis_id": getattr(row, "hypothesis_id", ""),
                "sentence_index": getattr(row, "sentence_index", ""),
                "hypothesis": getattr(row, "hypothesis", ""),
                "premise": getattr(row, "premise", ""),
                "source": getattr(row, "source", ""),
                "article_number": article,
                "retrieval_method": getattr(row, "retrieval_method", ""),
                "retrieval_rank": getattr(row, "retrieval_rank", ""),
                "retrieval_initial_rank": getattr(
                    row, "retrieval_initial_rank", ""
                ),
                "retrieval_score": getattr(row, "retrieval_score", ""),
                "reranker_score": getattr(row, "reranker_score", ""),
                "prediction": prediction,
                "raw_output": getattr(row, "raw_output", ""),
            }
            if reviewed_row is None:
                original_gold = "not mentioned"
                gold_label = "not mentioned"
                base.update(
                    {
                        "gold_label": gold_label,
                        "original_gold_label": original_gold,
                        "gold_source": "inferred_for_new_retrieval_pair",
                        "excel_row": "",
                        "expert_workbook": "",
                        "expert_comment": "",
                    }
                )
                inferred_rows.append(dict(base))
            else:
                reviewed_row["matched"] = True
                base.update(
                    {
                        "gold_label": reviewed_row["gold_label"],
                        "original_gold_label": reviewed_row["original_gold_label"],
                        "gold_source": "expert",
                        "excel_row": reviewed_row["excel_row"],
                        "expert_workbook": reviewed_row["expert_workbook"],
                        "expert_comment": reviewed_row["expert_comment"],
                    }
                )
            model_rows.append(base)
            alignment_rows.append({**base, "alignment_status": "retrieved"})

        for reviewed_row in gold_by_key.values():
            if reviewed_row["matched"]:
                continue
            original_gold = str(reviewed_row["original_gold_label"])
            relevant_miss = original_gold in {"contradiction", "entailment"}
            status = "rag_miss" if relevant_miss else "irrelevant_not_retrieved"
            missing_row = {
                "model": model_id,
                "task": task,
                "subject_key": subject_key,
                "document": document_name,
                "hypothesis_id": "",
                "sentence_index": "",
                "hypothesis": reviewed_row["hypothesis"],
                "premise": reviewed_row["premise"],
                "source": "",
                "article_number": reviewed_row["article_number"],
                "retrieval_method": "",
                "retrieval_rank": "",
                "retrieval_initial_rank": "",
                "retrieval_score": "",
                "reranker_score": "",
                "prediction": "rag_miss",
                "raw_output": "",
                "gold_label": reviewed_row["gold_label"],
                "original_gold_label": original_gold,
                "gold_source": "expert",
                "excel_row": reviewed_row["excel_row"],
                "expert_workbook": reviewed_row["expert_workbook"],
                "expert_comment": reviewed_row["expert_comment"],
                "alignment_status": status,
            }
            alignment_rows.append(missing_row)
            if relevant_miss:
                miss_rows.append(missing_row)

    model_frame = pd.DataFrame(model_rows)
    score_tables = []
    for scope in benchmark_scopes:
        frame = model_frame if scope == "autotest_model" else pd.concat(
            [model_frame, pd.DataFrame(miss_rows)], ignore_index=True, sort=False
        )
        score_tables.append(_evaluate(frame, model_id=model_id, task=task, scope=scope))
    rag_rows = [
        {
            "model": model_id,
            "task": task,
            "scope": "total",
            "original_gold_label": label,
            "missed_pairs": sum(
                row["original_gold_label"] == label for row in miss_rows
            ),
            "manually_marked_rag": sum(
                row["original_gold_label"] == label for row in manual_rag_rows
            ),
        }
        for label in ("contradiction", "entailment", "not mentioned")
    ]
    for document_name in sorted(document_names):
        rag_rows.append(
            {
                "model": model_id,
                "task": task,
                "scope": document_name,
                "original_gold_label": "all_relevant",
                "missed_pairs": sum(row["document"] == document_name for row in miss_rows),
                "manually_marked_rag": sum(
                    row["document"] == document_name for row in manual_rag_rows
                ),
            }
        )
    return AutotestScoringTables(
        scores=add_test_dataset(
            pd.concat([table[0] for table in score_tables], ignore_index=True), test_dataset
        ),
        per_class=add_test_dataset(
            pd.concat([table[1] for table in score_tables], ignore_index=True),
            test_dataset,
        ),
        confusion_matrix=add_test_dataset(
            pd.concat([table[2] for table in score_tables], ignore_index=True),
            test_dataset,
        ),
        rag_summary=add_test_dataset(pd.DataFrame(rag_rows), test_dataset),
        alignment=add_test_dataset(pd.DataFrame(alignment_rows), test_dataset),
        inferred_gold=add_test_dataset(pd.DataFrame(inferred_rows), test_dataset),
        excluded=add_test_dataset(pd.DataFrame(excluded_rows), test_dataset),
        file_matching=add_test_dataset(file_matching, test_dataset),
    )


def run_autotest_scoring(
    prediction_source: str | Path | Any,
    *,
    task: str | None = None,
    autotest_dir: str | Path = DEFAULT_AUTOTEST_DIR,
    docx_dir: str | Path = DEFAULT_TEST_DOCX_DIR,
    multiple_test: bool = False,
    output_dir: str | Path = DEFAULT_RESULTS_DIR,
):
    """Score an existing document-pair DataFrame or results XLSX workbook.

    The generated workbook contains both model-only and total-pipeline metrics.
    If ``task`` is omitted, every task present in ``document_pairs`` is scored.
    ``multiple_test=True`` scores paired immediate child folders separately.
    The combined score table is returned.
    """
    import pandas as pd

    source_path: Path | None = None
    if isinstance(prediction_source, (str, Path)):
        source_path = Path(prediction_source)
        pairs = pd.read_excel(source_path, sheet_name="document_pairs", engine="openpyxl")
    else:
        pairs = prediction_source.copy()
    datasets = discover_autotest_datasets(
        autotest_dir, docx_dir, multiple_test=multiple_test
    )
    if multiple_test and "test_dataset" not in pairs.columns:
        raise ValueError(
            "Multi-dataset scoring requires a test_dataset column in document_pairs"
        )
    if multiple_test:
        expected_datasets = {dataset.name for dataset in datasets}
        observed_datasets = {
            str(value) for value in pairs["test_dataset"].dropna().unique()
        }
        unknown_datasets = sorted(observed_datasets - expected_datasets)
        if unknown_datasets:
            raise ValueError(
                "Unknown test_dataset value(s): " + ", ".join(unknown_datasets)
            )
    all_tables = []
    for dataset in datasets:
        dataset_pairs = (
            pairs[pairs["test_dataset"] == dataset.name].copy()
            if multiple_test
            else pairs
        )
        if multiple_test and dataset_pairs.empty:
            raise ValueError(
                f"document_pairs contains no rows for test_dataset={dataset.name!r}"
            )
        dataset_tasks = (
            [validate_task(task)]
            if task
            else [validate_task(value) for value in sorted(set(dataset_pairs["task"]))]
        )
        for current_task in dataset_tasks:
            task_pairs = dataset_pairs[dataset_pairs["task"] == current_task]
            all_tables.append(
                score_autotest_predictions(
                    dataset_pairs,
                    dataset.documents,
                    model_id=(
                        str(task_pairs["model"].iloc[0])
                        if not task_pairs.empty
                        else "unknown"
                    ),
                    task=current_task,
                    autotest_dir=dataset.autotest_dir,
                    docx_dir=dataset.docx_dir,
                    test_dataset=dataset.name,
                )
            )
    combine = lambda name: pd.concat(  # noqa: E731
        [getattr(tables, name) for tables in all_tables], ignore_index=True
    )
    scores = combine("scores")
    metadata = {
        "workflow": "autotest_scoring",
        "multiple_test": multiple_test,
        "test_datasets": [dataset.name for dataset in datasets],
        "prediction_source": source_path or "in_memory_dataframe",
        "prediction_sha256": file_sha256(source_path) if source_path else "not_applicable",
        "autotest_files": {
            f"{dataset.name}/{path.name}": file_sha256(path)
            for dataset in datasets
            for path in sorted(dataset.autotest_dir.glob("*.xlsx"))
        },
        "docx_files": {
            f"{dataset.name}/{path.name}": file_sha256(path)
            for dataset in datasets
            for path in sorted(dataset.docx_dir.glob("*.docx"))
        },
    }
    write_results_workbook(
        "autotest_scoring",
        {
            "scores": scores,
            "per_class": combine("per_class"),
            "confusion_matrix": combine("confusion_matrix"),
            "rag_summary": combine("rag_summary"),
            "alignment": combine("alignment"),
            "inferred_gold": combine("inferred_gold"),
            "excluded": combine("excluded"),
            "file_matching": pd.concat(
                [
                    add_test_dataset(dataset.file_matching, dataset.name)
                    for dataset in datasets
                ],
                ignore_index=True,
            ),
        },
        metadata,
        output_dir=output_dir,
    )
    return scores


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_workbook", type=Path)
    parser.add_argument("--task", choices=("ternary",))
    parser.add_argument("--autotest-dir", type=Path, default=DEFAULT_AUTOTEST_DIR)
    parser.add_argument("--docx-dir", type=Path, default=DEFAULT_TEST_DOCX_DIR)
    parser.add_argument("--multiple-test", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    print(
        run_autotest_scoring(
            arguments.results_workbook,
            task=arguments.task,
            autotest_dir=arguments.autotest_dir,
            docx_dir=arguments.docx_dir,
            multiple_test=arguments.multiple_test,
            output_dir=arguments.output_dir,
        ).to_string(index=False)
    )
