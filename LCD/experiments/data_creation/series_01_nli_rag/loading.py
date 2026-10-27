"""Registry-free loading and weighted sampling for one portable split folder."""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import random


CSV_COLUMNS = (
    "row_id",
    "hypothesis",
    "premise",
    "article_number",
    "context",
    "label",
    "error_type",
    "source_filename",
    "decision_text_path",
)
LABELS = ("contradiction", "entailment", "not mentioned")
RELEVANCE_LABELS = ("related", "unrelated")
DETECTION_TARGETS = {
    "contradiction": 0.50,
    "entailment": 0.25,
    "not mentioned": 0.25,
}
RAG_TARGETS = {"related": 0.50, "unrelated": 0.50}


def _portable_text(split_dir: Path, relative_value: str) -> Path:
    relative = Path(relative_value)
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or "\\" in relative_value
        or relative.suffix.casefold() != ".txt"
    ):
        raise ValueError(f"Unsafe or non-TXT decision_text_path: {relative_value!r}")
    current = split_dir
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Decision text cannot be a symlink: {relative_value}")
    if not current.is_file() or not current.resolve().is_relative_to(split_dir.resolve()):
        raise ValueError(f"Missing or escaping decision text: {relative_value}")
    # Validate the portable artifact's promised encoding while it is local.
    current.read_text(encoding="utf-8")
    return current


def load_split(csv_path: str | Path) -> list[dict[str, str]]:
    """Load a released nine-column CSV without consulting its build registry.

    The CSV parent is the portable split root. Every referenced decision must be
    an existing UTF-8 TXT below that directory. Row order is preserved, but row
    identity comes exclusively from the permanent ``row_id`` field.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(CSV_COLUMNS):
            raise ValueError(
                f"Dataset columns must be exactly {list(CSV_COLUMNS)}; "
                f"found {reader.fieldnames}"
            )
        rows = [dict(row) for row in reader]
    seen, checked_paths = set(), set()
    for row_number, row in enumerate(rows, 2):
        if set(row) != set(CSV_COLUMNS) or any(not isinstance(row.get(name), str) for name in CSV_COLUMNS):
            raise ValueError(f"CSV row {row_number} has a missing or non-string field")
        row_id = row["row_id"].strip()
        if not row_id or row_id != row["row_id"] or row_id in seen:
            raise ValueError(f"CSV row IDs must be nonempty and unique: {row_id!r}")
        seen.add(row_id)
        if row["label"] not in LABELS:
            raise ValueError(f"Unknown NLI label for {row_id}: {row['label']!r}")
        if (row["label"] == "contradiction") != bool(row["error_type"].strip()):
            raise ValueError(f"error_type must be populated only for contradiction: {row_id}")
        relative = row["decision_text_path"]
        if relative not in checked_paths:
            _portable_text(path.parent, relative)
            checked_paths.add(relative)
    return rows


def _metadata_relevance(csv_path: Path, active_ids: set[str]) -> dict[str, str]:
    path = csv_path.parent / "metadata.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"RAG sampling requires {path}")
    result = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid metadata JSON on {path}:{line_number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"Metadata record on {path}:{line_number} is not an object")
        row_id = str(value.get("row_id", "")).strip()
        if row_id not in active_ids:
            # Metadata for a CSV row deleted by an expert is deliberately inert.
            continue
        if row_id in result:
            raise ValueError(f"Duplicate metadata for surviving row: {row_id}")
        relevance = str(value.get("relevance", "")).strip()
        if relevance not in RELEVANCE_LABELS:
            raise ValueError(f"Missing or invalid relevance for {row_id}: {relevance!r}")
        result[row_id] = relevance
    missing = sorted(active_ids - result.keys())
    if missing:
        raise ValueError(f"Metadata lacks relevance for surviving rows: {missing}")
    return result


def _balanced_probabilities(values: dict[str, str], targets: dict[str, float],
                            task: str) -> dict[str, float]:
    counts = Counter(values.values())
    missing = [label for label in targets if not counts[label]]
    if missing:
        raise ValueError(f"Cannot sample {task}; missing classes: {missing}")
    return {
        row_id: targets[label] / counts[label]
        for row_id, label in values.items()
    }


def sampling_probabilities(
    csv_path: str | Path, *, task: str = "detection"
) -> dict[str, float]:
    """Recompute target sampling probabilities from surviving CSV row IDs."""
    path = Path(csv_path)
    rows = load_split(path)
    if task == "detection":
        values = {row["row_id"]: row["label"] for row in rows}
        return _balanced_probabilities(values, DETECTION_TARGETS, task)
    if task == "rag":
        row_ids = {row["row_id"] for row in rows}
        values = _metadata_relevance(path, row_ids)
        # Preserve CSV order rather than metadata order.
        ordered = {row["row_id"]: values[row["row_id"]] for row in rows}
        return _balanced_probabilities(ordered, RAG_TARGETS, task)
    raise ValueError("task must be 'detection' or 'rag'")


def sample_row_ids(
    csv_path: str | Path, *, task: str, count: int, seed: int = 42
) -> list[str]:
    """Sample permanent row IDs with replacement using portable probabilities."""
    if type(count) is not int or count < 0:
        raise ValueError("count must be a non-negative integer")
    probabilities = sampling_probabilities(csv_path, task=task)
    row_ids = list(probabilities)
    weights = [probabilities[row_id] for row_id in row_ids]
    return random.Random(seed).choices(row_ids, weights=weights, k=count)

