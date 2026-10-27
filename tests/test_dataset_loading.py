import csv
import json
from pathlib import Path
import shutil

import pytest

from LCD.experiments.data_creation.series_01_nli_rag.loading import (
    CSV_COLUMNS,
    load_split,
    sample_row_ids,
    sampling_probabilities,
)


def _split(tmp_path: Path) -> Path:
    split = tmp_path / "source" / "train"
    decisions = split / "decisions" / "originals"
    decisions.mkdir(parents=True)
    rows = []
    labels = [
        "contradiction",
        "contradiction",
        "entailment",
        "not mentioned",
        "not mentioned",
        "not mentioned",
    ]
    relevance = ["related", "related", "related", "unrelated", "unrelated", "unrelated"]
    metadata = []
    for number, (label, relation) in enumerate(zip(labels, relevance), 1):
        relative = f"decisions/originals/doc-{number}.txt"
        (split / relative).write_text(f"Решение {number}", encoding="utf-8")
        row_id = f"train_{number:06d}"
        rows.append({
            "row_id": row_id,
            "hypothesis": f"Гипотеза {number}",
            "premise": f"Норма {number}",
            "article_number": f"{number}.1",
            "context": f"Контекст {number}\nВторая строка",
            "label": label,
            "error_type": "тип" if label == "contradiction" else "",
            "source_filename": f"main/18.8/doc-{number}.docx",
            "decision_text_path": relative,
        })
        metadata.append({"row_id": row_id, "relevance": relation, "unused": {"x": number}})
    with (split / "train.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (split / "metadata.jsonl").write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in metadata),
        encoding="utf-8",
    )
    return split


def _rewrite(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_split_loads_after_relocation_without_registry(tmp_path: Path) -> None:
    source = _split(tmp_path)
    destination = tmp_path / "moved" / "renamed-split"
    shutil.copytree(source, destination)
    shutil.rmtree(source.parent)

    rows = load_split(destination / "train.csv")

    assert len(rows) == 6
    assert list(rows[0]) == list(CSV_COLUMNS)
    assert all(isinstance(value, str) for row in rows for value in row.values())
    assert not (destination / "registry.json").exists()


def test_deleted_and_reordered_rows_recompute_detection_probabilities(tmp_path: Path) -> None:
    split = _split(tmp_path)
    csv_path = split / "train.csv"
    rows = load_split(csv_path)
    # Delete one contradiction and one neutral row, then reverse the survivors.
    survivors = [row for row in rows if row["row_id"] not in {"train_000002", "train_000006"}]
    _rewrite(csv_path, list(reversed(survivors)))

    probabilities = sampling_probabilities(csv_path, task="detection")

    assert list(probabilities) == [row["row_id"] for row in reversed(survivors)]
    by_label = {label: 0.0 for label in ("contradiction", "entailment", "not mentioned")}
    for row in survivors:
        by_label[row["label"]] += probabilities[row["row_id"]]
    assert by_label == pytest.approx(
        {"contradiction": 0.50, "entailment": 0.25, "not mentioned": 0.25}
    )
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_rag_ignores_deleted_metadata_and_balances_surviving_relevance(tmp_path: Path) -> None:
    split = _split(tmp_path)
    csv_path = split / "train.csv"
    rows = load_split(csv_path)
    survivors = [row for row in rows if row["row_id"] not in {"train_000002", "train_000006"}]
    _rewrite(csv_path, survivors)
    with (split / "metadata.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"row_id": "deleted_extra", "relevance": "invalid"}) + "\n")

    probabilities = sampling_probabilities(csv_path, task="rag")

    assert set(probabilities) == {row["row_id"] for row in survivors}
    assert probabilities["train_000001"] + probabilities["train_000003"] == pytest.approx(0.5)
    assert probabilities["train_000004"] + probabilities["train_000005"] == pytest.approx(0.5)


def test_sampling_is_seeded_weighted_and_with_replacement(tmp_path: Path) -> None:
    csv_path = _split(tmp_path) / "train.csv"

    first = sample_row_ids(csv_path, task="detection", count=30, seed=9)
    second = sample_row_ids(csv_path, task="detection", count=30, seed=9)

    assert first == second
    assert len(first) == 30
    assert len(set(first)) < len(first)
    assert set(first) <= {f"train_{number:06d}" for number in range(1, 7)}


def test_missing_classes_or_relevance_fail_without_fallback(tmp_path: Path) -> None:
    split = _split(tmp_path)
    csv_path = split / "train.csv"
    rows = load_split(csv_path)
    _rewrite(csv_path, [row for row in rows if row["label"] != "entailment"])
    with pytest.raises(ValueError, match="missing classes.*entailment"):
        sampling_probabilities(csv_path, task="detection")

    _rewrite(csv_path, rows)
    metadata_path = split / "metadata.jsonl"
    metadata = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
    metadata = [value for value in metadata if value["row_id"] != "train_000001"]
    metadata_path.write_text(
        "".join(json.dumps(value) + "\n" for value in metadata), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="lacks relevance.*train_000001"):
        sampling_probabilities(csv_path, task="rag")


def test_duplicate_ids_and_escaping_decision_paths_are_rejected(tmp_path: Path) -> None:
    split = _split(tmp_path)
    csv_path = split / "train.csv"
    rows = load_split(csv_path)
    rows[1]["row_id"] = rows[0]["row_id"]
    _rewrite(csv_path, rows)
    with pytest.raises(ValueError, match="unique"):
        load_split(csv_path)

    rows[1]["row_id"] = "train_000002"
    rows[0]["decision_text_path"] = "../outside.txt"
    _rewrite(csv_path, rows)
    with pytest.raises(ValueError, match="Unsafe or non-TXT"):
        load_split(csv_path)

