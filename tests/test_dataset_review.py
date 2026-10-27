import csv
import json
from pathlib import Path

import pytest

from LCD.experiments.data_creation.series_01_nli_rag import review
from LCD.experiments.data_creation.series_01_nli_rag.review import (
    CSV_COLUMNS,
    export_split,
    import_annotations,
    import_review_csv,
    register_rows,
)


RULE_NAME = "Неверно указан срок обжалования постановления"


def _bundle(tmp_path: Path, split_name: str = "train") -> Path:
    root = tmp_path / "bundle"
    split = root / split_name
    (split / "decisions" / "originals").mkdir(parents=True)
    (root / "rules.json").write_text(
        json.dumps({"rules": [{"rule_id": "xlsx_row_32", "name": RULE_NAME}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        json.dumps({"policy": {"val_contradiction_documents": 120,
                                "test_contradiction_documents": 40}}),
        encoding="utf-8",
    )
    for number in range(1, 4):
        (split / "decisions" / "originals" / f"d{number}.txt").write_text(
            f"Решение {number}", encoding="utf-8"
        )
    return split


def _prepared(number: int) -> dict:
    return {
        "stable_key": f"source-{number}:hypothesis:unit",
        "hypothesis": f"Гипотеза {number}",
        "premise": f"Норма {number}",
        "article_number": "30.3",
        "context": "Контекст 1\nКонтекст 2",
        "label": "",
        "error_type": "",
        "source_filename": f"d{number}.docx",
        "decision_text_path": f"decisions/originals/d{number}.txt",
        "source_id": f"source-{number}",
        "variant_id": "original",
        "active": True,
        "metadata": {"unit_id": f"unit-{number}"},
    }


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def test_register_rows_allocates_permanent_ids_and_does_not_renumber(tmp_path: Path) -> None:
    split = _bundle(tmp_path)
    assert register_rows(split, [_prepared(1), _prepared(2)]) == [
        "train_000001",
        "train_000002",
    ]
    assert register_rows(split, [_prepared(2), _prepared(3)]) == [
        "train_000002",
        "train_000003",
    ]

    registry = json.loads((split / "registry.json").read_text(encoding="utf-8"))
    assert registry["next_id"] == 4
    assert registry["rows"]["source-2:hypothesis:unit"]["row_id"] == "train_000002"


def test_train_release_uses_stratified_human_sample_and_all_synthetic_rows(tmp_path: Path) -> None:
    from LCD.experiments.data_creation.series_01_nli_rag.review_policy import plan_human_review

    split = _bundle(tmp_path)
    records = []
    for number in range(1, 31):
        label = "contradiction" if number == 1 else "entailment" if number == 2 else "not mentioned"
        records.append({**_prepared(number), "decision_text_path": "decisions/originals/d1.txt",
                        "status": "proposed", "label": label,
                        "relevance": "related" if number <= 2 else "unrelated",
                        "error_type": RULE_NAME if number == 1 else "",
                        "variant_id": "syn_example" if number == 3 else "original"})
    register_rows(split, records)
    plan = plan_human_review(split, fraction=0.1, seed=42)
    assert 3 <= plan["required"] < 30
    assert plan["uncertain_jsonl_rows"] == 0
    with (split / "human_review.csv").open(encoding="utf-8-sig", newline="") as stream:
        required_ids = {r["row_id"] for r in csv.DictReader(stream)}
    assert "train_000003" in required_ids
    with pytest.raises(ValueError, match="without human approval"):
        export_split(split, released=True)
    import_review_csv(split, split / "human_review.csv", sampled=True)
    assert export_split(split, released=True)["pairs"] == 30
    registry = json.loads((split / "registry.json").read_text(encoding="utf-8"))
    assert any(r["status"] == "proposed" for r in registry["rows"].values())

    register_rows(split, [{**records[-1], "stable_key": "new-after-audit"}])
    with pytest.raises(ValueError, match="stale"):
        export_split(split, released=True)


def test_moved_split_retains_expert_rule_lookup_for_review(tmp_path: Path) -> None:
    import shutil

    split = _bundle(tmp_path)
    register_rows(split, [_prepared(1)])
    export_split(split)
    moved = tmp_path / "elsewhere" / "train"
    moved.parent.mkdir()
    shutil.copytree(split, moved)
    shutil.rmtree(split.parent)
    annotation = tmp_path / "annotated.jsonl"
    _write_jsonl(annotation, [{"row_id": "train_000001", "label": "contradiction",
                              "relevance": "related", "error_type": "xlsx_row_32",
                              "rationale": "Проверка по приложенной таблице типов."}])
    assert import_annotations(moved, annotation) == {"proposed": 1}
    assert export_split(moved)["pairs"] == 1


def test_annotation_import_is_three_class_and_relevance_is_independent(tmp_path: Path) -> None:
    split = _bundle(tmp_path)
    register_rows(split, [_prepared(1), _prepared(2), _prepared(3)])
    annotations = tmp_path / "annotations.jsonl"
    _write_jsonl(
        annotations,
        [
            {"row_id": "train_000001", "label": "contradiction", "relevance": "related",
             "error_type": "xlsx_row_32", "rationale": "x"},
            {"row_id": "train_000002", "label": "entailment", "relevance": "related",
             "rationale": "y"},
            {"row_id": "train_000003", "label": "not mentioned", "relevance": "unrelated",
             "rationale": "z"},
        ],
    )
    assert import_annotations(split, annotations) == {"proposed": 3}
    report = export_split(split)

    assert report["labels"] == {
        "contradiction": 1,
        "entailment": 1,
        "not mentioned": 1,
    }
    assert report["samplers"]["detection"]["complete"]
    assert report["samplers"]["rag"]["complete"]
    assert report["contradiction_documents"] == 1
    assert report["contradiction_provenance"] == {
        "natural": {"pairs": 1, "documents": 1},
        "synthetic": {"pairs": 0, "documents": 0},
    }
    assert report["variants"] == {"original": 3, "synthetic": 0}
    assert report["error_type_documents"] == {RULE_NAME: 1}
    registry = json.loads((split / "registry.json").read_text(encoding="utf-8"))
    first = registry["rows"]["source-1:hypothesis:unit"]
    assert first["error_type"] == RULE_NAME
    assert first["rule_id"] == "xlsx_row_32"
    rows = list(csv.DictReader((split / "sampling_weights.csv").open(encoding="utf-8")))
    assert [float(row["detection_weight"]) for row in rows] == [0.5, 0.25, 0.25]
    assert [float(row["rag_weight"]) for row in rows] == [0.25, 0.25, 0.5]


def test_missing_sampler_class_leaves_weights_blank(tmp_path: Path) -> None:
    split = _bundle(tmp_path)
    register_rows(split, [_prepared(1)])
    annotations = tmp_path / "annotations.jsonl"
    _write_jsonl(
        annotations,
        [{"row_id": "train_000001", "label": "entailment", "relevance": "related",
          "rationale": "x"}],
    )
    import_annotations(split, annotations)
    report = export_split(split)

    assert report["samplers"]["detection"] == {
        "complete": False,
        "missing": ["contradiction", "not mentioned"],
    }
    weights = next(csv.DictReader((split / "sampling_weights.csv").open(encoding="utf-8")))
    assert weights["detection_weight"] == ""
    assert weights["rag_weight"] == ""


def test_review_csv_order_is_irrelevant_and_omission_persists(tmp_path: Path) -> None:
    split = _bundle(tmp_path)
    register_rows(split, [_prepared(1), _prepared(2), _prepared(3)])
    annotations = tmp_path / "annotations.jsonl"
    _write_jsonl(
        annotations,
        [
            {"row_id": "train_000001", "label": "contradiction", "relevance": "related",
             "error_type": RULE_NAME, "rationale": "x"},
            {"row_id": "train_000002", "label": "entailment", "relevance": "related",
             "rationale": "y"},
            {"row_id": "train_000003", "label": "not mentioned", "relevance": "unrelated",
             "rationale": "z"},
        ],
    )
    import_annotations(split, annotations)
    export_split(split)
    with (split / "train.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    review = tmp_path / "review.csv"
    with review.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(rows[2])
        corrected = dict(rows[0], label="entailment", error_type="")
        writer.writerow(corrected)

    assert import_review_csv(split, review) == {"approved": 2, "excluded": 1}
    assert register_rows(split, [_prepared(2)]) == ["train_000002"]
    registry = json.loads((split / "registry.json").read_text(encoding="utf-8"))
    assert registry["rows"]["source-2:hypothesis:unit"]["status"] == "excluded"
    assert registry["rows"]["source-1:hypothesis:unit"]["label"] == "entailment"
    with pytest.raises(ValueError, match="without human approval"):
        # No active unapproved rows now, so create a fourth pending row first.
        register_rows(split, [{**_prepared(3), "stable_key": "new-key"}])
        export_split(split, released=True)


def test_review_rejects_text_changes_and_export_rejects_symlinks(tmp_path: Path) -> None:
    split = _bundle(tmp_path)
    register_rows(split, [_prepared(1)])
    annotations = tmp_path / "annotations.jsonl"
    _write_jsonl(
        annotations,
        [{"row_id": "train_000001", "label": "entailment", "relevance": "related",
          "rationale": "x"}],
    )
    import_annotations(split, annotations)
    export_split(split)
    with (split / "train.csv").open(encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))
    row["hypothesis"] = "Подмененный текст"
    review = tmp_path / "review.csv"
    with review.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ValueError, match="immutable hypothesis"):
        import_review_csv(split, review)

    registry = json.loads((split / "registry.json").read_text(encoding="utf-8"))
    external = tmp_path / "external.txt"
    external.write_text("external", encoding="utf-8")
    link = split / "decisions" / "originals" / "link.txt"
    try:
        link.symlink_to(external)
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows account lacks the privilege to create symlink fixtures")
        raise
    registry["rows"]["source-1:hypothesis:unit"]["decision_text_path"] = (
        "decisions/originals/link.txt"
    )
    (split / "registry.json").write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(ValueError, match="symlink"):
        export_split(split)


def test_unlabeled_export_includes_pending_active_rows_and_blanks_labels(tmp_path: Path) -> None:
    split = _bundle(tmp_path)
    register_rows(split, [_prepared(1)])
    report = export_split(split)

    assert report["pairs"] == 0
    with (split / "pairs_unlabeled.csv").open(encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["row_id"] == "train_000001"
    assert row["label"] == row["error_type"] == ""
    with pytest.raises(ValueError, match="without human approval"):
        export_split(split, released=True)


def test_needs_review_keeps_uncertain_unmapped_type_pending(tmp_path: Path) -> None:
    split = _bundle(tmp_path)
    register_rows(split, [_prepared(1)])
    annotations = tmp_path / "annotations.jsonl"
    _write_jsonl(
        annotations,
        [{
            "row_id": "train_000001",
            "label": "contradiction",
            "relevance": "related",
            "error_type": "natural_unmapped_contradiction",
            "rationale": "Противоречие вероятно, но тип отсутствует в таблице.",
            "status": "needs_review",
        }],
    )

    assert import_annotations(split, annotations) == {"pending": 1}
    registry = json.loads((split / "registry.json").read_text(encoding="utf-8"))
    row = registry["rows"]["source-1:hypothesis:unit"]
    assert row["status"] == "pending"
    assert row["error_type"] == ""
    assert row["annotation_history"][-1]["submitted_status"] == "needs_review"
    assert row["annotation_history"][-1]["error_type"] == "natural_unmapped_contradiction"


def test_pending_row_from_draft_snapshot_can_be_deleted_safely(tmp_path: Path) -> None:
    split = _bundle(tmp_path)
    register_rows(split, [_prepared(1), _prepared(2)])
    annotations = tmp_path / "annotations.jsonl"
    _write_jsonl(
        annotations,
        [
            {"row_id": "train_000001", "label": "entailment", "relevance": "related",
             "rationale": "x"},
            {"row_id": "train_000002", "label": "entailment", "relevance": "related",
             "rationale": "y"},
        ],
    )
    import_annotations(split, annotations)
    export_split(split)
    with (split / "train.csv").open(encoding="utf-8-sig", newline="") as stream:
        first = next(csv.DictReader(stream))
    uncertain = tmp_path / "uncertain.jsonl"
    _write_jsonl(
        uncertain,
        [{"row_id": "train_000002", "label": "not mentioned", "relevance": "related",
          "rationale": "Недостаточно контекста.", "status": "needs_review"}],
    )
    import_annotations(split, uncertain)
    review = tmp_path / "review.csv"
    with review.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(first)

    assert import_review_csv(split, review) == {"approved": 1, "excluded": 1}
    registry = json.loads((split / "registry.json").read_text(encoding="utf-8"))
    assert registry["rows"]["source-2:hypothesis:unit"]["status"] == "excluded"


def test_proposed_semantic_relation_labels_must_be_related_and_reasoned(tmp_path: Path) -> None:
    split = _bundle(tmp_path)
    register_rows(split, [_prepared(1)])
    annotations = tmp_path / "annotations.jsonl"
    _write_jsonl(
        annotations,
        [{"row_id": "train_000001", "label": "entailment", "relevance": "unrelated",
          "rationale": "x"}],
    )
    with pytest.raises(ValueError, match="must be related"):
        import_annotations(split, annotations)

    _write_jsonl(
        annotations,
        [{"row_id": "train_000001", "label": "entailment", "relevance": "related",
          "rationale": ""}],
    )
    with pytest.raises(ValueError, match="needs a rationale"):
        import_annotations(split, annotations)


def test_export_hashes_shared_text_once_and_checks_preparation_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split = _bundle(tmp_path)
    first, second = _prepared(1), _prepared(2)
    second["decision_text_path"] = first["decision_text_path"]
    second["source_filename"] = first["source_filename"]
    register_rows(split, [first, second])
    annotations = tmp_path / "annotations.jsonl"
    _write_jsonl(
        annotations,
        [
            {"row_id": "train_000001", "label": "entailment", "relevance": "related",
             "rationale": "x"},
            {"row_id": "train_000002", "label": "not mentioned", "relevance": "unrelated",
             "rationale": "y"},
        ],
    )
    import_annotations(split, annotations)
    calls = 0
    original = review._portable_document

    def counting_portable(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(review, "_portable_document", counting_portable)
    export_split(split)
    assert calls == 1

    registry = json.loads((split / "registry.json").read_text(encoding="utf-8"))
    registry["rows"]["source-1:hypothesis:unit"]["metadata"]["text_sha256"] = "0" * 64
    (split / "registry.json").write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from prepared row"):
        export_split(split)


def test_review_correction_to_entailment_forces_related_relevance(tmp_path: Path) -> None:
    split = _bundle(tmp_path)
    register_rows(split, [_prepared(1)])
    annotations = tmp_path / "annotations.jsonl"
    _write_jsonl(
        annotations,
        [{"row_id": "train_000001", "label": "not mentioned",
          "relevance": "unrelated", "rationale": "x"}],
    )
    import_annotations(split, annotations)
    export_split(split)
    with (split / "train.csv").open(encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))
    row["label"] = "entailment"
    review_csv = tmp_path / "review.csv"
    with review_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(row)

    import_review_csv(split, review_csv)
    registry = json.loads((split / "registry.json").read_text(encoding="utf-8"))
    assert registry["rows"]["source-1:hypothesis:unit"]["relevance"] == "related"


def test_val_report_exposes_document_target_shortage(tmp_path: Path) -> None:
    split = _bundle(tmp_path, "val")
    register_rows(split, [_prepared(1)])
    annotations = tmp_path / "annotations.jsonl"
    _write_jsonl(
        annotations,
        [{"row_id": "val_000001", "label": "contradiction", "relevance": "related",
          "error_type": "xlsx_row_32", "rationale": "x"}],
    )
    import_annotations(split, annotations)

    report = export_split(split)
    assert report["contradiction_document_target"] == {
        "target": 120,
        "actual": 1,
        "shortage": 119,
        "excess": 0,
    }
