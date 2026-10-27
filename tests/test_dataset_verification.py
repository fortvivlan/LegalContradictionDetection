import json
from pathlib import Path

import pytest

from LCD.experiments.data_creation.series_01_nli_rag.verification import (
    import_verifications,
)


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def _split(tmp_path: Path) -> Path:
    split = tmp_path / "bundle" / "train"
    _write(
        split / "registry.json",
        {
            "next_id": 3,
            "rows": {
                "key-1": {
                    "row_id": "train_000001",
                    "source_id": "doc-1",
                    "variant_id": "syn-1",
                    "status": "proposed",
                    "label": "entailment",
                },
                "key-2": {
                    "row_id": "train_000002",
                    "source_id": "doc-2",
                    "variant_id": "original",
                    "status": "proposed",
                    "label": "not mentioned",
                },
            },
        },
    )
    _write(
        split / "manifest.json",
        {
            "split": "train",
            "variants": [
                {"source_id": "doc-1", "variant_id": "original"},
                {"source_id": "doc-1", "variant_id": "syn-1"},
            ],
        },
    )
    return split


def test_independent_accept_records_evidence_without_human_approval(tmp_path: Path) -> None:
    split = _split(tmp_path)
    path = tmp_path / "accept.jsonl"
    _jsonl(
        path,
        [
            {
                "object_type": "annotation",
                "object_id": "train_000001",
                "verdict": "accept",
                "reasons": ["Вся составная гипотеза подтверждается нормой."],
            },
            {
                "object_type": "variant",
                "object_id": "syn-1",
                "verdict": "accept",
                "reasons": ["Смещения и единственность сценария проверены."],
            },
        ],
    )

    assert import_verifications(split, path) == {"verified": 2, "disputed": 0}
    registry = json.loads((split / "registry.json").read_text(encoding="utf-8"))
    row = registry["rows"]["key-1"]
    assert row["status"] == "proposed"
    assert row["status"] != "approved"
    assert row["verifications"][0]["verdict"] == "accept"
    manifest = json.loads((split / "manifest.json").read_text(encoding="utf-8"))
    variant = next(value for value in manifest["variants"] if value["variant_id"] == "syn-1")
    assert variant["verifications"][0]["verdict"] == "accept"
    assert variant.get("verification_status") != "approved"


@pytest.mark.parametrize("verdict", ["reject", "needs_review"])
def test_disputed_annotation_returns_to_pending_review(
    tmp_path: Path, verdict: str
) -> None:
    split = _split(tmp_path)
    path = tmp_path / "disputed.jsonl"
    _jsonl(
        path,
        [{
            "object_type": "annotation",
            "object_id": "train_000002",
            "verdict": verdict,
            "reasons": ["Один существенный элемент гипотезы не подтверждён."],
        }],
    )

    assert import_verifications(split, path) == {"verified": 1, "disputed": 1}
    registry = json.loads((split / "registry.json").read_text(encoding="utf-8"))
    row = registry["rows"]["key-2"]
    assert row["status"] == "pending"
    assert row["verifications"][-1]["verdict"] == verdict


def test_invalid_batch_is_atomic_and_changes_neither_artifact(tmp_path: Path) -> None:
    split = _split(tmp_path)
    path = tmp_path / "invalid.jsonl"
    _jsonl(
        path,
        [
            {
                "object_type": "annotation",
                "object_id": "train_000001",
                "verdict": "accept",
                "reasons": ["Проверено."],
            },
            {
                "object_type": "annotation",
                "object_id": "train_000002",
                "verdict": "maybe",
                "reasons": ["Недопустимый вердикт."],
            },
        ],
    )
    registry_before = (split / "registry.json").read_bytes()
    manifest_before = (split / "manifest.json").read_bytes()

    with pytest.raises(ValueError, match="invalid verification verdict"):
        import_verifications(split, path)

    assert (split / "registry.json").read_bytes() == registry_before
    assert (split / "manifest.json").read_bytes() == manifest_before


def test_unknown_variant_is_rejected_without_changes(tmp_path: Path) -> None:
    split = _split(tmp_path)
    path = tmp_path / "unknown.jsonl"
    _jsonl(
        path,
        [{
            "object_type": "variant",
            "object_id": "syn-unknown",
            "verdict": "reject",
            "reasons": ["Объект отсутствует."],
        }],
    )
    registry_before = (split / "registry.json").read_bytes()
    manifest_before = (split / "manifest.json").read_bytes()

    with pytest.raises(ValueError, match="Unknown synthetic variant"):
        import_verifications(split, path)

    assert (split / "registry.json").read_bytes() == registry_before
    assert (split / "manifest.json").read_bytes() == manifest_before
