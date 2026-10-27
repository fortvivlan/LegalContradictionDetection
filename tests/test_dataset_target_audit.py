from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from LCD.experiments.data_creation.series_01_nli_rag.preparation import extract_hypotheses
from LCD.experiments.data_creation.series_01_nli_rag.target_audit import audit_variant_targets
from LCD.experiments.data_creation.series_01_nli_rag.workflow import digest


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _hypothesis_id(source_id: str, hypothesis: dict) -> str:
    return "hyp_" + digest([source_id, hypothesis["start"], hypothesis["text"]])[:20]


def _bundle(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    bundle, split_dir = tmp_path / "bundle", tmp_path / "bundle" / "train"
    source_id = "doc_source"
    original_text = (
        "Вводная часть.\nПОСТАНОВИЛ:\n"
        "Назначить штраф 5000 рублей.\n"
        "Постановление может быть обжаловано в течение 10 дней.\n"
    )
    original_hypotheses = extract_hypotheses(original_text)[0]
    fine, appeal = original_hypotheses
    fine_id, appeal_id = _hypothesis_id(source_id, fine), _hypothesis_id(source_id, appeal)
    original_relative = f"decisions/originals/{source_id}.txt"
    original_path = split_dir / original_relative
    original_path.parent.mkdir(parents=True)
    original_path.write_text(original_text, encoding="utf-8", newline="")
    original = {
        "source_id": source_id,
        "variant_id": "original",
        "decision_text_path": original_relative,
        "text_sha256": _sha(original_text),
        "source_filename": "main/18.8/source.docx",
        "active": True,
    }

    def variant(name: str, target_id: str, target_units: list[str], edits: list[dict]) -> dict:
        text = original_text
        for edit in reversed(sorted(edits, key=lambda value: value["start"])):
            text = text[:edit["start"]] + edit["after"] + text[edit["end"]:]
        relative = f"decisions/variants/{name}.txt"
        path = split_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="")
        return {
            "source_id": source_id,
            "variant_id": name,
            "decision_text_path": relative,
            "text_sha256": _sha(text),
            "source_filename": original["source_filename"],
            "target_hypothesis_id": target_id,
            "target_unit_ids": target_units,
            "rule_id": "rule",
            "edits": edits,
            "active": True,
        }

    fine_start = original_text.index("5000", fine["start"], fine["end"])
    missing_variant = variant("syn_missing", fine_id, ["unit_fine"], [{
        "start": fine_start, "end": fine_start + 4, "before": "5000", "after": "4000"
    }])
    # An insertion before the operative section shifts the target span while
    # preserving its exact hypothesis text, allowing reuse of a canonical row.
    shifted_variant = variant("syn_shifted", appeal_id, ["unit_appeal"], [{
        "start": 0, "end": 0, "before": "", "after": "Префикс.\n"
    }])
    deadline_start = original_text.index("10", appeal["start"], appeal["end"])
    deadline_variant = variant("syn_deadline", appeal_id, ["unit_appeal"], [{
        "start": deadline_start, "end": deadline_start + 2, "before": "10", "after": "5"
    }])
    variants = [original, missing_variant, shifted_variant, deadline_variant]

    registry_rows = {}

    def row(key: str, row_id: str, hypothesis: str, unit_id: str, variant_id: str,
            path: str, span: list[int]) -> None:
        registry_rows[key] = {
            "row_id": row_id,
            "source_id": source_id,
            "variant_id": variant_id,
            "hypothesis": hypothesis,
            "unit_id": unit_id,
            "decision_text_path": path,
            "metadata": {"hypothesis_span": span, "text_sha256": next(
                value["text_sha256"] for value in variants if value["variant_id"] == variant_id
            )},
            "status": "pending",
            "active": True,
        }

    missing_hypotheses = extract_hypotheses(
        (split_dir / missing_variant["decision_text_path"]).read_text(encoding="utf-8")
    )[0]
    missing_fine, missing_appeal = missing_hypotheses
    row("missing-target", "train_000001", missing_fine["text"], "unit_other",
        "syn_missing", missing_variant["decision_text_path"],
        [missing_fine["start"], missing_fine["end"]])
    # The gold unit exists in the same variant checkpoint, but for another
    # hypothesis. It must not satisfy the fine target audit.
    row("missing-other", "train_000002", missing_appeal["text"], "unit_fine",
        "syn_missing", missing_variant["decision_text_path"],
        [missing_appeal["start"], missing_appeal["end"]])
    # Canonical original row reused by the shifted variant.
    row("canonical-appeal", "train_000003", appeal["text"], "unit_appeal",
        "original", original_relative, [appeal["start"], appeal["end"]])
    deadline_hypotheses = extract_hypotheses(
        (split_dir / deadline_variant["decision_text_path"]).read_text(encoding="utf-8")
    )[0]
    edited_appeal = deadline_hypotheses[1]
    row("deadline-target", "train_000004", edited_appeal["text"], "unit_appeal",
        "syn_deadline", deadline_variant["decision_text_path"],
        [edited_appeal["start"], edited_appeal["end"]])

    prepared = {
        f"{source_id}:syn_missing": {"row_ids": ["train_000001", "train_000002"]},
        f"{source_id}:syn_shifted": {"row_ids": ["train_000003"]},
        f"{source_id}:syn_deadline": {"row_ids": ["train_000004"]},
    }
    manifest = {
        "split": "train",
        "documents": [{"source_id": source_id}],
        "variants": variants,
        "prepared": prepared,
    }
    _write_json(split_dir / "manifest.json", manifest)
    _write_json(split_dir / "registry.json", {"next_id": 5, "rows": registry_rows})
    for synthetic in variants[1:]:
        key = f"{source_id}:{synthetic['variant_id']}"
        text = (split_dir / synthetic["decision_text_path"]).read_text(encoding="utf-8")
        _write_json(
            split_dir / "extraction" / (digest(key)[:24] + ".json"),
            {"variant": synthetic, "hypotheses": extract_hypotheses(text)[0], "audit": []},
        )
    return bundle, {
        "source_id": source_id,
        "fine_id": fine_id,
        "appeal_id": appeal_id,
    }


def test_target_audit_is_hypothesis_scoped_maps_edits_and_is_idempotent(tmp_path: Path) -> None:
    bundle, _ = _bundle(tmp_path)

    first = audit_variant_targets(bundle, split="train")
    second = audit_variant_targets(bundle, split="train")

    assert first["audited_variants"] == second["audited_variants"] == 3
    assert first["target_retrieved"] == 2
    assert first["missing_target_variants"] == 1
    report = json.loads((bundle / "train/target_retrieval_audit.json").read_text())
    audits = {value["variant_id"]: value for value in report["audits"]}
    assert audits["syn_missing"]["missing_target_unit_ids"] == ["unit_fine"]
    assert audits["syn_missing"]["retrieved_unit_ids"] == ["unit_other"]
    assert audits["syn_shifted"]["status"] == "target_retrieved"
    assert audits["syn_shifted"]["mapped_target_span"][0] > audits["syn_shifted"]["original_target_span"][0]
    assert audits["syn_deadline"]["status"] == "target_retrieved"

    registry = json.loads((bundle / "train/registry.json").read_text())
    missing_other = registry["rows"]["missing-other"]["metadata"]
    assert "target_retrieval_audits" not in missing_other
    canonical = registry["rows"]["canonical-appeal"]["metadata"]["target_retrieval_audits"]
    assert [value["variant_id"] for value in canonical] == ["syn_shifted"]
    deadline = registry["rows"]["deadline-target"]["metadata"]["target_retrieval_audits"]
    assert [value["variant_id"] for value in deadline] == ["syn_deadline"]

    review = [json.loads(line) for line in
              (bundle / "target_retrieval_review.jsonl").read_text().splitlines() if line]
    assert len(review) == 1
    assert review[0]["variant_id"] == "syn_missing"
    assert review[0]["hypothesis_text"] == "Назначить штраф 4000 рублей."
    assert "broad-citation" in review[0]["note"]


@pytest.mark.parametrize("failure", ["missing_original", "unknown_target"])
def test_target_audit_fails_clearly_for_invalid_source_or_target(
    tmp_path: Path, failure: str
) -> None:
    bundle, _ = _bundle(tmp_path)
    manifest_path = bundle / "train/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    synthetic = next(value for value in manifest["variants"] if value["variant_id"] == "syn_missing")
    if failure == "missing_original":
        manifest["variants"] = [
            value for value in manifest["variants"] if value["variant_id"] != "original"
        ]
        expected = "no original source"
    else:
        synthetic["target_hypothesis_id"] = "hyp_does_not_exist"
        expected = "was not found exactly once"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=expected):
        audit_variant_targets(bundle, split="train")
