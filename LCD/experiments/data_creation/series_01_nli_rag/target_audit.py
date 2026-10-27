"""Audit whether synthetic targets survived retrieval without changing candidates.

The audit maps an original target hypothesis through the proposal's source
offset edits, finds the corresponding hypothesis in the prepared variant, and
checks only that hypothesis's already-retrieved rows. It never injects a gold
provision and never assigns or changes an NLI label.
"""
from __future__ import annotations

from pathlib import Path

from .preparation import extract_hypotheses
from .workflow import (SPLITS, digest, file_hash, portable_path, read_json,
                       write_json, write_jsonl)


def _mapped_span(start: int, end: int, edits: list[dict]) -> tuple[int, int]:
    """Map one half-open original interval through nonoverlapping source edits.

    Insertions exactly at ``start`` are treated as preceding the original text;
    insertions exactly at ``end`` are treated as following it. The unchanged
    original target therefore retains stable half-open boundary semantics while
    replacements inside it expand or contract its mapped interval.
    """
    mapped_start, mapped_end = start, end
    previous_end = -1
    for edit in sorted(edits, key=lambda value: (value["start"], value["end"])):
        edit_start, edit_end = edit["start"], edit["end"]
        if (type(edit_start) is not int or type(edit_end) is not int
                or edit_start < previous_end or edit_start > edit_end):
            raise ValueError("Variant contains invalid or overlapping edit offsets")
        previous_end = edit_end
        delta = len(edit["after"]) - (edit_end - edit_start)
        if edit_end <= start:
            mapped_start += delta
        if edit_start < end:
            mapped_end += delta
    if mapped_start > mapped_end:
        raise ValueError("Variant edits invert the mapped target span")
    return mapped_start, mapped_end


def _apply_edits(text: str, edits: list[dict]) -> str:
    ordered = sorted(edits, key=lambda value: (value["start"], value["end"]))
    previous_end = -1
    for edit in ordered:
        start, end = edit["start"], edit["end"]
        if (type(start) is not int or type(end) is not int or start < previous_end
                or not 0 <= start <= end <= len(text)
                or text[start:end] != edit.get("before")
                or not isinstance(edit.get("after"), str)):
            raise ValueError("Variant edit does not match the immutable original text")
        previous_end = end
    result = text
    for edit in reversed(ordered):
        result = result[:edit["start"]] + edit["after"] + result[edit["end"]:]
    return result


def _original_target(source_id: str, text: str, target_id: str) -> dict:
    matches = []
    for hypothesis in extract_hypotheses(text)[0]:
        hypothesis_id = "hyp_" + digest(
            [source_id, hypothesis["start"], hypothesis["text"]]
        )[:20]
        if hypothesis_id == target_id:
            matches.append(hypothesis)
    if len(matches) != 1:
        raise ValueError(
            f"Original target hypothesis {target_id!r} was not found exactly once for {source_id}"
        )
    return matches[0]


def _overlaps(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return first[0] < second[1] and second[0] < first[1]


def _audit_split(bundle: Path, split: str) -> tuple[dict, list[dict]]:
    split_dir = bundle / split
    manifest_path, registry_path = split_dir / "manifest.json", split_dir / "registry.json"
    manifest, registry = read_json(manifest_path), read_json(registry_path)
    by_row_id = {row["row_id"]: row for row in registry["rows"].values()}
    originals = {
        variant["source_id"]: variant for variant in manifest.get("variants", [])
        if variant.get("variant_id") == "original"
    }
    audits, review = [], []
    skipped_unprepared = 0

    for variant in manifest.get("variants", []):
        variant_id = variant.get("variant_id", "")
        if variant_id == "original":
            continue
        variant_key = f"{variant.get('source_id', '')}:{variant_id}"
        checkpoint = manifest.get("prepared", {}).get(variant_key)
        if checkpoint is None:
            skipped_unprepared += 1
            continue
        source_id = variant.get("source_id")
        original = originals.get(source_id)
        if original is None:
            raise ValueError(f"Synthetic variant {variant_id} has no original source")
        target_id = variant.get("target_hypothesis_id")
        if not isinstance(target_id, str) or not target_id:
            raise ValueError(f"Synthetic variant {variant_id} has no original target hypothesis ID")
        gold_ids = variant.get("target_unit_ids")
        if (not isinstance(gold_ids, list) or not gold_ids
                or any(not isinstance(value, str) or not value for value in gold_ids)):
            raise ValueError(f"Synthetic variant {variant_id} has no valid target unit IDs")
        gold_ids = list(dict.fromkeys(gold_ids))

        original_path = portable_path(split_dir, original["decision_text_path"])
        variant_path = portable_path(split_dir, variant["decision_text_path"])
        if file_hash(original_path) != original["text_sha256"]:
            raise ValueError(f"Original decision text hash differs for {source_id}")
        if file_hash(variant_path) != variant["text_sha256"]:
            raise ValueError(f"Synthetic decision text hash differs for {variant_id}")
        original_text = original_path.read_text(encoding="utf-8")
        variant_text = variant_path.read_text(encoding="utf-8")
        edits = variant.get("edits")
        if not isinstance(edits, list) or not edits:
            raise ValueError(f"Synthetic variant {variant_id} has no source edits")
        if _apply_edits(original_text, edits) != variant_text:
            raise ValueError(f"Synthetic decision {variant_id} is not reproduced by its edits")

        original_hypothesis = _original_target(source_id, original_text, target_id)
        mapped_span = _mapped_span(
            original_hypothesis["start"], original_hypothesis["end"], edits
        )
        extraction_path = split_dir / "extraction" / (digest(variant_key)[:24] + ".json")
        if not extraction_path.is_file():
            raise FileNotFoundError(f"Prepared extraction is missing: {extraction_path}")
        extraction = read_json(extraction_path)
        extracted_variant = extraction.get("variant", {})
        if (extracted_variant.get("variant_id") != variant_id
                or extracted_variant.get("text_sha256") != variant["text_sha256"]):
            raise ValueError(f"Prepared extraction metadata differs for {variant_id}")
        target_hypotheses = [
            hypothesis for hypothesis in extraction.get("hypotheses", [])
            if _overlaps(mapped_span, (hypothesis["start"], hypothesis["end"]))
        ]
        if not target_hypotheses:
            raise ValueError(f"Mapped target span does not overlap a hypothesis for {variant_id}")
        target_texts = {hypothesis["text"] for hypothesis in target_hypotheses}

        checkpoint_ids = list(dict.fromkeys(checkpoint.get("row_ids", [])))
        unknown_ids = sorted(set(checkpoint_ids) - by_row_id.keys())
        if unknown_ids:
            raise ValueError(f"Prepared checkpoint contains unknown row IDs: {unknown_ids}")
        target_rows = [
            by_row_id[row_id] for row_id in checkpoint_ids
            if by_row_id[row_id].get("hypothesis") in target_texts
        ]
        retrieved_ids = sorted({row.get("unit_id", "") for row in target_rows if row.get("unit_id")})
        retrieved_gold = [unit_id for unit_id in gold_ids if unit_id in retrieved_ids]
        missing = [unit_id for unit_id in gold_ids if unit_id not in retrieved_ids]
        audit = {
            "split": split,
            "source_id": source_id,
            "source_filename": variant.get("source_filename", ""),
            "variant_id": variant_id,
            "decision_text_path": variant["decision_text_path"],
            "rule_id": variant.get("rule_id", ""),
            "original_target_hypothesis_id": target_id,
            "original_target_span": [original_hypothesis["start"], original_hypothesis["end"]],
            "mapped_target_span": list(mapped_span),
            "target_hypotheses": [
                {"text": value["text"], "start": value["start"], "end": value["end"]}
                for value in target_hypotheses
            ],
            "target_row_ids": [row["row_id"] for row in target_rows],
            "target_unit_ids": gold_ids,
            "retrieved_target_unit_ids": retrieved_gold,
            "missing_target_unit_ids": missing,
            "retrieved_unit_ids": retrieved_ids,
            "status": "missing_target_units" if missing else "target_retrieved",
        }
        variant["retrieval_audit"] = audit
        audits.append(audit)

        # A canonical train row may represent identical text from several
        # variants. Keep one entry per variant instead of overwriting its audit.
        for row in registry["rows"].values():
            metadata = row.get("metadata", {})
            existing = metadata.get("target_retrieval_audits", [])
            if isinstance(existing, list):
                retained = [
                    value for value in existing if value.get("variant_id") != variant_id
                ]
                if retained:
                    metadata["target_retrieval_audits"] = retained
                else:
                    metadata.pop("target_retrieval_audits", None)
        row_audit = {
            key: audit[key] for key in (
                "variant_id", "original_target_hypothesis_id", "target_unit_ids",
                "retrieved_target_unit_ids", "missing_target_unit_ids", "status"
            )
        }
        for row in target_rows:
            row.setdefault("metadata", {}).setdefault("target_retrieval_audits", []).append(row_audit)

        if missing:
            review.append({
                "review_id": "target_retrieval_" + digest([split, variant_id, missing])[:20],
                "issue_type": "missing_target_provision",
                "audit_scope": "synthetic_target_retrieval",
                "split": split,
                "source_id": source_id,
                "source_filename": variant.get("source_filename", ""),
                "variant_id": variant_id,
                "decision_text_path": variant["decision_text_path"],
                "rule_id": variant.get("rule_id", ""),
                "original_target_hypothesis_id": target_id,
                "hypothesis_text": "\n".join(value["text"] for value in target_hypotheses),
                "target_unit_ids": gold_ids,
                "retrieved_target_unit_ids": retrieved_gold,
                "missing_target_unit_ids": missing,
                "note": (
                    "Gold target was not present among this edited hypothesis's retrieved rows; "
                    "the provision was not injected. Preparation failures and broad-citation "
                    "overflow remain in their separate audit artifacts."
                ),
            })

    write_json(manifest_path, manifest)
    write_json(registry_path, registry)
    report = {
        "split": split,
        "audited_variants": len(audits),
        "target_retrieved": sum(not value["missing_target_unit_ids"] for value in audits),
        "missing_target_variants": sum(bool(value["missing_target_unit_ids"]) for value in audits),
        "missing_target_units": sum(len(value["missing_target_unit_ids"]) for value in audits),
        "skipped_unprepared_variants": skipped_unprepared,
        "audits": audits,
    }
    write_json(split_dir / "target_retrieval_audit.json", report)
    return report, review


def audit_variant_targets(bundle: str | Path, *, split: str | None = None) -> dict:
    """Audit prepared synthetic variants and write portable missing-target review tasks."""
    bundle = Path(bundle)
    if split is not None and split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    selected = (split,) if split else SPLITS
    reports = {}
    for name in selected:
        reports[name], _ = _audit_split(bundle, name)

    review = []
    for name in SPLITS:
        path = bundle / name / "target_retrieval_audit.json"
        if not path.is_file():
            continue
        for audit in read_json(path).get("audits", []):
            missing = audit.get("missing_target_unit_ids", [])
            if not missing:
                continue
            review.append({
                "review_id": "target_retrieval_" + digest([name, audit["variant_id"], missing])[:20],
                "issue_type": "missing_target_provision",
                "audit_scope": "synthetic_target_retrieval",
                "split": name,
                "source_id": audit["source_id"],
                "source_filename": audit["source_filename"],
                "variant_id": audit["variant_id"],
                "decision_text_path": audit["decision_text_path"],
                "rule_id": audit["rule_id"],
                "original_target_hypothesis_id": audit["original_target_hypothesis_id"],
                "hypothesis_text": "\n".join(
                    value["text"] for value in audit.get("target_hypotheses", [])
                ),
                "target_unit_ids": audit["target_unit_ids"],
                "retrieved_target_unit_ids": audit["retrieved_target_unit_ids"],
                "missing_target_unit_ids": missing,
                "note": (
                    "Gold target was not present among this edited hypothesis's retrieved rows; "
                    "the provision was not injected. Preparation failures and broad-citation "
                    "overflow remain in their separate audit artifacts."
                ),
            })
    review.sort(key=lambda value: (value["split"], value["variant_id"]))
    write_jsonl(bundle / "target_retrieval_review.jsonl", review)
    return {
        "splits": reports,
        "audited_variants": sum(value["audited_variants"] for value in reports.values()),
        "target_retrieved": sum(value["target_retrieved"] for value in reports.values()),
        "missing_target_variants": sum(
            value["missing_target_variants"] for value in reports.values()
        ),
        "missing_target_units": sum(value["missing_target_units"] for value in reports.values()),
        "review_path": str(bundle / "target_retrieval_review.jsonl"),
    }
