"""Audited extraction refreshes that preserve every allocated row identity."""
from __future__ import annotations

from pathlib import Path


def refresh_filter(bundle: str | Path, *, split=None) -> dict:
    """Exclude previously prepared filter rejects without deleting registry history.

    Run after preparation has stopped, before distributing a new review export.
    Surviving IDs, labels and document links remain unchanged. This operation
    only removes irrelevant hypotheses; extraction changes adding hypotheses
    require a new preparation run/bundle rather than inventing missing pairs.
    """
    from . import preparation
    from .preparation import extract_hypotheses
    from .workflow import (SPLITS, file_hash, pair_identity, portable_path, read_json, write_json)
    from .review import export_split
    bundle = Path(bundle)
    if split is not None and split not in SPLITS:
        raise ValueError("Unknown split")
    report = {}
    for name in ((split,) if split else SPLITS):
        split_dir = bundle / name
        manifest = read_json(split_dir / "manifest.json")
        registry = read_json(split_dir / "registry.json")
        accepted = {}
        for variant in manifest["variants"]:
            path = portable_path(split_dir, variant["decision_text_path"])
            if file_hash(path) != variant["text_sha256"]:
                raise ValueError("Cannot refresh extraction from modified source text")
            hypotheses, _ = extract_hypotheses(path.read_text(encoding="utf-8"))
            accepted[(variant["source_id"], variant["variant_id"])] = {
                (h["start"], h["end"], h["text"]) for h in hypotheses}
        excluded, seen_pairs = [], {}
        for row in registry["rows"].values():
            if row.get("status") == "excluded":
                continue
            span = row["metadata"]["hypothesis_span"]
            identity = (span[0], span[1], row["hypothesis"])
            exact_pair = pair_identity(row)
            duplicate = seen_pairs.get(exact_pair)
            irrelevant = identity not in accepted[(row["source_id"], row["variant_id"])]
            if irrelevant or duplicate:
                row["status"] = "excluded"
                row.setdefault("annotation_history", []).append({
                    "actor": "extraction_filter", "status": "excluded",
                    "reason": ("irrelevant_hypothesis_after_filter_refresh" if irrelevant
                               else "exact_duplicate_pair"),
                    "filter_sha256": file_hash(Path(preparation.__file__))})
                excluded.append(row["row_id"])
                if duplicate:
                    row["metadata"]["duplicate_of"] = duplicate
            elif row.get("active", True):
                seen_pairs[exact_pair] = row["row_id"]
        # Older checkpoints used text-only deduplication. Re-open ambiguous
        # repeated IDs once so equal provision text at different legal
        # citations is recovered without changing any existing row ID.
        reopened = []
        for key, checkpoint in list(manifest.get("prepared", {}).items()):
            ids = checkpoint.get("row_ids", [])
            if checkpoint.get("identity_version", 1) < 2 and len(set(ids)) < len(ids):
                manifest.setdefault("preparation_repairs", {})[key] = {
                    "reason": "Rebuild with citation-aware pair identities", "previous_row_ids": ids}
                del manifest["prepared"][key]
                reopened.append(key)
        write_json(split_dir / "registry.json", registry)
        manifest.setdefault("filter_refreshes", []).append({
            "filter_sha256": file_hash(Path(preparation.__file__)), "excluded_ids": excluded})
        write_json(split_dir / "manifest.json", manifest)
        export_split(split_dir)
        report[name] = {"excluded_pairs": len(excluded), "row_ids": excluded,
                        "reopened_variants": reopened}
    enrich_preparation_errors(bundle)
    return report


def enrich_preparation_errors(bundle: str | Path) -> list[dict]:
    """Resolve preparation failures to readable hypotheses and portable source paths."""
    from LCD.shared.retrieval import extract_citations
    from .workflow import digest, read_json, write_json, write_jsonl

    bundle = Path(bundle)
    path = bundle / "preparation_errors.json"
    if not path.exists():
        return []
    errors = read_json(path)
    manifests = {}
    for item in errors:
        split = item.get("split")
        if split not in {"train", "val", "test"}:
            continue
        if split not in manifests:
            manifests[split] = read_json(bundle / split / "manifest.json")
        variant_id = item.get("variant_id", "original")
        variant = next(v for v in manifests[split]["variants"]
                       if v["source_id"] == item["source_id"] and v["variant_id"] == variant_id)
        item.update(source_filename=variant["source_filename"],
                    decision_text_path=variant["decision_text_path"])
        variant_key = item["source_id"] + ":" + variant_id
        audit_path = bundle / split / "extraction" / (digest(variant_key)[:24] + ".json")
        if not item.get("hypothesis") and audit_path.is_file():
            for record in read_json(audit_path)["audit"]:
                hyp_id = "hyp_" + digest([item["source_id"], record["start"], record["text"]])[:20]
                if hyp_id == item.get("hypothesis_id"):
                    item["hypothesis"] = record["text"]
                    break
        item["citations"] = [vars(c) for c in extract_citations(item.get("hypothesis", ""))]
        item["review_id"] = "preparation_" + digest([split, variant_key, item.get("hypothesis_id")])[:20]
    write_json(path, errors)
    write_jsonl(bundle / "preparation_review.jsonl", errors)
    return errors


def refresh_rule_metadata(bundle: str | Path, expert_table: str | Path) -> dict:
    """Recover auxiliary expert columns only from the original hash-pinned workbook."""
    from .preparation import read_expert_rules
    from .workflow import SPLITS, file_hash, read_json, write_json

    bundle, expert_table = Path(bundle), Path(expert_table)
    manifest = read_json(bundle / "manifest.json")
    if file_hash(expert_table) != manifest["expert_table_sha256"]:
        raise ValueError("Expert workbook changed; select a new bundle rather than changing frozen rules")
    previous = read_json(bundle / "rules.json")
    previous = previous.get("rules") if isinstance(previous, dict) else previous
    rules = read_expert_rules(expert_table)
    if len(previous) != len(rules) or any(
        any(rule.get(k) != v for k, v in old.items() if k != "additional_fields")
        for old, rule in zip(previous, rules)
    ):
        raise ValueError("Core expert rules changed during metadata refresh")
    write_json(bundle / "rules.json", rules)
    for split in SPLITS:
        write_json(bundle / split / "rules.json", rules)
    report = {"rules": len(rules), "expert_table_sha256": file_hash(expert_table),
              "additional_columns": sorted({key for rule in rules for key in rule["additional_fields"]})}
    write_json(bundle / "rule_metadata.json", report)
    return report
