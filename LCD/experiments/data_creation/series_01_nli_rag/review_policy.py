"""Deterministic human review scopes for the train sample and full evaluation splits."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from pathlib import Path


def _signature(row):
    return [row.get(k, "") for k in ("label", "relevance", "error_type")]


def _requires_review(row):
    variants = row.get("metadata", {}).get("variant_ids", [row.get("variant_id", "original")])
    return (any(v != "original" for v in variants)
            or any(v.get("verdict") != "accept" for v in row.get("verifications", []))
            or any(a.get("submitted_status") == "needs_review" or a.get("status") == "pending"
                   for a in row.get("annotation_history", [])))


def plan_human_review(split_dir: str | Path, *, fraction=0.1, seed=42) -> dict:
    """Write a sampled review CSV and complete uncertain-row worklist.

    Train includes every synthetic/flagged row and a seeded sample from each
    premise-article/label/error-type stratum. Val/test include every active row.
    The 10% train default is configurable and always selects at least one row
    per populated stratum. Unlabeled uncertain rows require human JSONL review.
    """
    from .review import CSV_COLUMNS
    from .workflow import digest, read_json, write_json, write_jsonl
    split_dir = Path(split_dir)
    if not 0 < fraction <= 1:
        raise ValueError("Review fraction must be in (0, 1]")
    registry = read_json(split_dir / "registry.json")
    rows = [r for r in registry["rows"].values() if r.get("active", True)
            and r.get("status") != "excluded"]
    required = {r["row_id"] for r in rows if _requires_review(r)}
    if split_dir.name != "train":
        required.update(r["row_id"] for r in rows)
    else:
        strata = defaultdict(list)
        for row in rows:
            if row.get("status") in {"proposed", "approved"}:
                strata[(row["article_number"], row["label"], row["error_type"])].append(row)
        for members in strata.values():
            ordered = sorted(members, key=lambda r: digest([seed, r["row_id"]]))
            required.update(r["row_id"] for r in ordered[:max(1, math.ceil(len(members) * fraction))])
    population = {r["row_id"]: _signature(r) for r in rows}
    plan = {"split": split_dir.name, "fraction": fraction, "seed": seed,
            "population": population, "required_row_ids": sorted(required)}
    write_json(split_dir / "human_review_plan.json", plan)
    write_json(split_dir / "human_review_plans" / (digest(plan) + ".json"), plan)
    selected = [r for r in rows if r["row_id"] in required and r.get("status") in {"proposed", "approved"}]
    with (split_dir / "human_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in CSV_COLUMNS} for r in selected)
    write_json(split_dir / "human_review_snapshot.json", {"row_ids": [r["row_id"] for r in selected]})
    uncertain = [r for r in rows if r.get("status") not in {"proposed", "approved"}]
    write_jsonl(split_dir / "human_review_pending.jsonl", uncertain)
    return {"required": len(required), "sampled_csv_rows": len(selected),
            "uncertain_jsonl_rows": len(uncertain), "population": len(rows)}


def check_release_review(split_dir: Path, active: list[dict]) -> None:
    """Require full evaluation/synthetic review and a completed train audit."""
    unapproved = [r for r in active if r.get("status") != "approved"]
    if not unapproved:
        return
    if split_dir.name != "train" or any(r.get("status") != "proposed" for r in unapproved):
        raise ValueError(f"Release contains {len(unapproved)} active rows without human approval")
    path = split_dir / "human_review_plan.json"
    if not path.is_file():
        raise ValueError("Train rows remain without human approval; create and complete a stratified review plan")
    plan = json.loads(path.read_text(encoding="utf-8"))
    required = set(plan["required_row_ids"])
    for row in active:
        row_id = row["row_id"]
        if row_id not in plan["population"]:
            raise ValueError("Training review plan is stale; new rows need a refreshed audit")
        if row.get("status") != "approved":
            if plan["population"][row_id] != _signature(row):
                raise ValueError("Training labels changed after review planning")
            if row_id in required or _requires_review(row):
                raise ValueError(f"Required train row {row_id} remains without human approval")
