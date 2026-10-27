"""Import independent subagent verification without granting human approval."""
from __future__ import annotations

import json
from pathlib import Path


def import_verifications(split_dir: str | Path, path: str | Path) -> dict:
    """Keep verification history; disputed proposed labels return to pending review."""
    from .workflow import read_json, write_json
    split_dir = Path(split_dir)
    registry = read_json(split_dir / "registry.json")
    rows = {r["row_id"]: r for r in registry["rows"].values()}
    manifest = read_json(split_dir / "manifest.json")
    variants = {v["variant_id"]: v for v in manifest["variants"] if v["variant_id"] != "original"}
    submissions = [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
                   if line.strip()]
    seen = set()
    for value in submissions:
        key = (value.get("object_type"), value.get("object_id"))
        if key in seen or value.get("verdict") not in {"accept", "reject", "needs_review"}:
            raise ValueError("Duplicate object or invalid verification verdict")
        seen.add(key)
        if not isinstance(value.get("reasons"), list) or not value["reasons"]:
            raise ValueError("Verification needs explicit evidence/reasons")
        if key[0] == "annotation":
            if key[1] not in rows or rows[key[1]].get("status") not in {"proposed", "pending", "excluded"}:
                raise ValueError("Only existing unapproved annotations may be verified")
        elif key[0] == "variant":
            if key[1] not in variants:
                raise ValueError("Unknown synthetic variant")
        else:
            raise ValueError("Unknown verification object type")
    for value in submissions:
        if value["object_type"] == "annotation":
            row = rows[value["object_id"]]
            row.setdefault("verifications", []).append(value)
            if value["verdict"] != "accept" and row.get("status") != "excluded":
                row["status"] = "pending"
        else:
            variant = variants[value["object_id"]]
            variant.setdefault("verifications", []).append(value)
            if value["verdict"] != "accept":
                variant["verification_status"] = "needs_review"
                for row in rows.values():
                    if row["variant_id"] == value["object_id"] and row.get("status") == "proposed":
                        row["status"] = "pending"
    write_json(split_dir / "registry.json", registry)
    write_json(split_dir / "manifest.json", manifest)
    return {"verified": len(submissions),
            "disputed": sum(v["verdict"] != "accept" for v in submissions)}
