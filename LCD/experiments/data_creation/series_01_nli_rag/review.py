"""Stable review registry and portable CSV exports for the rebuilt datasets.

The JSON registry is the source of truth.  CSV files are disposable review
views: their order has no meaning, and omitting a row records a persistent
exclusion rather than renumbering the remaining examples.
"""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Iterable, Mapping
from ._io import replace_checkpoint


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
IMMUTABLE_REVIEW_COLUMNS = (
    "row_id",
    "hypothesis",
    "premise",
    "article_number",
    "context",
    "source_filename",
    "decision_text_path",
)
LABELS = ("contradiction", "entailment", "not mentioned")
RELEVANCE_LABELS = ("related", "unrelated")
ACTIVE_REVIEW_STATES = ("proposed", "approved")
DETECTION_TARGETS = {
    "contradiction": 0.50,
    "entailment": 0.25,
    "not mentioned": 0.25,
}
RAG_TARGETS = {"related": 0.50, "unrelated": 0.50}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        if path.name == "registry.json":
            stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        else:
            json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    replace_checkpoint(temporary, path)


def _load_registry(split_dir: Path) -> dict:
    path = split_dir / "registry.json"
    if not path.is_file():
        return {"next_id": 1, "rows": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("next_id"), int)
        or value["next_id"] < 1
        or not isinstance(value.get("rows"), dict)
    ):
        raise ValueError(f"Invalid registry: {path}")
    return value


def _row_index(registry: Mapping) -> dict[str, tuple[str, dict]]:
    result: dict[str, tuple[str, dict]] = {}
    for stable_key, row in registry["rows"].items():
        row_id = row.get("row_id")
        if not isinstance(row_id, str) or row_id in result:
            raise ValueError("Registry row IDs must be present and unique")
        result[row_id] = (stable_key, row)
    return result


def _status(row: Mapping) -> str:
    """Read the canonical status, accepting early review_status registries."""
    return str(row.get("status", row.get("review_status", "pending")))


def register_rows(split_dir: str | Path, rows: Iterable[Mapping], *,
                  registry: dict | None = None) -> list[str]:
    """Allocate permanent IDs for prepared rows and return them in input order.

    Every input needs a deterministic ``stable_key`` plus the eight non-ID CSV
    fields. Re-registering identical data is idempotent. An excluded key remains
    excluded and keeps its original ID.
    """
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    split = split_dir.name
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", split):
        raise ValueError(f"Invalid split directory name: {split!r}")
    registry = _load_registry(split_dir) if registry is None else registry
    allocated = []
    required = set(CSV_COLUMNS) - {"row_id"}
    for supplied in rows:
        item = dict(supplied)
        stable_key = str(item.pop("stable_key", "")).strip()
        if not stable_key:
            raise ValueError("Every row requires a non-empty stable_key")
        missing = sorted(required - item.keys())
        if missing:
            raise ValueError(f"Row {stable_key!r} is missing fields: {missing}")
        existing = registry["rows"].get(stable_key)
        if existing is not None:
            for name in IMMUTABLE_REVIEW_COLUMNS[1:]:
                if str(existing.get(name, "")) != str(item.get(name, "")):
                    raise ValueError(
                        f"Prepared identity changed for {stable_key!r}: {name}"
                    )
            allocated.append(existing["row_id"])
            continue
        row_id = f"{split}_{registry['next_id']:06d}"
        registry["next_id"] += 1
        record = {name: str(item.pop(name, "")) for name in CSV_COLUMNS[1:]}
        record.update(item)
        record.update(
            {
                "row_id": row_id,
                "status": record.get("status", record.pop("review_status", "pending")),
                "annotation_history": list(record.get("annotation_history", [])),
                "created_at_utc": _now(),
            }
        )
        registry["rows"][stable_key] = record
        allocated.append(row_id)
    _row_index(registry)
    _atomic_json(split_dir / "registry.json", registry)
    return allocated


def _read_jsonl(path: Path) -> list[dict]:
    result = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON on {path}:{number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"Annotation on {path}:{number} is not an object")
        result.append(value)
    return result


def _eligible_rules(split_dir: Path) -> tuple[set[str], dict[str, str]]:
    local_rules = split_dir / "rules.json"
    path = local_rules if local_rules.is_file() else split_dir.parent / "rules.json"
    if not path.is_file():
        raise FileNotFoundError(f"Eligible expert rules are missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    rules = value.get("rules") if isinstance(value, dict) else value
    if not isinstance(rules, list):
        raise ValueError(f"Invalid expert rule export: {path}")
    names, identifiers = set(), {}
    for rule in rules:
        if not isinstance(rule, dict) or not rule.get("rule_id") or not rule.get("name"):
            raise ValueError(f"Invalid expert rule in {path}")
        rule_id, name = str(rule["rule_id"]), str(rule["name"])
        identifiers[rule_id] = name
        names.add(name)
    return names, identifiers


def _validate_annotation(
    annotation: Mapping, eligible_names: set[str], rule_ids: Mapping[str, str]
) -> dict:
    result = dict(annotation)
    row_id = str(result.get("row_id", "")).strip()
    if not row_id:
        raise ValueError("Every annotation requires row_id")
    submitted_status = str(result.get("status", "proposed")).strip().lower()
    if submitted_status not in {
        "pending", "needs_review", "proposed", "approved", "excluded"
    }:
        raise ValueError(f"Unknown annotation status for {row_id}: {submitted_status}")
    status = "pending" if submitted_status == "needs_review" else submitted_status
    label = str(result.get("label", "")).strip().lower()
    relevance = str(result.get("relevance", "")).strip().lower()
    submitted_error_type = str(result.get("error_type", "")).strip()
    rule_id = str(result.get("rule_id", "")).strip()
    rationale = str(result.get("rationale", "")).strip()
    if status in (*ACTIVE_REVIEW_STATES, "pending"):
        if label not in LABELS:
            raise ValueError(f"Invalid label for {row_id}: {label!r}")
        if relevance not in RELEVANCE_LABELS:
            raise ValueError(f"Invalid relevance for {row_id}: {relevance!r}")
    if status in ACTIVE_REVIEW_STATES and not rationale:
        raise ValueError(f"Proposed or approved annotation {row_id} needs a rationale")
    if status in ACTIVE_REVIEW_STATES and label in {"contradiction", "entailment"}:
        if relevance != "related":
            raise ValueError(f"{label} annotation {row_id} must be related")
    if submitted_error_type in rule_ids:
        inferred_rule_id = submitted_error_type
        error_type = rule_ids[submitted_error_type]
    elif submitted_error_type in eligible_names:
        inferred_rule_id = next(
            identifier for identifier, name in rule_ids.items()
            if name == submitted_error_type
        )
        error_type = submitted_error_type
    else:
        inferred_rule_id = ""
        error_type = submitted_error_type
    if rule_id and inferred_rule_id and rule_id != inferred_rule_id:
        raise ValueError(f"Rule and error type disagree for {row_id}")
    rule_id = rule_id or inferred_rule_id
    if status in ACTIVE_REVIEW_STATES and label == "contradiction":
        if not error_type or error_type not in eligible_names:
            raise ValueError(f"Contradiction {row_id} needs an eligible error_type")
        if rule_id and rule_ids.get(rule_id) != error_type:
            raise ValueError(f"Rule and error type disagree for {row_id}")
    elif status in ACTIVE_REVIEW_STATES and error_type:
        raise ValueError(f"Non-contradiction {row_id} cannot have an error_type")
    if status in ACTIVE_REVIEW_STATES and rule_id and rule_id not in rule_ids:
        raise ValueError(f"Unknown expert rule for {row_id}: {rule_id}")
    result.update(
        row_id=row_id,
        status=status,
        submitted_status=submitted_status,
        label=label,
        relevance=relevance,
        error_type=error_type,
        rule_id=rule_id,
        rationale=rationale,
    )
    return result


def import_annotations(
    split_dir: str | Path, annotations_path: str | Path, *, human: bool = False,
    skip_excluded: bool = False
) -> dict:
    """Validate JSONL annotations and append them to registry history."""
    split_dir, annotations_path = Path(split_dir), Path(annotations_path)
    registry = _load_registry(split_dir)
    index = _row_index(registry)
    names, rule_ids = _eligible_rules(split_dir)
    annotations = [
        _validate_annotation(value, names, rule_ids)
        for value in _read_jsonl(annotations_path)
    ]
    ids = [value["row_id"] for value in annotations]
    if len(ids) != len(set(ids)):
        raise ValueError("Annotation file contains duplicate row IDs")
    unknown = sorted(set(ids) - index.keys())
    if unknown:
        raise ValueError(f"Annotation file contains unknown row IDs: {unknown}")
    skipped = 0
    if skip_excluded:
        retained = [a for a in annotations if _status(index[a["row_id"]][1]) != "excluded"]
        skipped = len(annotations) - len(retained)
        annotations = retained
    for annotation in annotations:
        row = index[annotation["row_id"]][1]
        if _status(row) == "excluded":
            raise ValueError(f"Excluded row cannot be annotated: {annotation['row_id']}")
        status = annotation["status"]
        if human and status == "proposed":
            status = "approved"
        if not human and status == "approved":
            raise ValueError("Only a human annotation import may approve rows")
        event = {
            **annotation,
            "status": status,
            "actor": "human" if human else "subagent",
            "timestamp_utc": _now(),
        }
        row.setdefault("annotation_history", []).append(event)
        row["status"] = status
        if status in ACTIVE_REVIEW_STATES:
            row["label"] = annotation["label"]
            row["relevance"] = annotation["relevance"]
            row["error_type"] = annotation["error_type"]
            row["rule_id"] = annotation["rule_id"]
            row["annotation_rationale"] = annotation["rationale"]
    _atomic_json(split_dir / "registry.json", registry)
    counts = Counter(value["status"] for value in annotations)
    if skipped:
        counts["skipped_excluded"] = skipped
    return counts


def _portable_document(split_dir: Path, relative_value: str) -> tuple[Path, str]:
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Unsafe decision_text_path: {relative_value!r}")
    candidate = split_dir.joinpath(relative)
    current = split_dir
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Decision text cannot be a symlink: {relative_value}")
    if not candidate.is_file():
        raise FileNotFoundError(f"Decision text does not exist: {candidate}")
    resolved_root, resolved = split_dir.resolve(), candidate.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"Decision text escapes split directory: {relative_value}")
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    return candidate, digest


def _write_jsonl(path: Path, values: Iterable[Mapping]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def _sampler(
    rows: list[dict], field: str, targets: Mapping[str, float]
) -> tuple[dict[str, float] | None, list[str]]:
    counts = Counter(str(row.get(field, "")) for row in rows)
    missing = [name for name in targets if not counts[name]]
    if missing:
        return None, missing
    return {
        row["row_id"]: targets[str(row[field])] / counts[str(row[field])]
        for row in rows
    }, []


def export_split(split_dir: str | Path, *, released: bool = False) -> dict:
    """Write a portable review/release view without changing permanent IDs."""
    split_dir = Path(split_dir)
    if not (split_dir / "rules.json").exists() and (split_dir.parent / "rules.json").is_file():
        _atomic_json(split_dir / "rules.json",
                     json.loads((split_dir.parent / "rules.json").read_text(encoding="utf-8")))
    registry = _load_registry(split_dir)
    _row_index(registry)
    records = list(registry["rows"].values())
    active = [r for r in records if r.get("active", True) and _status(r) != "excluded"]
    if released:
        from .review_policy import check_release_review
        check_release_review(split_dir, active)
    selected = [
        r for r in active
        if _status(r) == "approved"
        or (_status(r) == "proposed" and (not released or split_dir.name == "train"))
    ]
    selected.sort(key=lambda row: row["row_id"])
    document_hashes = {}
    hashes_by_path = {}
    for row in active:
        relative = str(row["decision_text_path"])
        if relative not in hashes_by_path:
            _, digest = _portable_document(split_dir, relative)
            hashes_by_path[relative] = digest
        digest = hashes_by_path[relative]
        expected = row.get("metadata", {}).get("text_sha256")
        if expected and expected != digest:
            raise ValueError(
                f"Decision text hash differs from prepared row {row['row_id']}: {relative}"
            )
        document_hashes[row["row_id"]] = digest
    metadata = []
    for row in selected:
        extra = {key: value for key, value in row.items() if key not in CSV_COLUMNS}
        metadata.append(
            {"row_id": row["row_id"], **extra,
             "decision_text_sha256": document_hashes[row["row_id"]]}
        )
    csv_path = split_dir / f"{split_dir.name}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows({name: row.get(name, "") for name in CSV_COLUMNS} for row in selected)
    with (split_dir / "pairs_unlabeled.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in sorted(active, key=lambda item: item["row_id"]):
            portable = {name: row.get(name, "") for name in CSV_COLUMNS}
            portable["label"] = ""
            portable["error_type"] = ""
            writer.writerow(portable)
    _write_jsonl(split_dir / "metadata.jsonl", metadata)
    pending = [
        {"row_id": r["row_id"], "status": _status(r)}
        for r in active if _status(r) not in ACTIVE_REVIEW_STATES
    ]
    _write_jsonl(split_dir / "pending.jsonl", pending)
    detection, detection_missing = _sampler(selected, "label", DETECTION_TARGETS)
    rag, rag_missing = _sampler(selected, "relevance", RAG_TARGETS)
    with (split_dir / "sampling_weights.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("row_id", "detection_weight", "rag_weight")
        )
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    "row_id": row["row_id"],
                    "detection_weight": "" if detection is None else detection[row["row_id"]],
                    "rag_weight": "" if rag is None else rag[row["row_id"]],
                }
            )
    contradiction_rows = [r for r in selected if r["label"] == "contradiction"]
    contradiction_sources = {r.get("source_id", r["source_filename"])
                             for r in contradiction_rows}
    provenance_counts = {}
    for provenance, predicate in {
        "natural": lambda row: row.get("variant_id", "original") == "original",
        "synthetic": lambda row: row.get("variant_id", "original") != "original",
    }.items():
        matching = [row for row in contradiction_rows if predicate(row)]
        provenance_counts[provenance] = {
            "pairs": len(matching),
            "documents": len({row.get("source_id", row["source_filename"])
                              for row in matching}),
        }
    variants = {
        "original": {(r.get("source_id", r["source_filename"]),
                      r.get("variant_id", "original")) for r in selected
                     if r.get("variant_id", "original") == "original"},
        "synthetic": {(r.get("source_id", r["source_filename"]),
                       r.get("variant_id", "original")) for r in selected
                      if r.get("variant_id", "original") != "original"},
    }
    type_documents = {
        error_type: len({r.get("source_id", r["source_filename"])
                         for r in contradiction_rows if r["error_type"] == error_type})
        for error_type in sorted({r["error_type"] for r in contradiction_rows})
    }
    root_manifest_path = split_dir.parent / "manifest.json"
    root_manifest = (json.loads(root_manifest_path.read_text(encoding="utf-8"))
                     if root_manifest_path.is_file() else {})
    target = root_manifest.get("policy", {}).get(
        f"{split_dir.name}_contradiction_documents"
    )
    document_target = None if target is None else {
        "target": int(target),
        "actual": len(contradiction_sources),
        "shortage": max(0, int(target) - len(contradiction_sources)),
        "excess": max(0, len(contradiction_sources) - int(target)),
    }
    report = {
        "split": split_dir.name,
        "released": released,
        "pairs": len(selected),
        "documents": len({r.get("source_id", r["source_filename"]) for r in selected}),
        "contradiction_documents": len(contradiction_sources),
        "contradiction_provenance": provenance_counts,
        "variants": {name: len(values) for name, values in variants.items()},
        "variant_pairs": dict(Counter(
            "original" if r.get("variant_id", "original") == "original" else "synthetic"
            for r in selected
        )),
        "error_type_documents": type_documents,
        "contradiction_document_target": document_target,
        "labels": dict(Counter(r["label"] for r in selected)),
        "relevance": dict(Counter(r.get("relevance", "") for r in selected)),
        "error_types": dict(Counter(r["error_type"] for r in selected if r["error_type"])),
        "status": dict(Counter(_status(r) for r in records)),
        "samplers": {
            "detection": {"complete": not detection_missing, "missing": detection_missing},
            "rag": {"complete": not rag_missing, "missing": rag_missing},
        },
    }
    _atomic_json(split_dir / "report.json", report)
    _atomic_json(
        split_dir / "export_snapshot.json",
        {
            "row_ids": [row["row_id"] for row in selected],
            "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "created_at_utc": _now(),
        },
    )
    return report


def import_review_csv(split_dir: str | Path, review_path: str | Path, *, sampled=False) -> dict:
    """Import an expert-edited CSV; omitted IDs become permanently excluded."""
    split_dir, review_path = Path(split_dir), Path(review_path)
    registry = _load_registry(split_dir)
    index = _row_index(registry)
    snapshot_path = split_dir / ("human_review_snapshot.json" if sampled else "export_snapshot.json")
    if not snapshot_path.is_file():
        raise FileNotFoundError("Export a split before importing its review CSV")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    expected = set(snapshot.get("row_ids", []))
    with review_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(CSV_COLUMNS):
            raise ValueError("Review CSV columns or their order have changed")
        reviewed = list(reader)
    ids = [row["row_id"] for row in reviewed]
    if len(ids) != len(set(ids)):
        raise ValueError("Review CSV contains duplicate row IDs")
    unknown = sorted(set(ids) - expected)
    if unknown:
        raise ValueError(f"Review CSV contains unknown or inactive row IDs: {unknown}")
    names, rule_ids = _eligible_rules(split_dir)
    rule_id_by_name = {name: rule_id for rule_id, name in rule_ids.items()}
    for value in reviewed:
        canonical = index[value["row_id"]][1]
        for name in IMMUTABLE_REVIEW_COLUMNS:
            if value[name] != str(canonical.get(name, "")):
                raise ValueError(f"Review changed immutable {name} for {value['row_id']}")
        label, error_type = value["label"].strip().lower(), value["error_type"].strip()
        if label not in LABELS:
            raise ValueError(f"Invalid expert label for {value['row_id']}: {label!r}")
        if label == "contradiction" and error_type not in names:
            raise ValueError(f"Invalid expert error type for {value['row_id']}")
        if label != "contradiction" and error_type:
            raise ValueError(f"Non-contradiction {value['row_id']} has an error type")
    reviewed_by_id = {value["row_id"]: value for value in reviewed}
    for row_id in sorted(expected):
        row = index[row_id][1]
        if row_id not in reviewed_by_id:
            row["status"] = "excluded"
            row.setdefault("annotation_history", []).append(
                {"actor": "human", "status": "excluded", "source": "review_csv",
                 "timestamp_utc": _now()}
            )
            continue
        value = reviewed_by_id[row_id]
        row["label"] = value["label"].strip().lower()
        row["error_type"] = value["error_type"].strip()
        row["rule_id"] = rule_id_by_name.get(row["error_type"], "")
        if row["label"] in {"contradiction", "entailment"}:
            row["relevance"] = "related"
        row["status"] = "approved"
        row.setdefault("annotation_history", []).append(
            {"actor": "human", "status": "approved", "source": "review_csv",
             "label": row["label"], "error_type": row["error_type"],
             "timestamp_utc": _now()}
        )
    _atomic_json(split_dir / "registry.json", registry)
    return {"approved": len(reviewed), "excluded": len(expected - set(ids))}
