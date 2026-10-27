"""Deterministic, auditable scheduling of subagent generation assignments.

This module selects candidates and expert rules but never edits a decision and
never assigns an NLI label.  Every scheduled contradiction is explicitly an
unconfirmed candidate until annotation and human review establish otherwise.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Iterable, Mapping

from .preparation import quota_targets
from ._io import replace_checkpoint


SPLITS = ("train", "val", "test")
CONTROL_RULE_IDS = frozenset({"xlsx_row_27"})


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    values = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON on {path}:{line_number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object on {path}:{line_number}")
        values.append(value)
    return values


def _write_jsonl(path: Path, values: Iterable[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, newline="\n"
    ) as stream:
        for value in values:
            stream.write(json.dumps(dict(value), ensure_ascii=False) + "\n")
        temporary = Path(stream.name)
    replace_checkpoint(temporary, path)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, newline="\n"
    ) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    replace_checkpoint(temporary, path)


def _digest(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _portable_file(split_dir: Path, relative_value: str) -> Path:
    relative = Path(relative_value)
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or "\\" in relative_value
    ):
        raise ValueError(f"Unsafe decision text path: {relative_value!r}")
    current = split_dir
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Decision text must be a copied plain file: {relative_value}")
    if not current.is_file() or not current.resolve().is_relative_to(split_dir.resolve()):
        raise ValueError(f"Missing or escaping decision text: {relative_value}")
    return current


def _rules(bundle: Path) -> list[dict]:
    value = _read_json(bundle / "rules.json")
    rules = value.get("rules") if isinstance(value, dict) else value
    if not isinstance(rules, list) or not rules:
        raise ValueError("rules.json must contain eligible expert rules")
    result = []
    seen = set()
    for rule in rules:
        if not isinstance(rule, dict) or not rule.get("rule_id") or not rule.get("name"):
            raise ValueError("Every generation rule requires rule_id and name")
        rule_id = str(rule["rule_id"])
        if rule_id in seen:
            raise ValueError(f"Duplicate generation rule: {rule_id}")
        seen.add(rule_id)
        result.append(dict(rule))
    return result


def _lexical_terms(rule: Mapping[str, object]) -> tuple[str, ...]:
    """Return narrow feasibility hints; these never establish legal compatibility."""
    value = " ".join(
        str(rule.get(name, "")).casefold() for name in ("name", "comment", "instruction")
    )
    groups = (
        (("обжалован", "обжалования"), ("обжал", "жалоб")),
        (("уплаты", "неуплату", "неуплачен"), ("уплат", "неуплат")),
        (("самостоятельного контролируемого выезда",), ("самостоятельн", "выезд")),
        (("содержания в специальном учреждении",), ("содержан", "специальн")),
        (("федерального закона",), ("федеральн", "фз")),
        (("повторност", "предыдущее постановление"), ("повтор", "ранее")),
        (("выдворен",), ("выдвор",)),
        (("штраф",), ("штраф",)),
        (("отрицан", "антоним"), ("может", "обязан", "подлежит", "не ")),
        (("стать", "част", "пункт", "норм"), ("ст.", "стать", "част", "пункт", "коап")),
    )
    terms: list[str] = []
    for triggers, hints in groups:
        if any(trigger in value for trigger in triggers):
            terms.extend(hints)
    return tuple(dict.fromkeys(terms))


def _hint(rule: Mapping[str, object], hypothesis: str) -> dict:
    terms = _lexical_terms(rule)
    normalized = hypothesis.casefold()
    matched = [term for term in terms if term in normalized]
    return {
        "method": "lexical_hint_only",
        "terms": list(terms),
        "matched_terms": matched,
        "status": "candidate" if matched or not terms else "needs_agent_review",
    }


def _candidates(split_dir: Path, registry: Mapping, manifest: Mapping) -> list[dict]:
    originals = {
        value["source_id"]: value
        for value in manifest.get("variants", [])
        if value.get("variant_id") == "original"
    }
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in registry.get("rows", {}).values():
        if (
            row.get("variant_id") == "original"
            and row.get("status") != "excluded"
            and row.get("source_id") in originals
            and row.get("hypothesis_id")
        ):
            grouped[(str(row["source_id"]), str(row["hypothesis_id"]))].append(row)
    candidates = []
    for (source_id, hypothesis_id), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["row_id"])
        texts = {str(row.get("hypothesis", "")) for row in rows}
        spans = {
            tuple(row.get("metadata", {}).get("hypothesis_span", ())) for row in rows
        }
        if len(texts) != 1 or len(spans) != 1:
            raise ValueError(f"Inconsistent hypothesis records: {source_id}/{hypothesis_id}")
        span = next(iter(spans))
        if len(span) != 2 or not all(type(value) is int for value in span):
            raise ValueError(f"Hypothesis lacks integer source offsets: {hypothesis_id}")
        provisions, seen_units = [], set()
        for row in rows:
            unit_id = str(row.get("unit_id", ""))
            identity = unit_id or _digest([row.get("premise"), row.get("article_number")])
            if identity in seen_units:
                continue
            seen_units.add(identity)
            provisions.append(
                {
                    "unit_id": unit_id,
                    "text": str(row.get("premise", "")),
                    "article_number": str(row.get("article_number", "")),
                    "retrieval": {key: value for key, value in
                                  row.get("metadata", {}).get("retrieval", {}).items()
                                  if key != "text"},
                    "row_id": row["row_id"],
                }
            )
        original = originals[source_id]
        candidates.append(
            {
                "source_id": source_id,
                "source_filename": original["source_filename"],
                "decision_text_path": original["decision_text_path"],
                "target_hypothesis": {
                    "hypothesis_id": hypothesis_id,
                    "text": next(iter(texts)),
                    "start": span[0],
                    "end": span[1],
                },
                "known_provision_candidates": provisions,
            }
        )
    return candidates


def _ordered(values: Iterable[dict], seed: int, fields: tuple[str, ...]) -> list[dict]:
    return sorted(values, key=lambda value: _digest([seed, *[value[field] for field in fields]]))


def _choose_rule(
    candidate: Mapping, rules: list[dict], remaining: Counter, used_rules: set[str]
) -> tuple[dict, dict] | None:
    available = [
        rule for rule in rules
        if remaining[rule["rule_id"]] > 0 and rule["rule_id"] not in used_rules
    ]
    if not available:
        return None
    ranked = []
    for rule in available:
        hint = _hint(rule, candidate["target_hypothesis"]["text"])
        ranked.append((len(hint["matched_terms"]), remaining[rule["rule_id"]], rule, hint))
    _, _, rule, hint = max(
        ranked, key=lambda item: (item[0], item[1], item[2]["rule_id"])
    )
    return rule, hint


def _assignment(split: str, candidate: Mapping, rule: Mapping, hint: Mapping) -> dict:
    rule_id = str(rule["rule_id"])
    assignment_id = "gen_" + _digest(
        [split, candidate["source_id"], candidate["target_hypothesis"]["hypothesis_id"], rule_id]
    )[:20]
    return {
        "assignment_id": assignment_id,
        "split": split,
        "track": "control" if rule_id in CONTROL_RULE_IDS else "contradiction_candidate",
        "source_id": candidate["source_id"],
        "target_hypothesis_id": candidate["target_hypothesis"]["hypothesis_id"],
        "rule_id": rule_id,
        "feasibility": dict(hint),
        "planned_status": "unconfirmed",
    }


def _variant_signatures(manifest: Mapping) -> set[tuple[str, str, str]]:
    return {
        (str(item.get("source_id", "")), str(item.get("target_hypothesis_id", "")),
         str(item.get("rule_id", "")))
        for item in manifest.get("variants", [])
        if item.get("variant_id") != "original" and item.get("target_hypothesis_id")
    }


def _confirmed(registry: Mapping) -> tuple[set[str], Counter]:
    sources, rules = set(), Counter()
    for row in registry.get("rows", {}).values():
        if (
            row.get("active", True)
            and row.get("status") == "approved"
            and row.get("label") == "contradiction"
        ):
            source_id = str(row.get("source_id", ""))
            sources.add(source_id)
            rule_id = str(row.get("rule_id") or row.get("metadata", {}).get("scenario_rule_id", ""))
            if rule_id:
                rules[rule_id] += 1
    return sources, rules


def _unusable_variant_signatures(manifest: Mapping, registry: Mapping) -> set[tuple[str, str, str]]:
    """Identify applied scenarios that cannot currently reserve a positive slot.

    This only affects scheduling. It neither changes labels nor removes copies.
    Fully annotated target pairs without a contradiction are useful negatives,
    but cannot fill a contradiction-document quota.
    """
    rows = {row["row_id"]: row for row in registry.get("rows", {}).values()}
    unavailable = set()
    for variant in manifest.get("variants", []):
        if variant.get("variant_id") == "original":
            continue
        audit = variant.get("retrieval_audit", {})
        target_ids = audit.get("target_row_ids", [])
        target_rows = [rows[row_id] for row_id in target_ids if row_id in rows]
        neutral_target = bool(target_ids) and len(target_rows) == len(target_ids) and all(
            row.get("status") in {"proposed", "approved"}
            and row.get("label") in {"entailment", "not mentioned"}
            for row in target_rows
        )
        if (variant.get("verification_status") == "needs_review"
                or audit.get("status") == "missing_target_units" or neutral_target):
            unavailable.add((str(variant.get("source_id", "")),
                             str(variant.get("target_hypothesis_id", "")),
                             str(variant.get("rule_id", ""))))
    return unavailable


def _existing_plan(path: Path, split: str, rules: Mapping[str, dict],
                   candidates: Mapping[tuple[str, str], dict], known_hypotheses=None) -> list[dict]:
    plan = _read_jsonl(path)
    identities = set()
    for item in plan:
        identity = (str(item.get("source_id", "")), str(item.get("target_hypothesis_id", "")))
        candidate = candidates.get(
            identity,
            {
                "source_id": identity[0],
                "target_hypothesis": {"hypothesis_id": identity[1]},
            },
        )
        rule = rules.get(
            item.get("rule_id"), {"rule_id": item.get("rule_id")}
        )
        expected = _assignment(
            split, candidate, rule, item.get("feasibility", {})
        )["assignment_id"]
        if (
            item.get("split") != split
            or identity not in (known_hypotheses if known_hypotheses is not None else candidates)
            or item.get("rule_id") not in rules
            or item.get("assignment_id") != expected
            or item.get("assignment_id") in identities
        ):
            raise ValueError(f"Invalid persisted generation assignment in {path}")
        identities.add(item["assignment_id"])
        if identity not in candidates:
            item["planned_status"] = "inactive"
            item["inactive_reason"] = "All original candidate pairs were excluded"
    return plan


def _task(split_dir: Path, split: str, assignment: Mapping, candidate: Mapping,
          rule: Mapping, manifest: Mapping, source_texts: dict | None = None) -> dict:
    source_texts = {} if source_texts is None else source_texts
    relative = candidate["decision_text_path"]
    if relative not in source_texts:
        source_texts[relative] = _portable_file(split_dir, relative).read_text(encoding="utf-8")
    source_text = source_texts[relative]
    target = candidate["target_hypothesis"]
    if source_text[target["start"]:target["end"]] != target["text"]:
        raise ValueError(f"Stale hypothesis offsets: {target['hypothesis_id']}")
    existing = [
        {key: value for key, value in variant.items() if key != "source_text"}
        for variant in manifest.get("variants", [])
        if variant.get("source_id") == candidate["source_id"]
        and variant.get("variant_id") != "original"
    ]
    return {
        "task_id": assignment["assignment_id"],
        "kind": "generate",
        "split": split,
        "source_id": candidate["source_id"],
        "source_filename": candidate["source_filename"],
        "source_text": source_text,
        "target_hypothesis_id": target["hypothesis_id"],
        "target_hypothesis": target,
        "expert_rule": dict(rule),
        "known_provision_candidates": candidate["known_provision_candidates"],
        "existing_variants": existing,
        "feasibility": assignment["feasibility"],
        "generation_round": assignment.get("generation_round", 1),
        "policy": {
            "planned_track": assignment["track"],
            "planned_status": "unconfirmed",
            "max_train_variants_per_hypothesis": 2,
            "val_test_max_variants_per_source": 1,
        },
    }


def _build_one(bundle: Path, split: str, rules: list[dict], root: Mapping) -> dict:
    split_dir = bundle / split
    manifest = _read_json(split_dir / "manifest.json")
    registry = _read_json(split_dir / "registry.json")
    candidates = _candidates(split_dir, registry, manifest)
    candidate_index = {
        (candidate["source_id"], candidate["target_hypothesis"]["hypothesis_id"]): candidate
        for candidate in candidates
    }
    rules_by_id = {rule["rule_id"]: rule for rule in rules}
    contradiction_rules = [rule for rule in rules if rule["rule_id"] not in CONTROL_RULE_IDS]
    control_rules = [rule for rule in rules if rule["rule_id"] in CONTROL_RULE_IDS]
    if not contradiction_rules:
        raise ValueError("Generation requires at least one non-control expert rule")
    tasks_dir = split_dir / "tasks"
    plan_path = tasks_dir / "generation_plan.jsonl"
    known_hypotheses = {(str(r.get("source_id", "")), str(r.get("hypothesis_id", "")))
                        for r in registry.get("rows", {}).values() if r.get("variant_id") == "original"}
    historical = _existing_plan(plan_path, split, rules_by_id, candidate_index, known_hypotheses)
    # An applied edit is not itself evidence of a usable contradiction pair.
    # Keep its source reserved, but fill the positive-document slot elsewhere
    # when independent verification or actual target retrieval failed.
    unavailable_variants = (_unusable_variant_signatures(manifest, registry)
                            if split != "train" else set())
    failures = {}
    for path in sorted((split_dir / "generation_feedback").glob("*.json")):
        feedback = _read_json(path)
        if feedback.get("status") == "needs_review":
            signature = (feedback["source_id"], feedback["target_hypothesis_id"], feedback["rule_id"])
            failures[signature] = feedback
    inactive = []
    plan = []
    for item in historical:
        signature = (item["source_id"], item["target_hypothesis_id"], item["rule_id"])
        if signature in failures:
            item["planned_status"] = "needs_review"
            item["failure_reason"] = failures[signature].get("rationale", "")
        elif signature in unavailable_variants:
            item["planned_status"] = "needs_review"
            item["failure_reason"] = "Applied variant lacks a usable retrieved contradiction target"
        if item.get("planned_status") in {"needs_review", "inactive"}:
            inactive.append(item)
        else:
            plan.append(item)
    plan_by_hypothesis = Counter(
        (item["source_id"], item["target_hypothesis_id"]) for item in plan
    )
    if split == "train" and any(count > 2 for count in plan_by_hypothesis.values()):
        raise ValueError("Persisted train plan exceeds two variants per hypothesis")
    if split != "train" and len({item["source_id"] for item in plan}) != len(plan):
        raise ValueError("Persisted val/test plan assigns a source more than once")
    variant_signatures = _variant_signatures(manifest)
    plan_signatures = {
        (item["source_id"], item["target_hypothesis_id"], item["rule_id"])
        for item in plan
    }
    applied_assignment_ids = {
        _assignment(split, candidate_index[(source, hypothesis)], rules_by_id[rule_id], {})["assignment_id"]
        for source, hypothesis, rule_id in variant_signatures
        if (source, hypothesis) in candidate_index and rule_id in rules_by_id
    }
    seed = int(root.get("seed", 42))
    ordered = _ordered(candidates, seed, ("source_id",))
    used_by_hypothesis: dict[tuple[str, str], set[str]] = defaultdict(set)
    for source, hypothesis, rule_id in failures:
        used_by_hypothesis[(source, hypothesis)].add(rule_id)
    reservation_counts: Counter = Counter()
    used_sources = set()
    for item in plan:
        key = (item["source_id"], item["target_hypothesis_id"])
        used_by_hypothesis[key].add(item["rule_id"])
        reservation_counts[key] += 1
        used_sources.add(item["source_id"])
    for source, hypothesis, rule_id in variant_signatures:
        used_by_hypothesis[(source, hypothesis)].add(rule_id)
        used_sources.add(source)
    # A variant normally realizes a persisted assignment and must not consume a
    # second reservation. Manually imported variants without a plan do consume
    # one, including two distinct variants produced under the same rule.
    variant_counts = Counter()
    for variant in manifest.get("variants", []):
        signature = (
            str(variant.get("source_id", "")),
            str(variant.get("target_hypothesis_id", "")),
            str(variant.get("rule_id", "")),
        )
        if variant.get("variant_id") != "original":
            variant_counts[signature] += 1
    variants_without_plan = Counter()
    for signature, count in variant_counts.items():
        unmatched = count - int(signature in plan_signatures)
        if unmatched > 0:
            variants_without_plan[(signature[0], signature[1])] += unmatched
    reservation_counts.update(variants_without_plan)

    confirmed_sources, _ = _confirmed(registry)
    new_assignments: list[dict] = []
    shortage = Counter()
    requested = Counter()

    if split == "train":
        capacity = sum(max(0, 2 - reservation_counts[
            (candidate["source_id"], candidate["target_hypothesis"]["hypothesis_id"])
        ]) for candidate in ordered)
        requested.update(quota_targets(rules, capacity))
        remaining = Counter(requested)
        # Give every hypothesis its first assignment before adding second variants.
        for round_number in (0, 1):
            for candidate in ordered:
                key = (candidate["source_id"], candidate["target_hypothesis"]["hypothesis_id"])
                if reservation_counts[key] > round_number or reservation_counts[key] >= 2:
                    continue
                selected = _choose_rule(candidate, rules, remaining, used_by_hypothesis[key])
                if selected is None:
                    continue
                rule, hint = selected
                assignment = _assignment(split, candidate, rule, hint)
                assignment["generation_round"] = reservation_counts[key] + 1
                new_assignments.append(assignment)
                used_by_hypothesis[key].add(rule["rule_id"])
                reservation_counts[key] += 1
                remaining[rule["rule_id"]] -= 1
        shortage.update({rule_id: count for rule_id, count in remaining.items() if count > 0})
    else:
        target_key = f"{split}_contradiction_documents"
        default_target = 120 if split == "val" else 40
        target = int(root.get("policy", {}).get(target_key, default_target))
        planned_contradiction_sources = {
            item["source_id"] for item in plan if item["track"] == "contradiction_candidate"
        }
        variant_contradiction_sources = {
            source for source, hypothesis, rule_id in variant_signatures
            if rule_id not in CONTROL_RULE_IDS
            and (source, hypothesis, rule_id) not in unavailable_variants
        }
        remaining_target = max(
            0, target - len(confirmed_sources)
            - len((planned_contradiction_sources | variant_contradiction_sources) - confirmed_sources)
        )
        preferred_minimum = 3 if split == "val" else 1
        minimum = (preferred_minimum if remaining_target >= preferred_minimum * len(contradiction_rules)
                   else 1 if remaining_target >= len(contradiction_rules) else 0)
        requested.update(quota_targets(contradiction_rules, remaining_target, minimum=minimum))
        remaining = Counter(requested)
        available = [candidate for candidate in ordered if candidate["source_id"] not in used_sources
                     and candidate["source_id"] not in confirmed_sources]
        # One source may have many hypotheses. Choose the best lexical candidate
        # for the next still-needed rule while reserving the source exactly once.
        by_source: dict[str, list[dict]] = defaultdict(list)
        for candidate in available:
            by_source[candidate["source_id"]].append(candidate)
        for source_id in _ordered([{"source_id": key} for key in by_source], seed, ("source_id",)):
            source = source_id["source_id"]
            if not any(remaining.values()):
                break
            choices = []
            for candidate in by_source[source]:
                key = (candidate["source_id"], candidate["target_hypothesis"]["hypothesis_id"])
                selected = _choose_rule(candidate, contradiction_rules, remaining, used_by_hypothesis[key])
                if selected:
                    rule, hint = selected
                    choices.append((len(hint["matched_terms"]), remaining[rule["rule_id"]],
                                    candidate, rule, hint))
            if not choices:
                continue
            _, _, candidate, rule, hint = max(
                choices, key=lambda item: (item[0], item[1], item[3]["rule_id"])
            )
            assignment = _assignment(split, candidate, rule, hint)
            assignment["generation_round"] = 1
            new_assignments.append(assignment)
            remaining[rule["rule_id"]] -= 1
            used_sources.add(source)
        shortage.update({rule_id: count for rule_id, count in remaining.items() if count > 0})

        # The known justified-requalification control is scheduled separately
        # and never reduces the contradiction-document target.
        if control_rules:
            contradiction_weight = sum(float(rule["weight_percent"]) for rule in contradiction_rules)
            control_weight = sum(float(rule["weight_percent"]) for rule in control_rules)
            control_target = (
                max(1, round(target * control_weight / contradiction_weight))
                if target and control_weight else 0
            )
            existing_controls = {
                item["source_id"] for item in plan if item["track"] == "control"
            } | {source for source, _, rule_id in variant_signatures if rule_id in CONTROL_RULE_IDS}
            control_remaining_total = max(0, control_target - len(existing_controls))
            control_quotas = quota_targets(control_rules, control_remaining_total)
            control_remaining = Counter(control_quotas)
            for candidate in ordered:
                if candidate["source_id"] in used_sources or candidate["source_id"] in confirmed_sources:
                    continue
                key = (candidate["source_id"], candidate["target_hypothesis"]["hypothesis_id"])
                selected = _choose_rule(candidate, control_rules, control_remaining, used_by_hypothesis[key])
                if selected is None:
                    continue
                rule, hint = selected
                assignment = _assignment(split, candidate, rule, hint)
                assignment["generation_round"] = 1
                new_assignments.append(assignment)
                control_remaining[rule["rule_id"]] -= 1
                used_sources.add(candidate["source_id"])
                if not any(control_remaining.values()):
                    break
            shortage.update({f"control:{key}": value for key, value in control_remaining.items()
                             if value > 0})

    assignment_ids = {item["assignment_id"] for item in plan}
    for item in new_assignments:
        if item["assignment_id"] in assignment_ids:
            continue
        assignment_ids.add(item["assignment_id"])
        plan.append(item)
    plan.sort(key=lambda item: item["assignment_id"])
    _write_jsonl(plan_path, sorted([*plan, *inactive], key=lambda item: item["assignment_id"]))
    tasks = []
    source_texts = {}
    for item in plan:
        if item["assignment_id"] in applied_assignment_ids:
            continue
        candidate = candidate_index[(item["source_id"], item["target_hypothesis_id"])]
        tasks.append(_task(split_dir, split, item, candidate, rules_by_id[item["rule_id"]], manifest,
                           source_texts=source_texts))
    tasks.sort(key=lambda item: item["task_id"])
    task_path = tasks_dir / "generate.jsonl"
    _write_jsonl(task_path, tasks)
    planned_by_track = Counter(item["track"] for item in plan)
    report = {
        "split": split,
        "contradiction_document_target": None if split == "train" else int(
            root.get("policy", {}).get(
                f"{split}_contradiction_documents", 120 if split == "val" else 40
            )
        ),
        "confirmed_contradiction_documents": len(confirmed_sources),
        "planned_assignments": len(plan),
        "planned_unconfirmed_by_track": dict(sorted(planned_by_track.items())),
        "open_generation_tasks": len(tasks),
        "tasks": len(tasks),
        "applied_assignments": len(applied_assignment_ids),
        "infeasible_or_inactive_assignments": len(inactive),
        "infeasible_by_rule": dict(Counter(item["rule_id"] for item in inactive
                                           if item.get("planned_status") == "needs_review")),
        "new_rule_targets": dict(sorted(requested.items())),
        "shortage_by_rule": dict(sorted(shortage.items())),
        "note": "Planned candidates remain unconfirmed and do not change NLI labels.",
        "plan_path": str(plan_path),
        "task_path": str(task_path),
        "path": str(task_path),
    }
    _write_json(tasks_dir / "generation_report.json", report)
    return report


def build_generation_tasks(bundle: str | Path, *, split: str | None = None) -> dict:
    """Persist deterministic rule/hypothesis assignments and generation tasks.

    ``generation_plan.jsonl`` is the stable assignment ledger.  Re-running the
    function preserves its assignment IDs, omits already-applied variants from
    the open task file, and only adds assignments needed by remaining capacity.
    """
    bundle = Path(bundle)
    if split is not None and split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    root = _read_json(bundle / "manifest.json")
    rules = _rules(bundle)
    return {
        name: _build_one(bundle, name, rules, root)
        for name in ((split,) if split else SPLITS)
    }

