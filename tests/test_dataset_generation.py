import json
from pathlib import Path
import pytest

from LCD.experiments.data_creation.series_01_nli_rag.generation import (
    build_generation_tasks,
)


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _rule(rule_id: str, weight: float, name: str) -> dict:
    return {
        "rule_id": rule_id,
        "name": name,
        "kind": "Противоречие",
        "weight_percent": weight,
        "comment": "",
        "instruction": name,
    }


def _bundle(tmp_path: Path, split: str, sources: list[tuple[str, list[str]]], *,
            target: int = 2, include_control: bool = True) -> Path:
    bundle = tmp_path / "bundle"
    documents, variants, rows = [], [], {}
    next_id = 1
    for source_id, hypotheses in sources:
        text = "Вводная часть. ПОСТАНОВИЛ:\n" + "\n".join(hypotheses)
        relative = f"decisions/originals/{source_id}.txt"
        path = bundle / split / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        documents.append({"source_id": source_id, "source_filename": f"main/18.8/{source_id}.docx",
                          "split": split})
        variants.append({"source_id": source_id, "variant_id": "original",
                         "decision_text_path": relative,
                         "source_filename": f"main/18.8/{source_id}.docx", "active": True})
        for index, hypothesis in enumerate(hypotheses):
            start = text.index(hypothesis)
            hypothesis_id = f"hyp_{source_id}_{index}"
            for unit in range(2):
                row_id = f"{split}_{next_id:06d}"
                next_id += 1
                rows[f"key_{row_id}"] = {
                    "row_id": row_id, "hypothesis": hypothesis,
                    "premise": f"Норма {unit}", "article_number": str(30 + unit),
                    "context": "Контекст", "label": "", "error_type": "",
                    "source_filename": f"main/18.8/{source_id}.docx",
                    "decision_text_path": relative, "source_id": source_id,
                    "variant_id": "original", "hypothesis_id": hypothesis_id,
                    "unit_id": f"unit_{unit}", "active": True, "status": "pending",
                    "metadata": {"hypothesis_span": [start, start + len(hypothesis)],
                                 "retrieval": {"rank": unit + 1}},
                }
    policy = {f"{split}_contradiction_documents": target}
    _write(bundle / "manifest.json", {"seed": 42, "documents": documents, "policy": policy})
    rules = [
        _rule("xlsx_row_32", 60, "Неверно указан срок обжалования постановления"),
        _rule("xlsx_row_33", 40, "Неверно указан срок уплаты административного штрафа"),
    ]
    if include_control:
        rules.append(_rule("xlsx_row_27", 3,
                           "Статья не совпадает; переквалификация в мотивировке выявлена"))
    _write(bundle / "rules.json", rules)
    _write(bundle / split / "manifest.json",
           {"split": split, "documents": documents, "variants": variants, "prepared": {}})
    _write(bundle / split / "registry.json", {"next_id": next_id, "rows": rows})
    return bundle


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_failed_assignment_is_preserved_and_replaced_without_forcing_label(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, "val", [
        ("doc_a", ["Постановление может быть обжаловано в течение 10 суток."]),
        ("doc_b", ["Постановление может быть обжаловано в течение 10 суток."]),
        ("doc_c", ["Постановление может быть обжаловано в течение 10 суток."]),
    ], target=1, include_control=False)
    build_generation_tasks(bundle, split="val")
    plan_path = bundle / "val/tasks/generation_plan.jsonl"
    original = _jsonl(plan_path)[0]
    _write(bundle / "val/generation_feedback/failed.json", {
        "source_id": original["source_id"], "target_hypothesis_id": original["target_hypothesis_id"],
        "rule_id": original["rule_id"], "status": "needs_review", "edits": [],
        "rationale": "Нет подходящих исходных условий для минимальной правки."})
    report = build_generation_tasks(bundle, split="val")["val"]
    tasks = _jsonl(bundle / "val/tasks/generate.jsonl")
    assert report["infeasible_or_inactive_assignments"] == 1
    assert len(tasks) == 1
    assert tasks[0]["task_id"] != original["assignment_id"]
    assert any(p["planned_status"] == "needs_review" for p in _jsonl(plan_path))


def test_deleted_original_pairs_do_not_break_persisted_generation_plan(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, "train", [("doc_a", ["Штраф должен быть уплачен в течение 60 дней."])])
    build_generation_tasks(bundle, split="train")
    registry_path = bundle / "train/registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for row in registry["rows"].values():
        row["status"] = "excluded"
    _write(registry_path, registry)
    result = build_generation_tasks(bundle, split="train")["train"]
    assert result["open_generation_tasks"] == 0
    assert result["infeasible_or_inactive_assignments"] > 0


@pytest.mark.parametrize("failure", [
    {"retrieval_audit": {"status": "missing_target_units"}},
    {"verification_status": "needs_review"},
    {"retrieval_audit": {"target_row_ids": []}},
])
def test_unusable_applied_variant_releases_target_slot_but_keeps_source_reserved(tmp_path: Path, failure) -> None:
    bundle = _bundle(tmp_path, "test", [
        ("doc_a", ["Постановление может быть обжаловано в течение 10 суток."]),
        ("doc_b", ["Постановление может быть обжаловано в течение 10 суток."]),
        ("doc_c", ["Постановление может быть обжаловано в течение 10 суток."]),
    ], target=1, include_control=False)
    build_generation_tasks(bundle, split="test")
    original = _jsonl(bundle / "test/tasks/generation_plan.jsonl")[0]
    registry_path = bundle / "test/registry.json"
    if failure == {"retrieval_audit": {"target_row_ids": []}}:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        target_ids = []
        for row in registry["rows"].values():
            if row["source_id"] == original["source_id"]:
                row.update(status="proposed", label="not mentioned")
                target_ids.append(row["row_id"])
        failure = {"retrieval_audit": {"target_row_ids": target_ids}}
        _write(registry_path, registry)
    manifest_path = bundle / "test/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["variants"].append({
        "source_id": original["source_id"], "variant_id": "syn_failed",
        "target_hypothesis_id": original["target_hypothesis_id"],
        "rule_id": original["rule_id"], **failure,
    })
    _write(manifest_path, manifest)
    registry_before = registry_path.read_bytes()
    report = build_generation_tasks(bundle, split="test")["test"]
    tasks = _jsonl(bundle / "test/tasks/generate.jsonl")
    assert len(tasks) == 1
    assert tasks[0]["source_id"] != original["source_id"]
    assert report["infeasible_or_inactive_assignments"] == 1
    assert registry_path.read_bytes() == registry_before


def test_train_plan_is_stable_and_limits_distinct_rules_per_hypothesis(tmp_path: Path) -> None:
    bundle = _bundle(
        tmp_path,
        "train",
        [("doc_a", ["Постановление может быть обжаловано в течение 10 суток.",
                     "Штраф должен быть уплачен в течение 60 дней."])],
    )

    first = build_generation_tasks(bundle, split="train")["train"]
    plan_path = bundle / "train/tasks/generation_plan.jsonl"
    task_path = bundle / "train/tasks/generate.jsonl"
    plan = _jsonl(plan_path)
    tasks = _jsonl(task_path)
    before = plan_path.read_bytes()
    second = build_generation_tasks(bundle, split="train")["train"]

    assert len(plan) == 4
    by_hypothesis = {}
    for assignment in plan:
        by_hypothesis.setdefault(assignment["target_hypothesis_id"], set()).add(
            assignment["rule_id"]
        )
        assert assignment["planned_status"] == "unconfirmed"
        assert assignment["generation_round"] in {1, 2}
    assert all(len(values) == 2 for values in by_hypothesis.values())
    assert len(tasks) == 4
    assert all(set(task["expert_rule"]) >= {"rule_id", "name"} for task in tasks)
    assert all("expert_rules" not in task for task in tasks)
    assert all(task["target_hypothesis"]["text"] in task["source_text"] for task in tasks)
    assert all(len(task["known_provision_candidates"]) == 2 for task in tasks)
    assert first["planned_assignments"] == second["planned_assignments"] == 4
    assert plan_path.read_bytes() == before


def test_validation_subtracts_confirmed_and_pending_source_reservations(tmp_path: Path) -> None:
    bundle = _bundle(
        tmp_path,
        "val",
        [(f"doc_{letter}", ["Постановление может быть обжаловано в течение 10 суток."])
         for letter in "abcd"],
        target=3,
        include_control=False,
    )
    registry_path = bundle / "val/registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    confirmed = next(row for row in registry["rows"].values() if row["source_id"] == "doc_a")
    confirmed.update(status="approved", label="contradiction", rule_id="xlsx_row_32")
    _write(registry_path, registry)
    manifest_path = bundle / "val/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["variants"].append({
        "source_id": "doc_b", "variant_id": "syn_existing",
        "target_hypothesis_id": "hyp_doc_b_0", "rule_id": "xlsx_row_33",
        "decision_text_path": "decisions/originals/doc_b.txt",
        "source_filename": "main/18.8/doc_b.docx", "active": True,
    })
    _write(manifest_path, manifest)

    report = build_generation_tasks(bundle, split="val")["val"]
    plan = _jsonl(bundle / "val/tasks/generation_plan.jsonl")

    assert report["confirmed_contradiction_documents"] == 1
    assert report["planned_assignments"] == 1
    assert len(plan) == 1
    assert plan[0]["source_id"] in {"doc_c", "doc_d"}
    assert report["shortage_by_rule"] == {}


def test_known_control_is_separate_and_never_fills_contradiction_target(tmp_path: Path) -> None:
    bundle = _bundle(
        tmp_path,
        "test",
        [(f"doc_{number}", ["Статья указана в постановлении."]) for number in range(12)],
        target=10,
        include_control=True,
    )

    report = build_generation_tasks(bundle, split="test")["test"]
    plan = _jsonl(bundle / "test/tasks/generation_plan.jsonl")

    contradictions = [item for item in plan if item["track"] == "contradiction_candidate"]
    controls = [item for item in plan if item["track"] == "control"]
    assert len(contradictions) == 10
    assert len(controls) == 1
    assert controls[0]["rule_id"] == "xlsx_row_27"
    assert report["confirmed_contradiction_documents"] == 0
    assert "unconfirmed" in report["note"]


def test_shortage_is_reported_without_claiming_confirmation(tmp_path: Path) -> None:
    bundle = _bundle(
        tmp_path,
        "test",
        [("doc_only", ["Штраф должен быть уплачен в течение 60 дней."])],
        target=3,
        include_control=False,
    )

    report = build_generation_tasks(bundle, split="test")["test"]

    assert report["planned_assignments"] == 1
    assert sum(report["shortage_by_rule"].values()) == 2
    assert report["confirmed_contradiction_documents"] == 0


def test_applied_assignment_stays_in_plan_but_leaves_open_tasks(tmp_path: Path) -> None:
    bundle = _bundle(
        tmp_path,
        "train",
        [("doc_a", ["Постановление может быть обжаловано в течение 10 суток."])],
        include_control=False,
    )
    build_generation_tasks(bundle, split="train")
    plan = _jsonl(bundle / "train/tasks/generation_plan.jsonl")
    applied = plan[0]
    manifest_path = bundle / "train/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["variants"].append({
        "source_id": applied["source_id"], "variant_id": "syn_applied",
        "target_hypothesis_id": applied["target_hypothesis_id"],
        "rule_id": applied["rule_id"],
        "decision_text_path": "decisions/originals/doc_a.txt",
        "source_filename": "main/18.8/doc_a.docx", "active": True,
    })
    _write(manifest_path, manifest)

    report = build_generation_tasks(bundle, split="train")["train"]

    assert report["planned_assignments"] == 2
    assert report["applied_assignments"] == 1
    assert report["open_generation_tasks"] == 1
    assert len(_jsonl(bundle / "train/tasks/generation_plan.jsonl")) == 2

