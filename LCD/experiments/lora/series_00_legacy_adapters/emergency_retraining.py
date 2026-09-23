"""Replace the full_pipeline_v1 LoRA adapters using source-prefixed premises."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ALL_MODELS = ("llama", "ministral", "qwen", "t-lite")
ALL_TASKS = ("ternary",)

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _repository_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "LCD"
        ).is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the LCD repository")


def _adapter_target(artifact_root: Path, model_alias: str, task: str) -> Path:
    from LCD.shared.common import resolve_model, slugify_model_id

    spec = resolve_model(model_alias)
    return artifact_root / "models" / "lora" / slugify_model_id(spec.model_id) / task


def _load_existing_recipe(target: Path, model_alias: str, task: str) -> dict[str, Any]:
    from LCD.shared.common import resolve_model

    required = (
        target / "run_config.json",
        target / "adapter_config.json",
        target / "tokenizer_config.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Cannot replace an incomplete adapter: " + ", ".join(missing))
    if not any(
        (target / filename).is_file()
        for filename in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise FileNotFoundError(f"Adapter weights are missing: {target}")
    recipe = json.loads(required[0].read_text(encoding="utf-8"))
    spec = resolve_model(model_alias)
    if recipe.get("model_id") != spec.model_id or recipe.get("task") != task:
        raise ValueError(f"Saved recipe does not match {model_alias}/{task}: {target}")
    if not isinstance(recipe.get("hyperparameters"), dict):
        raise ValueError(f"Saved recipe has no hyperparameters: {required[0]}")
    if not recipe.get("resolved_revision"):
        raise ValueError(f"Saved recipe has no resolved base-model revision: {required[0]}")
    return recipe


def run_emergency_lora_retraining(
    *,
    repo_root: str | Path | None = None,
    campaign_id: str = "full_pipeline_v1",
    models: Sequence[str] = ALL_MODELS,
    tasks: Sequence[str] = ALL_TASKS,
    dry_run: bool = False,
):
    """Retrain selected campaign LoRAs in place from their saved configurations.

    Each adapter keeps its recorded hyperparameters and immutable base-model
    revision. Successful training replaces the adapter at the same campaign
    path; validation and DOCX evaluation are intentionally left to the separate
    full-pipeline evaluation workflow.
    """
    import pandas as pd

    unknown_models = sorted(set(models) - set(ALL_MODELS))
    unknown_tasks = sorted(set(tasks) - set(ALL_TASKS))
    if unknown_models:
        raise ValueError("Unknown model aliases: " + ", ".join(unknown_models))
    if unknown_tasks:
        raise ValueError("Unknown tasks: " + ", ".join(unknown_tasks))
    root = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else _repository_root(Path(__file__).resolve())
    )
    artifact_root = root / "local" / "experiments" / "baselines" / "series_01_classifiers" / "campaigns" / campaign_id / "artifacts"
    if not artifact_root.is_dir():
        raise FileNotFoundError(f"Campaign artifact directory does not exist: {artifact_root}")

    jobs: list[dict[str, Any]] = []
    for model_alias in models:
        for task in tasks:
            target = _adapter_target(artifact_root, model_alias, task)
            recipe = _load_existing_recipe(target, model_alias, task)
            jobs.append(
                {
                    "model_alias": model_alias,
                    "task": task,
                    "target": str(target),
                    "model_id": recipe["model_id"],
                    "resolved_revision": recipe["resolved_revision"],
                    "hyperparameters": recipe["hyperparameters"],
                    "rag_revision": recipe.get("rag_revision", "main"),
                }
            )

    plan = pd.DataFrame(
        [
            {
                "position": index,
                "model_alias": job["model_alias"],
                "task": job["task"],
                "model_id": job["model_id"],
                "target": job["target"],
                "status": "planned",
            }
            for index, job in enumerate(jobs, start=1)
        ]
    )
    if dry_run:
        print(plan.to_string(index=False))
        return plan

    from LCD.shared.inference import SOURCE_PREFIXED_PREMISE_FORMAT
    from LCD.shared.lora import run as run_lora

    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_root = root / "local" / "experiments" / "baselines" / "series_01_classifiers" / "campaigns" / campaign_id / "results" / "lora_retraining"
    trainer_root = result_root / "trainer" / session_id
    result_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    report_path = result_root / f"replacement_{session_id}.json"
    for position, job in enumerate(jobs, start=1):
        print(
            f"[{position}/{len(jobs)}] Replacing "
            f"{job['model_alias']}/{job['task']} at {job['target']}"
        )
        record = {
            "position": position,
            "model_alias": job["model_alias"],
            "task": job["task"],
            "target": job["target"],
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        records.append(record)
        report_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        try:
            run_lora(
                job["model_alias"],
                job["task"],
                job["hyperparameters"],
                train_path=root / f"train_{job['task']}.csv",
                val_path=root / f"val_{job['task']}.csv",
                rag_dir=root / "dms-rag",
                rag_revision=job["rag_revision"],
                artifact_root=artifact_root,
                revision=job["resolved_revision"],
                use_existing_model=False,
                results_dir=result_root,
                trainer_output_dir=(
                    trainer_root / job["model_alias"] / job["task"]
                ),
                training_only=True,
            )
            manifest = json.loads(
                (Path(job["target"]) / "run_config.json").read_text(encoding="utf-8")
            )
            if manifest.get("premise_format") != SOURCE_PREFIXED_PREMISE_FORMAT:
                raise RuntimeError(
                    f"Replacement manifest has the wrong premise format: {job['target']}"
                )
            record.update(
                {
                    "status": "completed",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "premise_format": SOURCE_PREFIXED_PREMISE_FORMAT,
                }
            )
        except Exception as error:
            record.update(
                {
                    "status": "failed",
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            report_path.write_text(
                json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            raise
        report_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"All {len(jobs)} LoRA adapters were replaced. Report: {report_path}")
    return pd.DataFrame(records)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--campaign-id", default="full_pipeline_v1")
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS)
    parser.add_argument("--tasks", nargs="+", choices=ALL_TASKS, default=ALL_TASKS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    run_emergency_lora_retraining(
        repo_root=arguments.repo_root,
        campaign_id=arguments.campaign_id,
        models=arguments.models,
        tasks=arguments.tasks,
        dry_run=arguments.dry_run,
    )
