"""Shared resumable engine for staged local LoRA hyperparameter searches."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from LCD.shared.common import (
    file_sha256,
    merge_parameters,
    prompt_sha256,
    resolve_huggingface_revision,
    resolve_model,
    slugify_model_id,
    source_tree_sha256,
)
from LCD.shared.lora import DEFAULT_LORA_HYPERPARAMETERS
from LCD.shared.prompting import prompt_for_task
from LCD.shared.common import REPOSITORY_ROOT

ALL_MODELS = ("qwen", "llama", "ministral", "t-lite")
ALL_TASKS = ("ternary",)
DEFAULT_SWEEP_TASKS = ("ternary",)
DEFAULT_SEARCH_ID = "lora_coordinate_imbalanced_val"
PINNED_RAG_REVISION = "e6eab944161e1266c1b4452f172a9a725b1abe97"
PINNED_MODEL_REVISIONS = {
    "qwen": "b968826d9c46dd6066d109eabc6255188de91218",
    "llama": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
    "ministral": "2f494a194c5b980dfb9772cb92d26cbb671fce5a",
    "t-lite": "d125c970c553de58fcee3c937d5e4867d4a448d8",
}


def _rag_path(root: Path) -> Path:
    organized = root / "local" / "resources" / "dms-rag"
    return organized if organized.exists() else root / "dms-rag"
STAGE_ORDER = ("target_modules", "rank", "learning_rate", "alpha", "dropout")
STAGE_PREDECESSOR = {
    "target_modules": None,
    "rank": "target_modules",
    "learning_rate": "rank",
    "alpha": "learning_rate",
    "dropout": "alpha",
}
DATASETS = ("Full",)
BENCHMARK_SCOPES = ("autotest_model",)

HISTORICAL_LORA_OVERRIDES: dict[str, Any] = {
    "lora_rank": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj"],
    "lora_dropout": 0.1,
    "max_seq_length": 1024,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "epochs": 5,
    "learning_rate": 2e-5,
    "lr_scheduler_type": "linear",
    "warmup_ratio": 0.0,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "gradient_checkpointing": True,
    "optimizer": "adamw_torch_fused",
    "logging_steps": 10,
    "eval_strategy": "epoch",
    "eval_batch_size": 8,
    "save_strategy": "epoch",
    "save_total_limit": 1,
    "load_best_model_at_end": False,
    "num_workers": 0,
    "quantization": True,
    "device_map": "auto",
    "precision": "float16",
    "inference_batch_size": 1,
    "document_batch_size": 4,
    "embedding_device": "cpu",
    "seed": 42,
    "deterministic": True,
}

TARGET_MODULE_GRID = (
    ("qv", ["q_proj", "v_proj"]),
    ("qkv", ["q_proj", "k_proj", "v_proj"]),
    ("all_linear", "all-linear"),
)
RANK_GRID = (8, 16, 32)
LEARNING_RATE_GRID = (2e-5, 1e-4, 2e-4, 1e-5)
ALPHA_MULTIPLIER_GRID = (1, 2)
DROPOUT_GRID = (0.0, 0.05, 0.1)


@dataclass(frozen=True)
class SweepCandidate:
    """One logical stage candidate backed by one canonical experiment."""

    stage: str
    model_alias: str
    task: str
    label: str
    order: int
    parameters: dict[str, Any]

    @property
    def candidate_id(self) -> str:
        return f"{self.stage}:{self.model_alias}:{self.task}:{self.label}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_csv(path: Path, frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)


def _load_scores(path: Path):
    import pandas as pd

    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _validate_scope(models: Sequence[str], tasks: Sequence[str]) -> None:
    if not models or not tasks:
        raise ValueError("models and tasks cannot be empty")
    unknown_models = sorted(set(models) - set(ALL_MODELS))
    unknown_tasks = sorted(set(tasks) - set(ALL_TASKS))
    if unknown_models:
        raise ValueError("Unknown model aliases: " + ", ".join(unknown_models))
    if unknown_tasks:
        raise ValueError("Unknown tasks: " + ", ".join(unknown_tasks))
    if len(models) != len(set(models)) or len(tasks) != len(set(tasks)):
        raise ValueError("models and tasks cannot contain duplicates")


def _base_parameters(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    if overrides and any(
        key in overrides
        for key in (
            "target_modules",
            "lora_rank",
            "lora_alpha",
            "learning_rate",
            "lora_dropout",
        )
    ):
        raise ValueError(
            "Swept parameters cannot be supplied through hyperparameters"
        )
    parameters = merge_parameters(
        DEFAULT_LORA_HYPERPARAMETERS, HISTORICAL_LORA_OVERRIDES
    )
    return merge_parameters(parameters, overrides)


def _configuration(
    root: Path,
    models: Sequence[str],
    tasks: Sequence[str],
    base_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    entrypoints = [
        root / name
        for name in (
            "LCD/experiments/lora/series_01_coordinate_search/target_modules.py",
            "LCD/experiments/lora/series_01_coordinate_search/rank.py",
            "LCD/experiments/lora/series_01_coordinate_search/learning_rate.py",
            "LCD/experiments/lora/series_01_coordinate_search/alpha.py",
            "LCD/experiments/lora/series_01_coordinate_search/dropout.py",
            "LCD/experiments/lora/series_01_coordinate_search/compare_llm.py",
        )
    ]
    return _json_value(
        {
            "models": list(models),
            "tasks": list(tasks),
            "base_hyperparameters": dict(base_parameters),
            "dataset_sha256": {
                f"{split}_{task}": file_sha256(root / "local" / "data" / "classification" / f"{split}.csv")
                for split in ("train", "val")
                for task in tasks
            },
            "prompt_sha256": {
                task: prompt_sha256(prompt_for_task(task)) for task in tasks
            },
            "source_tree_sha256": source_tree_sha256(),
            "entrypoint_sha256": {
                path.name: file_sha256(path) for path in entrypoints if path.is_file()
            },
            "rag_requested_revision": PINNED_RAG_REVISION,
            "test_datasets": list(DATASETS),
            "benchmark_scopes": list(BENCHMARK_SCOPES),
            "stage_order": list(STAGE_ORDER),
        }
    )


def _preflight(root: Path, models: Sequence[str], tasks: Sequence[str], *, gpu: bool) -> None:
    required = [
        root / "local" / "data" / "classification" / f"{split}.csv"
        for split in ("train", "val")
        for task in tasks
    ]
    required.extend(
        [
            _rag_path(root) / "codex.csv",
            _rag_path(root) / "faiss_index" / "index.faiss",
            _rag_path(root) / "faiss_index" / "index.pkl",
        ]
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing experiment artifacts: " + ", ".join(missing))
    from LCD.shared.autotest_scoring import discover_autotest_datasets

    discovered = discover_autotest_datasets(
        root / "local" / "data" / "benchmarks" / "autotest", root / "local" / "data" / "benchmarks" / "test_docx", multiple_test=True,
        selected_datasets=DATASETS,
    )
    counts = {dataset.name: len(dataset.documents) for dataset in discovered}
    if counts != {"Full": 30}:
        raise ValueError(f"Unexpected benchmark composition: {counts}")
    if not gpu:
        return
    if "llama" in models and not (
        os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    ):
        raise RuntimeError(
            "Llama experiments require HF_TOKEN or HUGGING_FACE_HUB_TOKEN"
        )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-capable NVIDIA GPU is required")
    for package in (
        "accelerate",
        "bitsandbytes",
        "faiss",
        "langchain_community",
        "peft",
        "sentence_transformers",
        "transformers",
    ):
        __import__(package)


def _resolve_pins(root: Path, models: Sequence[str]) -> dict[str, Any]:
    from LCD.shared.local_support import get_huggingface_token
    from LCD.shared.retrieval import ensure_rag_repository

    _, rag_revision = ensure_rag_repository(_rag_path(root), revision=PINNED_RAG_REVISION)
    return {"model_revisions": {alias: PINNED_MODEL_REVISIONS[alias] for alias in models},
            "rag_revision": rag_revision}


def _load_or_create_state(
    path: Path,
    *,
    search_id: str,
    configuration: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("search_id") != search_id:
            raise ValueError(f"Search state does not match {search_id!r}: {path}")
        if state.get("configuration") != configuration:
            raise ValueError(
                "The search ID belongs to different code, inputs, scope, or settings"
            )
        for experiment in state.get("experiments", {}).values():
            if experiment.get("status") == "running":
                experiment["status"] = "interrupted"
                experiment["error"] = "Previous process ended during this experiment"
        for record in state.get("comparison", {}).get("models", {}).values():
            if record.get("status") == "running":
                record["status"] = "interrupted"
                record["error"] = "Previous process ended during this evaluation"
        return state
    models = configuration["models"]
    return {
        "search_id": search_id,
        "configuration": dict(configuration),
        "pins": _resolve_pins(root, models),
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "stages": {},
        "experiments": {},
        "comparison": {"models": {}},
    }


def _recipe_id(
    model_alias: str,
    task: str,
    parameters: Mapping[str, Any],
    state: Mapping[str, Any],
) -> str:
    config = state["configuration"]
    payload = _json_value(
        {
            "model_alias": model_alias,
            "model_id": resolve_model(model_alias).model_id,
            "task": task,
            "parameters": parameters,
            "revision": state["pins"]["model_revisions"][model_alias],
            "rag_revision": state["pins"]["rag_revision"],
            "train_sha256": config["dataset_sha256"][f"train_{task}"],
            "validation_sha256": config["dataset_sha256"][f"val_{task}"],
            "prompt_sha256": config["prompt_sha256"][task],
        }
    )
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _format_float(value: float) -> str:
    number = float(value)
    if number == 0:
        return "0"
    if not math.isfinite(number):
        raise ValueError("Sweep values must be finite")
    mantissa, exponent = f"{number:.12e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"


def _winner_parameters(
    state: Mapping[str, Any], stage: str, model_alias: str, task: str
) -> dict[str, Any]:
    winner = state["stages"][stage]["winners"][f"{model_alias}:{task}"]
    experiment = state["experiments"][winner["recipe_id"]]
    return dict(experiment["parameters"])


def build_stage_candidates(
    stage: str,
    state: Mapping[str, Any],
) -> list[SweepCandidate]:
    """Build one stage's deterministic candidates from prior-stage winners."""
    if stage not in STAGE_ORDER:
        raise ValueError(f"Unknown sweep stage: {stage}")
    predecessor = STAGE_PREDECESSOR[stage]
    if predecessor and state.get("stages", {}).get(predecessor, {}).get("status") != "completed":
        raise RuntimeError(
            f"Stage {stage!r} requires completed stage {predecessor!r}"
        )
    config = state["configuration"]
    base = dict(config["base_hyperparameters"])
    candidates: list[SweepCandidate] = []
    for model_alias in config["models"]:
        for task in config["tasks"]:
            inherited = (
                _winner_parameters(state, predecessor, model_alias, task)
                if predecessor
                else dict(base)
            )
            variants: list[tuple[str, dict[str, Any]]] = []
            if stage == "target_modules":
                for label, target_modules in TARGET_MODULE_GRID:
                    parameters = dict(inherited)
                    parameters["target_modules"] = target_modules
                    variants.append((label, parameters))
            elif stage == "rank":
                for rank in RANK_GRID:
                    parameters = dict(inherited)
                    parameters["lora_rank"] = rank
                    parameters["lora_alpha"] = 2 * rank
                    variants.append((f"r_{rank}", parameters))
            elif stage == "learning_rate":
                for rate in LEARNING_RATE_GRID:
                    parameters = dict(inherited)
                    parameters["learning_rate"] = rate
                    variants.append((f"lr_{_format_float(rate)}", parameters))
            elif stage == "alpha":
                rank = int(inherited["lora_rank"])
                for multiplier in ALPHA_MULTIPLIER_GRID:
                    parameters = dict(inherited)
                    parameters["lora_alpha"] = multiplier * rank
                    variants.append((f"alpha_{multiplier}r", parameters))
            else:
                for dropout in DROPOUT_GRID:
                    parameters = dict(inherited)
                    parameters["lora_dropout"] = dropout
                    variants.append((f"dropout_{dropout:g}", parameters))
            candidates.extend(
                SweepCandidate(stage, model_alias, task, label, order, parameters)
                for order, (label, parameters) in enumerate(variants)
            )
    return candidates


def _register_stage(
    state: dict[str, Any], stage: str, candidates: Sequence[SweepCandidate]
) -> dict[str, Any]:
    records = {}
    for candidate in candidates:
        recipe_id = _recipe_id(
            candidate.model_alias, candidate.task, candidate.parameters, state
        )
        experiment = state["experiments"].setdefault(
            recipe_id,
            {
                "recipe_id": recipe_id,
                "model_alias": candidate.model_alias,
                "task": candidate.task,
                "parameters": _json_value(candidate.parameters),
                "origin_stage": stage,
                "status": "pending",
                "attempts": 0,
            },
        )
        if (
            experiment["model_alias"] != candidate.model_alias
            or experiment["task"] != candidate.task
            or experiment["parameters"] != _json_value(candidate.parameters)
        ):
            raise RuntimeError(f"Recipe hash collision: {recipe_id}")
        records[candidate.candidate_id] = {
            "candidate_id": candidate.candidate_id,
            "model_alias": candidate.model_alias,
            "task": candidate.task,
            "label": candidate.label,
            "order": candidate.order,
            "recipe_id": recipe_id,
            "parameters": _json_value(candidate.parameters),
            "reused_from_stage": (
                experiment["origin_stage"] if experiment["origin_stage"] != stage else None
            ),
        }
    existing = state["stages"].get(stage)
    if existing:
        if existing["candidates"] != records:
            raise ValueError(f"Saved {stage} candidate matrix does not match")
        return existing
    stage_state = {
        "status": "running",
        "created_at": _utc_now(),
        "candidates": records,
        "winners": {},
    }
    state["stages"][stage] = stage_state
    return stage_state


def _experiment_artifact_root(artifact_root: Path, recipe_id: str) -> Path:
    return artifact_root / "experiments" / recipe_id


def _adapter_target(
    artifact_root: Path, experiment: Mapping[str, Any]
) -> Path:
    spec = resolve_model(experiment["model_alias"])
    return (
        _experiment_artifact_root(artifact_root, experiment["recipe_id"])
        / "models"
        / "lora"
        / slugify_model_id(spec.model_id)
        / experiment["task"]
    )


def _adapter_matches(
    artifact_root: Path,
    experiment: Mapping[str, Any],
    state: Mapping[str, Any],
) -> bool:
    from LCD.shared.inference import SOURCE_PREFIXED_PREMISE_FORMAT
    from LCD.shared.lora import _prompt_processing_strategy

    target = _adapter_target(artifact_root, experiment)
    required = ("run_config.json", "adapter_config.json", "tokenizer_config.json")
    weights = ("adapter_model.safetensors", "adapter_model.bin")
    if not all((target / name).is_file() for name in required) or not any(
        (target / name).is_file() for name in weights
    ):
        return False
    try:
        manifest = json.loads((target / "run_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    task = experiment["task"]
    config = state["configuration"]
    expected = {
        "model_id": resolve_model(experiment["model_alias"]).model_id,
        "task": task,
        "resolved_revision": state["pins"]["model_revisions"][experiment["model_alias"]],
        "train_sha256": config["dataset_sha256"][f"train_{task}"],
        "validation_sha256": config["dataset_sha256"][f"val_{task}"],
        "prompt_sha256": config["prompt_sha256"][task],
        "prompt_processing": _prompt_processing_strategy(
            experiment["model_alias"]
        ),
        "premise_format": SOURCE_PREFIXED_PREMISE_FORMAT,
        "rag_revision": state["pins"]["rag_revision"],
        "hyperparameters": experiment["parameters"],
    }
    return all(manifest.get(key) == value for key, value in expected.items())


def _latest_checkpoint(trainer_dir: Path) -> Path | None:
    checkpoints = []
    if trainer_dir.is_dir():
        for path in trainer_dir.glob("checkpoint-*"):
            try:
                step = int(path.name.removeprefix("checkpoint-"))
            except ValueError:
                continue
            if path.is_dir():
                checkpoints.append((step, path))
    return max(checkpoints, default=(0, None))[1]


def _validate_experiment_scores(experiment: Mapping[str, Any], scores):
    required = {"task", "evaluation_scope", "test_dataset"}
    if not required.issubset(scores.columns):
        raise ValueError(f"Incomplete score columns for {experiment['recipe_id']}")
    expected = {
        (dataset, scope) for dataset in DATASETS for scope in BENCHMARK_SCOPES
    }
    benchmark = scores[scores["evaluation_scope"].isin(BENCHMARK_SCOPES)]
    observed = {
        (str(row.test_dataset), str(row.evaluation_scope))
        for row in benchmark.itertuples(index=False)
    }
    validation = scores[scores["evaluation_scope"] == "validation"]
    if observed != expected or len(validation) != 1 or len(scores) != 2:
        raise ValueError(f"Incomplete score rows for {experiment['recipe_id']}")
    if set(scores["task"].astype(str)) != {experiment["task"]}:
        raise ValueError(f"Wrong task in scores for {experiment['recipe_id']}")
    return scores


def _validate_ready_scores(model_alias: str, scores, tasks: Sequence[str]):
    required = {"model_alias", "task", "evaluation_scope", "test_dataset"}
    if not required.issubset(scores.columns):
        raise ValueError(f"Incomplete ready-LLM score columns for {model_alias}")
    model_scores = scores[scores["model_alias"] == model_alias]
    expected_rows = 2 * len(tasks)
    if len(model_scores) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} ready-LLM score rows for {model_alias}, "
            f"got {len(model_scores)}"
        )
    for task in tasks:
        task_scores = model_scores[model_scores["task"] == task]
        pseudo_experiment = {"recipe_id": model_alias, "task": task}
        _validate_experiment_scores(pseudo_experiment, task_scores)
    return model_scores


def _run_experiment(
    experiment: dict[str, Any],
    *,
    root: Path,
    artifact_root: Path,
    result_root: Path,
    state: Mapping[str, Any],
):
    from LCD.shared.lora import run as run_lora

    recipe_id = experiment["recipe_id"]
    experiment_results = result_root / "experiments" / recipe_id
    trainer_dir = experiment_results / "trainer"
    use_existing = _adapter_matches(artifact_root, experiment, state)
    checkpoint = None if use_existing else _latest_checkpoint(trainer_dir)
    return run_lora(
        experiment["model_alias"],
        experiment["task"],
        experiment["parameters"],
        train_path=root / "local" / "data" / "classification" / "train.csv",
        val_path=root / "local" / "data" / "classification" / "val.csv",
        rag_dir=_rag_path(root),
        rag_revision=state["pins"]["rag_revision"],
        artifact_root=_experiment_artifact_root(artifact_root, recipe_id),
        revision=state["pins"]["model_revisions"][experiment["model_alias"]],
        use_existing_model=use_existing,
        autotest_dir=root / "local" / "data" / "benchmarks" / "autotest",
        test_docx_dir=root / "local" / "data" / "benchmarks" / "test_docx",
        score_autotest=True,
        multiple_test=True,
        selected_datasets=DATASETS,
        benchmark_scopes=BENCHMARK_SCOPES,
        results_dir=experiment_results,
        trainer_output_dir=trainer_dir,
        resume_from_checkpoint=checkpoint,
    )


def _decorate_scores(scores, experiment: Mapping[str, Any], search_id: str):
    result = scores.copy()
    parameters = experiment["parameters"]
    fields = (
        ("search_id", search_id),
        ("recipe_id", experiment["recipe_id"]),
        ("model_alias", experiment["model_alias"]),
        ("lora_rank", parameters["lora_rank"]),
        ("lora_alpha", parameters["lora_alpha"]),
        ("target_modules", json.dumps(parameters["target_modules"])),
        ("learning_rate", parameters["learning_rate"]),
        ("lora_dropout", parameters["lora_dropout"]),
        ("completed_at", _utc_now()),
    )
    for column, value in reversed(fields):
        result.insert(0, column, value)
    return result


def _cleanup_cuda(logger: logging.Logger) -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except RuntimeError:
                pass
            logger.info(
                "CUDA cleanup: %.2f GiB allocated, %.2f GiB reserved",
                torch.cuda.memory_allocated() / 2**30,
                torch.cuda.memory_reserved() / 2**30,
            )
    except ModuleNotFoundError:
        pass


def _validation_row(scores, recipe_id: str):
    rows = scores[
        (scores["recipe_id"] == recipe_id)
        & (scores["evaluation_scope"] == "validation")
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one validation row for {recipe_id}")
    return rows.iloc[0]


def _metric(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _rank_and_finalize(
    state: dict[str, Any], stage: str, scores
) -> None:
    stage_state = state["stages"][stage]
    winners = {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for candidate in stage_state["candidates"].values():
        grouped.setdefault((candidate["model_alias"], candidate["task"]), []).append(candidate)
    for (model_alias, task), candidates in grouped.items():
        ranked = []
        for candidate in candidates:
            row = _validation_row(scores, candidate["recipe_id"])
            key = (
                -_metric(row.get("macro_f1"), -math.inf),
                -_metric(row.get("contradiction_f1"), -math.inf),
                _metric(row.get("invalid_predictions"), math.inf),
                int(candidate["order"]),
            )
            ranked.append((key, candidate, row))
        ranked.sort(key=lambda item: item[0])
        _, winner, row = ranked[0]
        winners[f"{model_alias}:{task}"] = {
            **winner,
            "validation_macro_f1": _metric(row.get("macro_f1"), -math.inf),
            "validation_contradiction_f1": _metric(
                row.get("contradiction_f1"), -math.inf
            ),
        }
    stage_state["winners"] = winners
    stage_state["status"] = "completed"
    stage_state["completed_at"] = _utc_now()


def _stage_frames(state: Mapping[str, Any], stage: str, scores):
    import pandas as pd

    candidates = pd.DataFrame(state["stages"][stage]["candidates"].values())
    if scores.empty:
        return candidates, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    expanded = candidates.merge(scores, on="recipe_id", how="left")
    validation = expanded[expanded["evaluation_scope"] == "validation"].copy()
    if not validation.empty:
        validation = validation.sort_values(
            [
                "model_alias_x",
                "task_x",
                "macro_f1",
                "contradiction_f1",
                "invalid_predictions",
                "order",
            ],
            ascending=[True, True, False, False, True, True],
            kind="stable",
        )
        validation["validation_rank"] = validation.groupby(
            ["model_alias_x", "task_x"], sort=False
        ).cumcount() + 1
    benchmark = expanded[
        expanded["evaluation_scope"].isin(BENCHMARK_SCOPES)
    ].copy()
    if not benchmark.empty:
        benchmark["scope_rank"] = benchmark.groupby(
            ["model_alias_x", "task_x", "test_dataset", "evaluation_scope"],
            sort=False,
            dropna=False,
        )["macro_f1"].rank(method="min", ascending=False)
    winners = pd.DataFrame(state["stages"][stage].get("winners", {}).values())
    return candidates, validation, benchmark, winners


def _write_stage_artifacts(
    state: Mapping[str, Any], stage: str, scores, result_root: Path
) -> None:
    import pandas as pd

    stage_dir = result_root / "stages" / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    candidates, validation, benchmark, winners = _stage_frames(state, stage, scores)
    stage_scores = (
        candidates.merge(scores, on="recipe_id", how="left")
        if not scores.empty
        else pd.DataFrame()
    )
    _atomic_csv(stage_dir / "scores.csv", stage_scores)
    experiments = pd.DataFrame(state["experiments"].values())
    failures = experiments[
        experiments.get("status", pd.Series(dtype=str)).isin(["failed", "interrupted"])
    ]
    temporary = stage_dir / ".results.tmp.xlsx"
    with pd.ExcelWriter(temporary, engine="openpyxl") as writer:
        stage_scores.to_excel(writer, sheet_name="scores", index=False)
        validation.to_excel(writer, sheet_name="validation_ranking", index=False)
        benchmark.to_excel(writer, sheet_name="benchmark_ranking", index=False)
        winners.to_excel(writer, sheet_name="winners", index=False)
        candidates.to_excel(writer, sheet_name="candidates", index=False)
        experiments.to_excel(writer, sheet_name="experiments", index=False)
        failures.to_excel(writer, sheet_name="failures", index=False)
    temporary.replace(stage_dir / "results.xlsx")


def _stage_complete(
    state: Mapping[str, Any], stage: str, scores
) -> bool:
    if scores.empty or "recipe_id" not in scores.columns:
        return False
    candidate_records = state["stages"][stage]["candidates"].values()
    for candidate in candidate_records:
        experiment = state["experiments"][candidate["recipe_id"]]
        if experiment["status"] != "completed":
            return False
        existing = scores[scores["recipe_id"] == candidate["recipe_id"]]
        try:
            _validate_experiment_scores(experiment, existing)
        except Exception:
            return False
    return True


def run_sweep_stage(
    stage: str,
    *,
    search_id: str = DEFAULT_SEARCH_ID,
    repo_root: str | Path | None = None,
    models: Sequence[str] = ALL_MODELS,
    tasks: Sequence[str] = DEFAULT_SWEEP_TASKS,
    hyperparameters: Mapping[str, Any] | None = None,
    max_attempts_per_run: int = 6,
    max_retries: int = 1,
    dry_run: bool = False,
):
    """Run or resume one stage of the coordinated local LoRA search."""
    if stage not in STAGE_ORDER:
        raise ValueError(f"Unknown sweep stage: {stage}")
    if max_attempts_per_run < 1:
        raise ValueError("max_attempts_per_run must be positive")
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")
    if not search_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in search_id):
        raise ValueError("search_id contains unsupported characters")
    _validate_scope(models, tasks)
    root = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else REPOSITORY_ROOT
    )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    base = _base_parameters(hyperparameters)
    _preflight(root, models, tasks, gpu=not dry_run)

    run_root = root / "local" / "experiments" / "lora" / "series_01_coordinate_search" / search_id
    result_root = run_root / "results"
    artifact_root = run_root / "artifacts"
    state_path = result_root / "search_state.json"
    configuration = _configuration(root, models, tasks, base)
    if dry_run and not state_path.is_file():
        if stage != "target_modules":
            raise RuntimeError(
                f"Dry-run for {stage} requires completed predecessor state"
            )
        mock_state = {
            "configuration": configuration,
            "stages": {},
            "experiments": {},
            "pins": {
                "model_revisions": {model: "dry-run" for model in models},
                "rag_revision": "dry-run",
            },
        }
        candidates = build_stage_candidates(stage, mock_state)
        print_stage_dry_run(stage, candidates, max_attempts_per_run)
        return candidates

    if not state_path.is_file() and dry_run:
        raise RuntimeError("Search state is unavailable")
    if not dry_run:
        result_root.mkdir(parents=True, exist_ok=True)
        artifact_root.mkdir(parents=True, exist_ok=True)
    state = _load_or_create_state(
        state_path,
        search_id=search_id,
        configuration=configuration,
        root=root,
    )
    candidates = build_stage_candidates(stage, state)
    if dry_run:
        print_stage_dry_run(stage, candidates, max_attempts_per_run, state=state)
        return candidates

    stage_state = _register_stage(state, stage, candidates)
    scores_path = result_root / "all_experiment_scores.csv"
    scores = _load_scores(scores_path)
    if not scores.empty and "recipe_id" not in scores.columns:
        raise ValueError(f"Invalid experiment score artifact: {scores_path}")
    logger = logging.getLogger(f"jura_lora_search.{search_id}.{stage}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(
            result_root / f"{stage}.log", encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    attempts_this_run = 0
    candidate_recipe_ids = list(
        dict.fromkeys(
            candidate["recipe_id"] for candidate in stage_state["candidates"].values()
        )
    )
    for recipe_id in candidate_recipe_ids:
        experiment = state["experiments"][recipe_id]
        existing = scores[scores["recipe_id"] == recipe_id] if not scores.empty else scores
        if experiment["status"] == "completed":
            try:
                _validate_experiment_scores(experiment, existing)
                continue
            except Exception:
                experiment["status"] = "interrupted"
                experiment["error"] = "Saved scores were missing or invalid"
        local_attempts = 0
        while (
            attempts_this_run < max_attempts_per_run
            and local_attempts <= max_retries
            and experiment["status"] != "completed"
        ):
            attempts_this_run += 1
            local_attempts += 1
            experiment["attempts"] = int(experiment.get("attempts", 0)) + 1
            experiment["status"] = "running"
            experiment["started_at"] = _utc_now()
            experiment["error"] = ""
            state["updated_at"] = _utc_now()
            _atomic_json(state_path, state)
            message = (
                f"Starting {recipe_id} ({experiment['model_alias']}/"
                f"{experiment['task']}), attempt {experiment['attempts']}"
            )
            print(message, flush=True)
            logger.info(message)
            started = time.monotonic()
            try:
                result = _run_experiment(
                    experiment,
                    root=root,
                    artifact_root=artifact_root,
                    result_root=result_root,
                    state=state,
                )
                result = _validate_experiment_scores(experiment, result)
                decorated = _decorate_scores(result, experiment, search_id)
                if not scores.empty:
                    scores = scores[scores["recipe_id"] != recipe_id]
                import pandas as pd

                scores = pd.concat([scores, decorated], ignore_index=True)
                experiment.update(
                    {
                        "status": "completed",
                        "completed_at": _utc_now(),
                        "duration_seconds": round(time.monotonic() - started, 3),
                        "error": "",
                    }
                )
                _atomic_csv(scores_path, scores)
                logger.info("Completed recipe %s", recipe_id)
            except Exception as error:
                experiment.update(
                    {
                        "status": "failed",
                        "failed_at": _utc_now(),
                        "duration_seconds": round(time.monotonic() - started, 3),
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    }
                )
                logger.exception("Failed recipe %s", recipe_id)
                print(
                    f"Failed {recipe_id}: {type(error).__name__}: {error}",
                    flush=True,
                )
            finally:
                state["updated_at"] = _utc_now()
                _atomic_json(state_path, state)
                _cleanup_cuda(logger)
        if attempts_this_run >= max_attempts_per_run:
            break
    if _stage_complete(state, stage, scores):
        _rank_and_finalize(state, stage, scores)
    state["updated_at"] = _utc_now()
    _atomic_json(state_path, state)
    _write_stage_artifacts(state, stage, scores, result_root)
    remaining = sum(
        state["experiments"][recipe_id]["status"] != "completed"
        for recipe_id in candidate_recipe_ids
    )
    if state["stages"][stage]["status"] == "completed":
        print(f"Stage {stage} complete. Results: {result_root / 'stages' / stage / 'results.xlsx'}")
    else:
        print(
            f"Stage {stage} paused after {attempts_this_run} attempt(s); "
            f"{remaining} unique experiment(s) remain. Rerun the same command."
        )
    return _stage_frames(state, stage, scores)[1]


def print_stage_dry_run(
    stage: str,
    candidates: Sequence[SweepCandidate],
    max_attempts_per_run: int,
    *,
    state: Mapping[str, Any] | None = None,
) -> None:
    rows = []
    seen: set[str] = set()
    existing = set(state.get("experiments", {})) if state is not None else set()
    for candidate in candidates:
        recipe_id = (
            _recipe_id(candidate.model_alias, candidate.task, candidate.parameters, state)
            if state is not None
            else "unresolved"
        )
        is_new = recipe_id == "unresolved" or (
            recipe_id not in existing and recipe_id not in seen
        )
        seen.add(recipe_id)
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "model": candidate.model_alias,
                "task": candidate.task,
                "label": candidate.label,
                "new_experiment": is_new,
                "status": (
                    state["experiments"].get(recipe_id, {}).get("status", "new")
                    if state is not None
                    else "unresolved"
                ),
            }
        )
    import pandas as pd

    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))
    logical = len(candidates)
    print(
        f"Dry run: {stage}, {logical} logical candidates; "
        f"at most {max_attempts_per_run} workflow attempts per invocation."
    )


def run_llm_comparison(
    *,
    search_id: str = DEFAULT_SEARCH_ID,
    repo_root: str | Path | None = None,
    max_attempts_per_run: int = 4,
    dry_run: bool = False,
):
    """Compare final dropout-stage LoRA winners with matched ready LLMs."""
    root = Path(repo_root or REPOSITORY_ROOT).expanduser().resolve()
    result_root = root / "local" / "experiments" / "lora" / "series_01_coordinate_search" / search_id / "results"
    state_path = result_root / "search_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Search state does not exist: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("stages", {}).get("dropout", {}).get("status") != "completed":
        raise RuntimeError("The dropout stage must be completed before comparison")
    models = state["configuration"]["models"]
    tasks = state["configuration"]["tasks"]
    current_configuration = _configuration(
        root,
        models,
        tasks,
        state["configuration"]["base_hyperparameters"],
    )
    if state["configuration"] != current_configuration:
        raise ValueError(
            "The search was created with different code, inputs, or settings"
        )
    if dry_run:
        print(f"Dry run: {len(models)} ready-LLM evaluations: {', '.join(models)}")
        return models
    if max_attempts_per_run < 1:
        raise ValueError("max_attempts_per_run must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _preflight(root, models, tasks, gpu=True)
    from LCD.shared.llm_evaluation import run_llm_evaluation
    import pandas as pd

    comparison_state = state.setdefault("comparison", {"models": {}})
    for record in comparison_state.get("models", {}).values():
        if record.get("status") == "running":
            record["status"] = "interrupted"
            record["error"] = "Previous process ended during this evaluation"
    ready_path = result_root / "comparison" / "ready_llm_scores.csv"
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    ready_scores = _load_scores(ready_path)
    if not ready_scores.empty and "model_alias" not in ready_scores.columns:
        raise ValueError(f"Invalid ready-LLM score artifact: {ready_path}")
    base = state["configuration"]["base_hyperparameters"]
    inference_parameters = {
        "batch_size": base["inference_batch_size"],
        "document_batch_size": base["document_batch_size"],
        "max_input_length": base["max_input_length"],
        "max_new_tokens": base["max_new_tokens"],
        "quantization": base["quantization"],
        "device_map": base["device_map"],
        "precision": base["precision"],
        "retrieval_top_k": base["retrieval_top_k"],
        "embedding_device": base["embedding_device"],
        "embedding_revision": base["embedding_revision"],
        "seed": base["seed"],
        "deterministic": base["deterministic"],
    }
    comparison_state["inference_parameters"] = _json_value(inference_parameters)
    attempts = 0
    for model_alias in models:
        record = comparison_state["models"].setdefault(
            model_alias, {"status": "pending", "attempts": 0}
        )
        if record["status"] == "completed" and not ready_scores.empty:
            existing = ready_scores[ready_scores["model_alias"] == model_alias]
            try:
                _validate_ready_scores(model_alias, existing, tasks)
                continue
            except ValueError:
                record["status"] = "interrupted"
                record["error"] = "Saved scores were missing or invalid"
        if attempts >= max_attempts_per_run:
            break
        attempts += 1
        record.update(
            {"status": "running", "attempts": int(record["attempts"]) + 1, "error": ""}
        )
        state["updated_at"] = _utc_now()
        _atomic_json(state_path, state)
        print(f"Starting matched ready-LLM evaluation: {model_alias}", flush=True)
        try:
            scores = run_llm_evaluation(
                model_alias,
                tasks=tasks,
                val_path=root / "local" / "data" / "classification" / "val.csv",
                rag_dir=_rag_path(root),
                rag_revision=state["pins"]["rag_revision"],
                revision=state["pins"]["model_revisions"][model_alias],
                autotest_dir=root / "local" / "data" / "benchmarks" / "autotest",
                test_docx_dir=root / "local" / "data" / "benchmarks" / "test_docx",
                score_autotest=True,
                multiple_test=True,
                selected_datasets=DATASETS,
                benchmark_scopes=BENCHMARK_SCOPES,
                results_dir=result_root / "comparison" / model_alias,
                inference_parameters=inference_parameters,
            )
            expected_rows = 2 * len(tasks)
            if len(scores) != expected_rows:
                raise ValueError(
                    f"Expected {expected_rows} ready-LLM score rows, got {len(scores)}"
                )
            scores = scores.copy()
            scores.insert(0, "model_alias", model_alias)
            _validate_ready_scores(model_alias, scores, tasks)
            if not ready_scores.empty:
                ready_scores = ready_scores[ready_scores["model_alias"] != model_alias]
            ready_scores = pd.concat([ready_scores, scores], ignore_index=True)
            _atomic_csv(ready_path, ready_scores)
            record.update({"status": "completed", "completed_at": _utc_now(), "error": ""})
        except Exception as error:
            record.update(
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            )
            print(
                f"Failed ready-LLM evaluation {model_alias}: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
        finally:
            state["updated_at"] = _utc_now()
            _atomic_json(state_path, state)
            _cleanup_cuda(logging.getLogger("jura_lora_comparison"))
    completed = all(
        comparison_state["models"].get(model, {}).get("status") == "completed"
        for model in models
    )
    if not completed:
        remaining = sum(
            comparison_state["models"].get(model, {}).get("status") != "completed"
            for model in models
        )
        print(f"Comparison paused; {remaining} ready-LLM evaluation(s) remain.")
        return ready_scores
    all_scores = _load_scores(result_root / "all_experiment_scores.csv")
    winner_records = list(state["stages"]["dropout"]["winners"].values())
    winner_ids = [record["recipe_id"] for record in winner_records]
    lora_scores = all_scores[all_scores["recipe_id"].isin(winner_ids)].copy()
    keys = ["model_alias", "task", "evaluation_scope", "test_dataset"]
    for frame in (ready_scores, lora_scores):
        frame["test_dataset"] = frame["test_dataset"].fillna("")
    metrics = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "contradiction_precision",
        "contradiction_recall",
        "contradiction_f1",
        "invalid_predictions",
    ]
    ready_columns = keys + [metric for metric in metrics if metric in ready_scores.columns]
    lora_columns = keys + [metric for metric in metrics if metric in lora_scores.columns]
    comparison = ready_scores[ready_columns].merge(
        lora_scores[lora_columns], on=keys, suffixes=("_llm", "_lora"), validate="one_to_one"
    )
    for metric in metrics:
        llm_column = f"{metric}_llm"
        lora_column = f"{metric}_lora"
        if llm_column in comparison and lora_column in comparison:
            comparison[f"{metric}_delta"] = comparison[lora_column] - comparison[llm_column]
    winners = pd.DataFrame(winner_records)
    output = result_root / "comparison" / "lora_vs_llm.xlsx"
    temporary = output.with_name(".lora_vs_llm.tmp.xlsx")
    with pd.ExcelWriter(temporary, engine="openpyxl") as writer:
        comparison.to_excel(writer, sheet_name="metric_deltas", index=False)
        lora_scores.to_excel(writer, sheet_name="winning_lora_scores", index=False)
        ready_scores.to_excel(writer, sheet_name="ready_llm_scores", index=False)
        winners.to_excel(writer, sheet_name="winning_recipes", index=False)
    temporary.replace(output)
    comparison_state["status"] = "completed"
    comparison_state["completed_at"] = _utc_now()
    _atomic_json(state_path, state)
    print(f"LoRA versus LLM comparison complete: {output}")
    return comparison
