"""Run the complete local JURA experiment matrix with resumable checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

# Deterministic CUDA settings must be configured before importing torch/workflows.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ALL_MODELS = ("llama", "ministral", "qwen", "t-lite")
ALL_FAMILIES = ("bert", "ready_llm", "lora")
TASKS = ("ternary",)
DATASETS = ("Full",)
BENCHMARK_SCOPES = ("autotest_model", "autotest_total")


@dataclass(frozen=True)
class ExperimentJob:
    """One independently checkpointed experiment invocation."""

    family: str
    model_alias: str
    task: str | None = None

    @property
    def job_id(self) -> str:
        suffix = f":{self.task}" if self.task else ""
        return f"{self.family}:{self.model_alias}{suffix}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _find_repository_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "LCD"
        ).is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the LCD repository")


def build_jobs(
    *, families: Sequence[str] = ALL_FAMILIES, models: Sequence[str] = ALL_MODELS
) -> list[ExperimentJob]:
    """Build the deterministic sequential experiment matrix."""
    unknown_families = sorted(set(families) - set(ALL_FAMILIES))
    unknown_models = sorted(set(models) - set(ALL_MODELS))
    if unknown_families:
        raise ValueError("Unknown experiment families: " + ", ".join(unknown_families))
    if unknown_models:
        raise ValueError("Unknown model aliases: " + ", ".join(unknown_models))
    jobs: list[ExperimentJob] = []
    if "bert" in families:
        jobs.extend(ExperimentJob("bert", "bert", task) for task in TASKS)
    if "ready_llm" in families:
        jobs.extend(ExperimentJob("ready_llm", model) for model in models)
    if "lora" in families:
        jobs.extend(
            ExperimentJob("lora", model, task)
            for model in models
            for task in TASKS
        )
    job_ids = [job.job_id for job in jobs]
    if len(job_ids) != len(set(job_ids)):
        raise RuntimeError("Experiment matrix contains duplicate job IDs")
    return jobs


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    temporary.replace(path)


def _load_state(path: Path, campaign_id: str, jobs: Sequence[ExperimentJob]) -> dict[str, Any]:
    configured_job_ids = [job.job_id for job in jobs]
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("campaign_id") != campaign_id:
            raise ValueError(f"Campaign state does not match {campaign_id!r}: {path}")
        if state.get("configured_job_ids") != configured_job_ids:
            raise ValueError(
                "The campaign ID already belongs to a different job matrix"
            )
    else:
        state = {
            "campaign_id": campaign_id,
            "configured_job_ids": configured_job_ids,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "jobs": {},
        }
    configured_ids = set(configured_job_ids)
    existing_ids = set(state.get("jobs", {}))
    unexpected = sorted(existing_ids - configured_ids)
    if unexpected:
        raise ValueError(
            "The campaign ID already belongs to a different job matrix: "
            + ", ".join(unexpected)
        )
    for job in jobs:
        record = state["jobs"].setdefault(
            job.job_id,
            {
                "family": job.family,
                "model_alias": job.model_alias,
                "task": job.task,
                "status": "pending",
                "attempts": 0,
            },
        )
        if record.get("status") == "running":
            record["status"] = "interrupted"
            record["error"] = "Previous process ended while this job was running"
    return state


def _status_frame(state: dict[str, Any]):
    import pandas as pd

    rows = []
    for job_id, record in state["jobs"].items():
        row = {"job_id": job_id, **record}
        if row.get("traceback"):
            row["traceback"] = str(row["traceback"])[-30_000:]
        rows.append(row)
    return pd.DataFrame(rows)


def _write_score_artifacts(scores, state: dict[str, Any], result_dir: Path) -> None:
    import pandas as pd

    scores_path = result_dir / "all_experiment_scores.csv"
    temporary_csv = result_dir / ".all_experiment_scores.tmp.csv"
    scores.to_csv(temporary_csv, index=False, encoding="utf-8")
    temporary_csv.replace(scores_path)

    status = _status_frame(state)
    failures = status[status["status"].isin(["failed", "interrupted"])]
    benchmark = (
        scores[scores["evaluation_scope"].isin(BENCHMARK_SCOPES)].copy()
        if "evaluation_scope" in scores.columns
        else pd.DataFrame()
    )
    workbook_path = result_dir / "all_experiment_scores.xlsx"
    temporary_workbook = result_dir / ".all_experiment_scores.tmp.xlsx"
    with pd.ExcelWriter(temporary_workbook, engine="openpyxl") as writer:
        scores.to_excel(writer, sheet_name="scores", index=False)
        benchmark.to_excel(writer, sheet_name="benchmark_scores", index=False)
        status.to_excel(writer, sheet_name="job_status", index=False)
        failures.to_excel(writer, sheet_name="failures", index=False)
    temporary_workbook.replace(workbook_path)


def _load_score_artifact(path: Path):
    """Load resumable scores, treating a headerless empty artifact as no scores."""
    import pandas as pd

    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _validate_job_scores(job: ExperimentJob, scores):
    expected_tasks = TASKS if job.family == "ready_llm" else (job.task,)
    expected = {
        (task, dataset, scope)
        for task in expected_tasks
        for dataset in DATASETS
        for scope in BENCHMARK_SCOPES
    }
    benchmark = scores[scores["evaluation_scope"].isin(BENCHMARK_SCOPES)].copy()
    observed = {
        (str(row.task), str(row.test_dataset), str(row.evaluation_scope))
        for row in benchmark.itertuples(index=False)
    }
    validation_tasks = set(
        scores.loc[scores["evaluation_scope"] == "validation", "task"].astype(str)
    )
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"Incomplete benchmark scores for {job.job_id}; missing={missing}, extra={extra}"
        )
    if validation_tasks != set(expected_tasks):
        raise ValueError(
            f"Incomplete validation scores for {job.job_id}: {sorted(validation_tasks)}"
        )
    expected_rows = len(expected) + len(expected_tasks)
    if len(scores) != expected_rows:
        raise ValueError(
            f"Unexpected score row count for {job.job_id}: {len(scores)} != {expected_rows}"
        )
    return scores


def _artifact_complete(job: ExperimentJob, artifact_root: Path) -> bool:
    if job.family == "bert":
        target = artifact_root / "models" / "bert" / str(job.task)
        required = ("run_config.json", "config.json", "tokenizer_config.json")
        weights = ("model.safetensors", "pytorch_model.bin")
    elif job.family == "lora":
        from LCD.shared.common import resolve_model, slugify_model_id

        spec = resolve_model(job.model_alias)
        target = (
            artifact_root
            / "models"
            / "lora"
            / slugify_model_id(spec.model_id)
            / str(job.task)
        )
        required = ("run_config.json", "adapter_config.json", "tokenizer_config.json")
        weights = ("adapter_model.safetensors", "adapter_model.bin")
    else:
        return False
    return all((target / name).is_file() for name in required) and any(
        (target / name).is_file() for name in weights
    )


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
            allocated = torch.cuda.memory_allocated() / 2**30
            reserved = torch.cuda.memory_reserved() / 2**30
            logger.info(
                "CUDA after cleanup: %.2f GiB allocated, %.2f GiB reserved",
                allocated,
                reserved,
            )
    except ModuleNotFoundError:
        pass


def _run_job(
    job: ExperimentJob,
    *,
    repo_root: Path,
    artifact_root: Path,
    result_dir: Path,
    retry_mode: bool,
    use_existing_model: bool,
):
    rag_dir = repo_root / "local" / "resources" / "dms-rag"
    if not rag_dir.exists():
        rag_dir = repo_root / "dms-rag"
    common = {
        "rag_dir": rag_dir,
        "autotest_dir": repo_root / "local" / "data" / "benchmarks" / "autotest",
        "test_docx_dir": repo_root / "local" / "data" / "benchmarks" / "test_docx",
        "score_autotest": True,
        "multiple_test": True,
        "results_dir": result_dir,
    }
    if job.family == "bert":
        from LCD.shared.bert import run_bert_ternary

        parameters = {
            "batch_size": 8 if retry_mode else 16,
            "inference_batch_size": 16 if retry_mode else 32,
            "gradient_checkpointing": retry_mode,
        }
        runner = run_bert_ternary
        return runner(
            train_path=repo_root / "local" / "data" / "classification" / "train.csv",
            val_path=repo_root / "local" / "data" / "classification" / "val.csv",
            artifact_root=artifact_root,
            use_existing_model=use_existing_model,
            hyperparameters=parameters,
            **common,
        )
    if job.family == "ready_llm":
        from LCD.shared.llm_evaluation import run_llm_evaluation

        return run_llm_evaluation(
            job.model_alias,
            val_path=repo_root / "local" / "data" / "classification" / "val.csv",
            inference_parameters={
                "batch_size": 1,
                "document_batch_size": 1 if retry_mode else 4,
                "quantization": True,
                "embedding_device": "cpu",
            },
            **common,
        )
    from LCD.shared.lora import run as run_lora

    return run_lora(
        job.model_alias,
        str(job.task),
        {
            "batch_size": 1,
            "gradient_accumulation_steps": 16,
            "gradient_checkpointing": True,
            "quantization": True,
            "inference_batch_size": 1,
            "document_batch_size": 1 if retry_mode else 4,
            "embedding_device": "cpu",
        },
        train_path=repo_root / "local" / "data" / "classification" / "train.csv",
        val_path=repo_root / "local" / "data" / "classification" / "val.csv",
        artifact_root=artifact_root,
        use_existing_model=use_existing_model,
        **common,
    )


def _preflight(repo_root: Path, jobs: Sequence[ExperimentJob], *, require_gpu: bool) -> None:
    required_files = [
        repo_root / "local" / "data" / "classification" / f"{split}.csv"
        for split in ("train", "val")
    ]
    rag_dir = repo_root / "local" / "resources" / "dms-rag"
    if not rag_dir.exists():
        rag_dir = repo_root / "dms-rag"
    required_files.extend(
        [
            rag_dir / "codex.csv",
            rag_dir / "faiss_index" / "index.faiss",
            rag_dir / "faiss_index" / "index.pkl",
        ]
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing experiment artifacts: " + ", ".join(missing))

    from LCD.shared.autotest_scoring import discover_autotest_datasets

    datasets = discover_autotest_datasets(
        repo_root / "local" / "data" / "benchmarks" / "autotest", repo_root / "local" / "data" / "benchmarks" / "test_docx", multiple_test=True
    )
    counts = {dataset.name: len(dataset.documents) for dataset in datasets}
    if counts != {"Full": 30}:
        raise ValueError(f"Unexpected benchmark composition: {counts}")

    if require_gpu and any(job.model_alias == "llama" for job in jobs) and not (
        os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    ):
        raise RuntimeError(
            "Llama jobs require HF_TOKEN or HUGGING_FACE_HUB_TOKEN in the environment"
        )
    if require_gpu:
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


def _decorate_scores(scores, job: ExperimentJob, campaign_id: str, attempt: int):
    result = scores.copy()
    completed_at = _utc_now()
    for column, value in reversed(
        (
            ("campaign_id", campaign_id),
            ("job_id", job.job_id),
            ("experiment_family", job.family),
            ("model_alias", job.model_alias),
            ("attempt", attempt),
            ("completed_at", completed_at),
        )
    ):
        result.insert(0, column, value)
    return result


def _validate_campaign(scores, jobs: Sequence[ExperimentJob]) -> None:
    completed_ids = set(scores["job_id"].astype(str)) if not scores.empty else set()
    expected_ids = {job.job_id for job in jobs}
    if completed_ids != expected_ids:
        raise RuntimeError(
            f"Campaign scores are incomplete; missing={sorted(expected_ids - completed_ids)}"
        )
    expected_total = sum(10 if job.family == "ready_llm" else 5 for job in jobs)
    expected_benchmark = sum(8 if job.family == "ready_llm" else 4 for job in jobs)
    benchmark_count = int(scores["evaluation_scope"].isin(BENCHMARK_SCOPES).sum())
    if len(scores) != expected_total or benchmark_count != expected_benchmark:
        raise RuntimeError(
            "Campaign score cardinality mismatch: "
            f"total={len(scores)}/{expected_total}, "
            f"benchmark={benchmark_count}/{expected_benchmark}"
        )


def run_all_experiments(
    *,
    campaign_id: str = "full_pipeline_v1",
    repo_root: str | Path | None = None,
    families: Sequence[str] = ALL_FAMILIES,
    models: Sequence[str] = ALL_MODELS,
    max_retries: int = 1,
    dry_run: bool = False,
):
    """Run or resume all selected local experiments and return consolidated scores."""
    import pandas as pd

    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", campaign_id):
        raise ValueError(
            "campaign_id must contain only letters, digits, dots, underscores, and hyphens"
        )
    root = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else _find_repository_root(Path(__file__).resolve())
    )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    jobs = build_jobs(families=families, models=models)
    _preflight(root, jobs, require_gpu=not dry_run)
    if dry_run:
        frame = pd.DataFrame(
            [
                {
                    "position": index,
                    "job_id": job.job_id,
                    "family": job.family,
                    "model_alias": job.model_alias,
                    "task": job.task or "ternary",
                    "expected_score_rows": 3,
                }
                for index, job in enumerate(jobs, start=1)
            ]
        )
        print(frame.to_string(index=False))
        print(
            f"Dry run: {len(jobs)} jobs, "
            f"{frame['expected_score_rows'].sum()} expected score rows"
        )
        return frame

    artifact_root = root / "local" / "experiments" / "baselines" / "series_01_classifiers" / "campaigns" / campaign_id / "artifacts"
    result_dir = root / "local" / "experiments" / "baselines" / "series_01_classifiers" / "campaigns" / campaign_id / "results"
    artifact_root.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"jura_campaign.{campaign_id}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(result_dir / "campaign.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)

    state_path = result_dir / "campaign_state.json"
    state = _load_state(state_path, campaign_id, jobs)
    state["updated_at"] = _utc_now()
    _atomic_json(state_path, state)
    scores_path = result_dir / "all_experiment_scores.csv"
    scores = _load_score_artifact(scores_path)

    for position, job in enumerate(jobs, start=1):
        record = state["jobs"][job.job_id]
        if record["status"] == "completed":
            existing = scores[scores["job_id"] == job.job_id] if not scores.empty else scores
            try:
                _validate_job_scores(job, existing)
                logger.info("Skipping completed job %s", job.job_id)
                continue
            except Exception:
                record["status"] = "interrupted"
                record["error"] = "Saved scores were missing or invalid"

        logger.info("Starting job %d/%d: %s", position, len(jobs), job.job_id)
        succeeded = False
        prior_attempts = int(record.get("attempts", 0))
        for local_attempt in range(max_retries + 1):
            retry_mode = local_attempt > 0
            attempt = prior_attempts + local_attempt + 1
            reuse = (prior_attempts > 0 or retry_mode) and _artifact_complete(
                job, artifact_root
            )
            record.update(
                {
                    "status": "running",
                    "attempts": attempt,
                    "started_at": _utc_now(),
                    "use_existing_model": reuse,
                    "error": "",
                }
            )
            state["updated_at"] = _utc_now()
            _atomic_json(state_path, state)
            started = time.monotonic()
            try:
                job_scores = _run_job(
                    job,
                    repo_root=root,
                    artifact_root=artifact_root,
                    result_dir=result_dir,
                    retry_mode=retry_mode,
                    use_existing_model=reuse,
                )
                job_scores = _validate_job_scores(job, job_scores)
                decorated = _decorate_scores(job_scores, job, campaign_id, attempt)
                if not scores.empty:
                    scores = scores[scores["job_id"] != job.job_id]
                scores = pd.concat([scores, decorated], ignore_index=True)
                record.update(
                    {
                        "status": "completed",
                        "completed_at": _utc_now(),
                        "duration_seconds": round(time.monotonic() - started, 3),
                        "error": "",
                    }
                )
                state["updated_at"] = _utc_now()
                _write_score_artifacts(scores, state, result_dir)
                _atomic_json(state_path, state)
                succeeded = True
                logger.info("Completed %s", job.job_id)
                break
            except Exception as error:
                record.update(
                    {
                        "status": "failed",
                        "failed_at": _utc_now(),
                        "duration_seconds": round(time.monotonic() - started, 3),
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    }
                )
                state["updated_at"] = _utc_now()
                _atomic_json(state_path, state)
                logger.exception("Attempt %d failed for %s", attempt, job.job_id)
            finally:
                _cleanup_cuda(logger)
        if not succeeded:
            logger.error("Job exhausted retries: %s", job.job_id)
        _write_score_artifacts(scores, state, result_dir)

    failed = [
        job_id
        for job_id, record in state["jobs"].items()
        if record["status"] != "completed"
    ]
    if failed:
        raise RuntimeError(
            "Campaign finished with failed jobs; rerun the same command to resume: "
            + ", ".join(failed)
        )
    _validate_campaign(scores, jobs)
    state["status"] = "completed"
    state["completed_at"] = _utc_now()
    state["updated_at"] = _utc_now()
    _write_score_artifacts(scores, state, result_dir)
    _atomic_json(state_path, state)
    logger.info("Campaign complete: %d jobs, %d score rows", len(jobs), len(scores))
    print(f"Campaign complete. Results: {result_dir / 'all_experiment_scores.xlsx'}")
    return scores


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default="full_pipeline_v1")
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--families", nargs="+", choices=ALL_FAMILIES, default=ALL_FAMILIES)
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    run_all_experiments(
        campaign_id=arguments.campaign_id,
        repo_root=arguments.repo_root,
        families=arguments.families,
        models=arguments.models,
        max_retries=arguments.max_retries,
        dry_run=arguments.dry_run,
    )
