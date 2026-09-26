"""Retrain the four ternary baseline LoRAs with complete training context."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from LCD.shared.common import (
    DEFAULT_AUTOTEST_DIR,
    DEFAULT_RAG_DIR,
    DEFAULT_TEST_DOCX_DIR,
    REPOSITORY_ROOT,
    file_sha256,
    merge_parameters,
    prompt_sha256,
    resolve_model,
    slugify_model_id,
)
from LCD.shared.inference import SOURCE_PREFIXED_PREMISE_FORMAT
from LCD.shared.lora import DEFAULT_LORA_HYPERPARAMETERS, _prompt_processing_strategy
from LCD.shared.prompting import prompt_for_task

MODELS = ("llama", "ministral", "qwen", "t-lite")
MODEL_REVISIONS = {
    "llama": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
    "ministral": "2f494a194c5b980dfb9772cb92d26cbb671fce5a",
    "qwen": "b968826d9c46dd6066d109eabc6255188de91218",
    "t-lite": "d125c970c553de58fcee3c937d5e4867d4a448d8",
}
RAG_REVISION = "e6eab944161e1266c1b4452f172a9a725b1abe97"
RUN_ID = "baseline_full_context_v1"


def baseline_lora_parameters(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the original baseline adapter recipe with complete-context defaults."""
    parameters = merge_parameters(DEFAULT_LORA_HYPERPARAMETERS, {
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "gradient_checkpointing": True,
        "quantization": True,
        "inference_batch_size": 1,
        "document_batch_size": 4,
        "embedding_device": "cpu",
    })
    return merge_parameters(parameters, overrides)


def _job_config(
    alias: str,
    *,
    train_path: Path,
    val_path: Path,
    revision: str,
    rag_revision: str,
    parameters: Mapping[str, Any],
    rag_dir: Path | None = None,
    autotest_dir: Path | None = None,
    test_docx_dir: Path | None = None,
) -> dict[str, Any]:
    spec = resolve_model(alias)
    return {
        "model_alias": alias,
        "model_id": spec.model_id,
        "task": "ternary",
        "revision": revision,
        "rag_revision": rag_revision,
        "rag_dir": str(rag_dir) if rag_dir is not None else None,
        "train_path": str(train_path),
        "train_sha256": file_sha256(train_path),
        "val_path": str(val_path),
        "validation_sha256": file_sha256(val_path),
        "prompt_sha256": prompt_sha256(prompt_for_task("ternary")),
        "prompt_processing": _prompt_processing_strategy(alias),
        "premise_format": SOURCE_PREFIXED_PREMISE_FORMAT,
        "hyperparameters": dict(parameters),
        "test_dataset": "Full",
        "evaluation_scope": "autotest_model",
        "autotest_full_sha256": _tree_sha256(autotest_dir / "Full") if autotest_dir else None,
        "test_docx_full_sha256": _tree_sha256(test_docx_dir / "Full") if test_docx_dir else None,
    }


def _tree_sha256(directory: Path) -> str | None:
    """Fingerprint one local benchmark folder without opening private rows."""
    if not directory.is_dir():
        return None
    digest = hashlib.sha256()
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files:
        return None
    for path in files:
        relative = path.relative_to(directory).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(file_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _validate_scores(scores, alias: str) -> None:
    """Reject missing or extra validation and Full/model score rows."""
    if len(scores) != 2 or set(scores["task"]) != {"ternary"}:
        raise ValueError(f"Incomplete validation or Full/model scores for {alias}")
    validation = scores.loc[scores["evaluation_scope"].eq("validation")]
    benchmark = scores.loc[
        scores["evaluation_scope"].eq("autotest_model")
        & scores["test_dataset"].eq("Full")
    ]
    if len(validation) != 1 or len(benchmark) != 1:
        raise ValueError(f"Incomplete validation or Full/model scores for {alias}")


def _check_adapter(target: Path, config: Mapping[str, Any]) -> bool:
    """Return whether a compatible adapter exists; reject conflicting artifacts."""
    if not target.exists():
        return False
    manifest_path = target / "run_config.json"
    if not manifest_path.is_file() or not (target / "adapter_config.json").is_file() or not any(
        (target / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise ValueError(f"Incomplete existing adapter: {target}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "model_id": config["model_id"],
        "task": config["task"],
        "resolved_revision": config["revision"],
        "train_sha256": config["train_sha256"],
        "validation_sha256": config["validation_sha256"],
        "prompt_sha256": config["prompt_sha256"],
        "prompt_processing": config["prompt_processing"],
        "premise_format": config["premise_format"],
        "rag_requested_revision": config["rag_revision"],
        "hyperparameters": config["hyperparameters"],
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Existing adapter uses a different recipe or input: {target}")
    return True


def run_baseline_lora_retraining(
    *,
    models: Sequence[str] = MODELS,
    run_id: str = RUN_ID,
    repo_root: str | Path = REPOSITORY_ROOT,
    train_path: str | Path | None = None,
    val_path: str | Path | None = None,
    rag_dir: str | Path = DEFAULT_RAG_DIR,
    rag_revision: str = RAG_REVISION,
    autotest_dir: str | Path = DEFAULT_AUTOTEST_DIR,
    test_docx_dir: str | Path = DEFAULT_TEST_DOCX_DIR,
    output_root: str | Path | None = None,
    model_revisions: Mapping[str, str] | None = None,
    hyperparameters: Mapping[str, Any] | None = None,
    dry_run: bool = False,
):
    """Train isolated baseline adapters and score validation and Full/model pairs.

    Completed jobs are reused only when their full configuration and adapter
    manifest match. A failed evaluation can resume from its completed adapter.
    No historical adapters or checkpoints are loaded.
    """
    import pandas as pd

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("run_id must be a single nonempty directory name")
    if not models or len(set(models)) != len(models) or set(models) - set(MODELS):
        raise ValueError(f"models must be unique members of {MODELS}")
    root = Path(repo_root).expanduser().resolve()

    def path(value: str | Path) -> Path:
        candidate = Path(value).expanduser()
        return (candidate if candidate.is_absolute() else root / candidate).resolve()

    train = path(train_path or "local/data/classification/train.csv")
    val = path(val_path or "local/data/classification/val.csv")
    rag = path(rag_dir)
    autotest = path(autotest_dir)
    documents = path(test_docx_dir)
    output = path(output_root or "local/experiments/baselines/series_01_classifiers/lora_retraining") / run_id
    revisions = dict(MODEL_REVISIONS)
    if model_revisions:
        if set(model_revisions) - set(MODELS):
            raise ValueError("Unknown model revision alias")
        revisions.update(model_revisions)
    parameters = baseline_lora_parameters(hyperparameters)
    jobs = [
        _job_config(alias, train_path=train, val_path=val, revision=revisions[alias],
                    rag_revision=rag_revision, parameters=parameters, rag_dir=rag,
                    autotest_dir=autotest, test_docx_dir=documents)
        for alias in models
    ]
    if dry_run:
        frame = pd.DataFrame(jobs)
        print(frame[["model_alias", "model_id", "revision", "test_dataset", "evaluation_scope"]].to_string(index=False))
        return frame

    if any(job["autotest_full_sha256"] is None or job["test_docx_full_sha256"] is None for job in jobs):
        raise FileNotFoundError("Full benchmark folders are missing or empty")

    from LCD.shared.lora import run as run_lora

    score_frames = []
    for config in jobs:
        alias = config["model_alias"]
        artifact_root = output / "artifacts" / alias
        result_dir = output / "results" / alias
        target = artifact_root / "models" / "lora" / slugify_model_id(config["model_id"]) / "ternary"
        saved = _check_adapter(target, config)
        score_path = result_dir / "scores.csv"
        config_path = result_dir / "job_config.json"
        if config_path.exists():
            recorded = json.loads(config_path.read_text(encoding="utf-8"))
            if recorded != config:
                raise ValueError(f"Existing results use a different configuration: {result_dir}")
        if score_path.exists():
            if not config_path.exists() or not saved:
                raise ValueError(f"Cannot verify existing scores: {score_path}")
            scores = pd.read_csv(score_path)
        else:
            result_dir.mkdir(parents=True, exist_ok=True)
            scores = run_lora(
                alias, "ternary", parameters,
                train_path=train, val_path=val, rag_dir=rag,
                rag_revision=rag_revision, artifact_root=artifact_root,
                revision=config["revision"], use_existing_model=saved,
                autotest_dir=autotest, test_docx_dir=documents,
                multiple_test=True, selected_datasets=("Full",),
                benchmark_scopes=("autotest_model",), results_dir=result_dir,
                trainer_output_dir=output / "trainers" / alias,
            )
            _validate_scores(scores, alias)
            temporary_config = config_path.with_suffix(".tmp")
            temporary_config.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary_config.replace(config_path)
            temporary_scores = score_path.with_suffix(".tmp")
            scores.to_csv(temporary_scores, index=False)
            temporary_scores.replace(score_path)
        _validate_scores(scores, alias)
        frame = scores.copy()
        frame.insert(0, "model_alias", alias)
        frame.insert(0, "run_id", run_id)
        score_frames.append(frame)
    combined = pd.concat(score_frames, ignore_index=True)
    results_root = output / "results"
    combined_path = results_root / "all_lora_scores.csv"
    temporary_combined = combined_path.with_suffix(".tmp")
    combined.to_csv(temporary_combined, index=False)
    temporary_combined.replace(combined_path)
    return combined


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--train-path", type=Path)
    parser.add_argument("--val-path", type=Path)
    parser.add_argument("--rag-dir", type=Path, default=DEFAULT_RAG_DIR)
    parser.add_argument("--rag-revision", default=RAG_REVISION)
    parser.add_argument("--autotest-dir", type=Path, default=DEFAULT_AUTOTEST_DIR)
    parser.add_argument("--test-docx-dir", type=Path, default=DEFAULT_TEST_DOCX_DIR)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--revisions-json", type=Path)
    parser.add_argument("--hyperparameters-json", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_baseline_lora_retraining(
        models=args.models, run_id=args.run_id, repo_root=args.repo_root,
        train_path=args.train_path, val_path=args.val_path, rag_dir=args.rag_dir,
        rag_revision=args.rag_revision, autotest_dir=args.autotest_dir,
        test_docx_dir=args.test_docx_dir, output_root=args.output_root,
        model_revisions=(json.loads(args.revisions_json.read_text(encoding="utf-8")) if args.revisions_json else None),
        hyperparameters=(json.loads(args.hyperparameters_json.read_text(encoding="utf-8")) if args.hyperparameters_json else None),
        dry_run=args.dry_run,
    )
