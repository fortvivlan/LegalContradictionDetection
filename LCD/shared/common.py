"""Shared labels, model registry, dataset loading, seeding, and evaluation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

Task = Literal["ternary"]

TERNARY_LABELS = ("contradiction", "entailment", "not mentioned")
LABELS_BY_TASK: dict[Task, tuple[str, ...]] = {
    "ternary": TERNARY_LABELS,
}
LABEL2ID_BY_TASK: dict[Task, dict[str, int]] = {
    task: {label: index for index, label in enumerate(labels)}
    for task, labels in LABELS_BY_TASK.items()
}

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LOCAL_ROOT = REPOSITORY_ROOT / "local"
DATA_ROOT = LOCAL_ROOT / "data"
DEFAULT_RAG_REPOSITORY = "https://github.com/fortvivlan/dms-rag"
DEFAULT_RAG_DIR = (
    LOCAL_ROOT / "resources" / "dms-rag"
    if (LOCAL_ROOT / "resources" / "dms-rag").exists()
    else REPOSITORY_ROOT / "dms-rag"
)
DEFAULT_ARTIFACT_ROOT = LOCAL_ROOT / "experiments"
DEFAULT_RESULTS_DIR = LOCAL_ROOT / "experiments" / "shared_evaluation" / "results"
DEFAULT_AUTOTEST_DIR = DATA_ROOT / "benchmarks" / "autotest"
DEFAULT_TEST_DOCX_DIR = DATA_ROOT / "benchmarks" / "test_docx"
DEFAULT_BERT_REVISION = "89deeaa197d9d146e5763ac1f5fe32bf66817126"
DEFAULT_EMBEDDING_REVISION = DEFAULT_BERT_REVISION
DEFAULT_RAG_REVISION = "main"


@dataclass(frozen=True)
class ModelSpec:
    """Configuration for one supported causal language model."""

    alias: str
    model_id: str
    revision: str | None
    trust_remote_code: bool = False


MODEL_SPECS = (
    # Llama is gated, so its immutable commit is resolved with the user's HF
    # token at run start and then stored in the run manifest.
    ModelSpec("llama", "meta-llama/Llama-3.1-8B", None),
    ModelSpec(
        "ministral",
        "mistralai/Ministral-8B-Instruct-2410",
        "2f494a194c5b980dfb9772cb92d26cbb671fce5a",
    ),
    ModelSpec(
        "qwen",
        "Qwen/Qwen3-8B",
        "b968826d9c46dd6066d109eabc6255188de91218",
    ),
    ModelSpec(
        "t-lite",
        "t-tech/T-lite-it-2.1",
        "d125c970c553de58fcee3c937d5e4867d4a448d8",
    ),
)
_MODEL_BY_NAME = {
    name: spec
    for spec in MODEL_SPECS
    for name in (spec.alias.lower(), spec.model_id.lower())
}


def validate_task(task: str) -> Task:
    """Return a validated task name."""
    normalized = task.strip().lower()
    if normalized not in LABELS_BY_TASK:
        raise ValueError("task must be 'ternary'")
    return normalized  # type: ignore[return-value]


def resolve_model(model_name: str) -> ModelSpec:
    """Resolve a supported model alias or full Hugging Face model ID."""
    try:
        return _MODEL_BY_NAME[model_name.strip().lower()]
    except KeyError as error:
        supported = ", ".join(spec.alias for spec in MODEL_SPECS)
        raise ValueError(
            f"Unsupported model {model_name!r}; choose one of: {supported}"
        ) from error


def slugify_model_id(model_id: str) -> str:
    """Convert a model ID to a filesystem-safe stable slug."""
    return model_id.replace("/", "_").replace(" ", "_")


def default_dataset_path(split: str, task: Task) -> Path:
    """Return the repository-local CSV path for a split and task."""
    validate_task(task)
    return DATA_ROOT / "classification" / f"{split}.csv"


def load_dataset(path: str | Path, task: Task):
    """Load and validate a project CSV without changing row order."""
    import pandas as pd

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {path}")
    dataframe = pd.read_csv(path)
    columns = ("premise", "hypothesis", "source", "tag")
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    dataframe = dataframe.loc[:, columns].copy()
    training_columns = ("premise", "hypothesis", "tag")
    if dataframe[list(training_columns)].isna().any().any():
        raise ValueError(
            f"{path} contains missing premise, hypothesis, or tag values"
        )
    # ``source`` is useful audit metadata but is not a model input. The source
    # datasets contain one legitimately unlabeled source cell, so preserve the
    # example and use an empty string in exported result tables.
    dataframe["source"] = dataframe["source"].fillna("").astype(str)
    valid_labels = LABELS_BY_TASK[task]
    invalid = sorted(set(dataframe["tag"]) - set(valid_labels))
    if invalid:
        raise ValueError(f"{path} contains invalid {task} labels: {invalid}")
    dataframe.insert(
        0,
        "example_id",
        [f"{path.stem}:{index:06d}" for index in range(len(dataframe))],
    )
    return dataframe


def merge_parameters(
    defaults: Mapping[str, Any], overrides: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Overlay validated parameter overrides on defaults."""
    result = dict(defaults)
    if overrides is None:
        return result
    unknown = sorted(set(overrides) - set(defaults))
    if unknown:
        raise ValueError(f"Unknown hyperparameter(s): {', '.join(unknown)}")
    result.update(overrides)
    return result


def announce_stage(workflow: str, stage: str, message: str) -> None:
    """Print a consistent, immediately flushed workflow progress message."""
    print(f"[JURA][{workflow}][{stage.upper()}] {message}", flush=True)


def configure_reproducibility(seed: int, *, deterministic: bool = True) -> None:
    """Seed all RNGs and enforce deterministic PyTorch/CUDA execution.

    Strict deterministic mode raises if PyTorch encounters an operation for
    which it cannot provide a deterministic implementation. This is preferable
    to silently completing a run that cannot be reproduced.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    previous_cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if deterministic:
        # Must be set before the first cuBLAS workspace is initialized.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass
    try:
        import torch

        cuda_was_initialized = torch.cuda.is_initialized()
        if (
            deterministic
            and cuda_was_initialized
            and previous_cublas_config not in {":16:8", ":4096:8"}
        ):
            raise RuntimeError(
                "Strict deterministic mode was requested after CUDA was already "
                "initialized without a deterministic CUBLAS_WORKSPACE_CONFIG. "
                "Restart the runtime and call the workflow before using CUDA."
            )
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(deterministic, warn_only=False)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = deterministic
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("highest")
    except ModuleNotFoundError:
        pass


def set_random_seed(seed: int) -> None:
    """Backward-compatible alias for strict deterministic configuration."""
    configure_reproducibility(seed, deterministic=True)


def seed_data_loader_worker(worker_id: int) -> None:
    """Deterministically seed Python and NumPy inside a PyTorch data worker."""
    del worker_id
    import numpy as np
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_huggingface_revision(
    model_id: str,
    revision: str | None,
    *,
    token: str | None = None,
) -> str:
    """Resolve a model tag/branch to an immutable 40-character commit hash."""
    requested = revision or "main"
    if re.fullmatch(r"[0-9a-f]{40}", requested):
        return requested
    from huggingface_hub import model_info

    info = model_info(model_id, revision=requested, token=token)
    if not info.sha or not re.fullmatch(r"[0-9a-f]{40}", info.sha):
        raise RuntimeError(
            f"Could not resolve an immutable Hugging Face revision for {model_id}"
        )
    return info.sha


def source_tree_sha256() -> str:
    """Fingerprint executable project sources and the dependency lockfile."""
    paths = sorted((REPOSITORY_ROOT / "LCD").rglob("*.py"))
    paths.extend(
        path
        for path in (
            REPOSITORY_ROOT / "pyproject.toml",
            REPOSITORY_ROOT / "uv.lock",
        )
        if path.is_file()
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(REPOSITORY_ROOT).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(path.read_bytes())
    return digest.hexdigest()


def reproducibility_metadata(seed: int, *, deterministic: bool) -> dict[str, Any]:
    """Capture code, package, Python, OS, and accelerator run identity."""
    distributions = (
        "accelerate",
        "bitsandbytes",
        "faiss-cpu",
        "langchain-community",
        "langchain-huggingface",
        "numpy",
        "pandas",
        "peft",
        "scikit-learn",
        "sentence-transformers",
        "torch",
        "transformers",
    )
    versions: dict[str, str] = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    try:
        commit = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(REPOSITORY_ROOT), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit, dirty = "unavailable", True
    accelerator: dict[str, Any] = {}
    try:
        import torch

        accelerator = {
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "gpu_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        }
    except ModuleNotFoundError:
        accelerator = {"cuda_available": False}
    return {
        "seed": seed,
        "strict_determinism": deterministic,
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": versions,
        "repository_commit": commit,
        "repository_dirty": dirty,
        "source_tree_sha256": source_tree_sha256(),
        "accelerator": accelerator,
    }


def file_sha256(path: str | Path) -> str:
    """Return a SHA-256 checksum for run metadata."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_saved_artifact_manifest(
    artifact_dir: str | Path,
    *,
    required_files: Sequence[str],
    weight_files: Sequence[str],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a saved model/adapter directory and return its run manifest.

    Reuse is deliberately strict: incomplete or incompatible artifacts raise
    instead of triggering an expensive, unintended training run.
    """
    artifact_dir = Path(artifact_dir)
    if not artifact_dir.is_dir():
        raise FileNotFoundError(
            f"Previously trained artifact was requested but is absent: {artifact_dir}"
        )
    missing = [name for name in required_files if not (artifact_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Saved artifact at {artifact_dir} is incomplete; missing: "
            + ", ".join(missing)
        )
    if not any((artifact_dir / name).is_file() for name in weight_files):
        raise FileNotFoundError(
            f"Saved artifact at {artifact_dir} has no supported weight file "
            f"({', '.join(weight_files)})"
        )
    manifest_path = artifact_dir / "run_config.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"Cannot read saved artifact manifest: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"Saved artifact manifest is not a JSON object: {manifest_path}")
    mismatches = [
        f"{key}: saved={manifest.get(key)!r}, requested={value!r}"
        for key, value in expected.items()
        if manifest.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            f"Saved artifact at {artifact_dir} is incompatible: "
            + "; ".join(mismatches)
        )
    return manifest


def saved_artifact_revision(
    manifest: Mapping[str, Any], artifact_dir: str | Path
) -> str:
    """Return the immutable base revision recorded for a saved artifact."""
    revision = manifest.get("resolved_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(
            f"Saved artifact manifest has no immutable resolved revision: {artifact_dir}"
        )
    return revision


def prompt_sha256(prompt_text: str) -> str:
    """Return a stable prompt version identifier."""
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


@dataclass
class EvaluationTables:
    """Normalized validation outputs used by all experiment workflows."""

    scores: Any
    per_class: Any
    confusion_matrix: Any
    predictions: Any


def evaluate_predictions(
    dataframe,
    predictions: Sequence[str | None],
    raw_outputs: Sequence[str],
    *,
    model_id: str,
    task: Task,
) -> EvaluationTables:
    """Build score, per-class, confusion, and prediction tables."""
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    if len(dataframe) != len(predictions) or len(predictions) != len(raw_outputs):
        raise ValueError("Validation rows, predictions, and raw outputs must align")
    labels = LABELS_BY_TASK[task]
    gold = dataframe["tag"].tolist()
    normalized = [prediction if prediction in labels else "invalid" for prediction in predictions]
    macro = precision_recall_fscore_support(
        gold, normalized, labels=list(labels), average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        gold, normalized, labels=list(labels), average="weighted", zero_division=0
    )
    per_values = precision_recall_fscore_support(
        gold, normalized, labels=list(labels), average=None, zero_division=0
    )
    contradiction_index = labels.index("contradiction")
    summary = pd.DataFrame(
        [
            {
                "model": model_id,
                "task": task,
                "evaluation_scope": "validation",
                "support": len(gold),
                "accuracy": accuracy_score(gold, normalized),
                "macro_precision": macro[0],
                "macro_recall": macro[1],
                "macro_f1": macro[2],
                "weighted_f1": weighted[2],
                "contradiction_precision": per_values[0][contradiction_index],
                "contradiction_recall": per_values[1][contradiction_index],
                "contradiction_f1": per_values[2][contradiction_index],
                "invalid_predictions": normalized.count("invalid"),
                "rag_misses": 0,
            }
        ]
    )
    per_class = pd.DataFrame(
        [
            {
                "model": model_id,
                "task": task,
                "evaluation_scope": "validation",
                "label": label,
                "precision": per_values[0][index],
                "recall": per_values[1][index],
                "f1": per_values[2][index],
                "support": int(per_values[3][index]),
            }
            for index, label in enumerate(labels)
        ]
    )
    matrix_labels = list(labels) + (["invalid"] if "invalid" in normalized else [])
    matrix = confusion_matrix(gold, normalized, labels=matrix_labels)
    confusion = pd.DataFrame(
        [
            {
                "model": model_id,
                "task": task,
                "evaluation_scope": "validation",
                "gold_label": gold_label,
                "predicted_label": predicted_label,
                "count": int(matrix[row, column]),
            }
            for row, gold_label in enumerate(matrix_labels)
            for column, predicted_label in enumerate(matrix_labels)
        ]
    )
    result_rows = dataframe.copy()
    result_rows["prediction"] = normalized
    result_rows["raw_output"] = list(raw_outputs)
    result_rows["correct"] = result_rows["tag"] == result_rows["prediction"]
    result_rows.insert(1, "model", model_id)
    result_rows.insert(2, "task", task)
    return EvaluationTables(summary, per_class, confusion, result_rows)


def json_value(value: Any) -> str:
    """Serialize a metadata value consistently."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return str(value)
