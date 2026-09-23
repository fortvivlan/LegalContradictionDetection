"""Discover full-pipeline model artifacts or validate an explicit JSON list."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .common import validate_task

ModelFamily = Literal["bert", "base_llm", "lora"]
InferenceBackend = Literal["transformers", "llama_cpp"]


@dataclass(frozen=True)
class InferenceModel:
    """One model/task artifact consumed by the umbrella evaluator."""

    name: str
    family: ModelFamily
    task: str
    path_or_id: str
    revision: str | None = None
    base_model_path_or_id: str | None = None
    trust_remote_code: bool = False
    training_prompt_sha256: str | None = None
    training_premise_format: str | None = None
    backend: InferenceBackend = "transformers"


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _complete_weights(path: Path, names: tuple[str, ...]) -> bool:
    return any((path / name).is_file() for name in names)


def _relative_name(root: Path, path: Path, suffix: str) -> str:
    relative = path.relative_to(root).as_posix().replace("/", "__")
    return f"{relative}__{suffix}"


def discover_models(directory: str | Path) -> list[InferenceModel]:
    """Recursively discover BERT, LoRA, local causal, and referenced base models."""
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Model source is not a directory: {root}")
    models: list[InferenceModel] = []
    adapter_bases: dict[str, tuple[str | None, bool]] = {}
    local_causal_ids: dict[str, Path] = {}

    def active_files(filename: str):
        return sorted(
            path for path in root.rglob(filename)
            if ".legacy" not in path.relative_to(root).parts
        )

    for config_path in active_files("config.json"):
        artifact = config_path.parent
        config = _read_json(config_path)
        architectures = [str(value) for value in config.get("architectures", [])]
        if any("ForCausalLM" in value for value in architectures) and _complete_weights(
            artifact, ("model.safetensors", "pytorch_model.bin")
        ):
            model_id = str(config.get("_name_or_path") or artifact)
            local_causal_ids[model_id] = artifact

    for manifest_path in active_files("run_config.json"):
        artifact = manifest_path.parent
        manifest = _read_json(manifest_path)
        task_value = manifest.get("task")
        if task_value != "ternary":
            continue
        task = validate_task(str(task_value))
        if (artifact / "adapter_config.json").is_file() and _complete_weights(
            artifact, ("adapter_model.safetensors", "adapter_model.bin")
        ):
            adapter = _read_json(artifact / "adapter_config.json")
            base_id = str(
                adapter.get("base_model_name_or_path") or manifest.get("model_id") or ""
            )
            if not base_id:
                raise ValueError(f"LoRA has no base model identity: {artifact}")
            revision = manifest.get("resolved_revision")
            trust = bool(adapter.get("trust_remote_code", False))
            adapter_bases.setdefault(base_id, (str(revision) if revision else None, trust))
            models.append(
                InferenceModel(
                    name=_relative_name(root, artifact, task),
                    family="lora",
                    task=task,
                    path_or_id=str(artifact),
                    revision=str(revision) if revision else None,
                    base_model_path_or_id=str(local_causal_ids.get(base_id, base_id)),
                    trust_remote_code=trust,
                    training_prompt_sha256=manifest.get("prompt_sha256"),
                    training_premise_format=manifest.get("premise_format"),
                )
            )
        elif (artifact / "config.json").is_file() and _complete_weights(
            artifact, ("model.safetensors", "pytorch_model.bin")
        ):
            config = _read_json(artifact / "config.json")
            architectures = [str(value) for value in config.get("architectures", [])]
            if any("SequenceClassification" in value for value in architectures):
                models.append(
                    InferenceModel(
                        name=_relative_name(root, artifact, task),
                        family="bert",
                        task=task,
                        path_or_id=str(artifact),
                        revision=manifest.get("resolved_revision"),
                    )
                )

    for base_id, (revision, trust) in sorted(adapter_bases.items()):
        source = str(local_causal_ids.get(base_id, base_id))
        slug = base_id.replace("/", "_").replace(" ", "_")
        for task in ("ternary",):
            models.append(
                InferenceModel(
                    name=f"base__{slug}__{task}",
                    family="base_llm",
                    task=task,
                    path_or_id=source,
                    revision=revision,
                    trust_remote_code=trust,
                )
            )
    return _validate_unique(models)


def _from_json(path: Path) -> list[InferenceModel]:
    value = _read_json(path)
    entries = value.get("models")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path} must contain a nonempty models list")
    models: list[InferenceModel] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"models[{index}] must be an object")
        family = entry.get("family")
        if family not in {"bert", "base_llm", "lora"}:
            raise ValueError(f"models[{index}] has invalid family: {family!r}")
        backend = str(entry.get("backend", "transformers"))
        if backend not in {"transformers", "llama_cpp"}:
            raise ValueError(f"models[{index}] has invalid backend: {backend!r}")
        if backend == "llama_cpp" and family != "base_llm":
            raise ValueError(
                f"models[{index}] may use llama_cpp only with family='base_llm'"
            )
        tasks = entry.get("tasks") if family == "base_llm" else [entry.get("task")]
        if tasks is None:
            tasks = ["ternary"]
        if not isinstance(tasks, list) or not tasks:
            raise ValueError(f"models[{index}] must declare task(s)")
        source = entry.get("path") or entry.get("model_id")
        if not source:
            raise ValueError(f"models[{index}] requires path or model_id")
        if entry.get("path"):
            candidate = Path(str(source)).expanduser()
            source = str(candidate if candidate.is_absolute() else (path.parent / candidate).resolve())
        base_source = entry.get("base_model_path") or entry.get("base_model_id")
        if entry.get("base_model_path"):
            candidate = Path(str(base_source)).expanduser()
            base_source = str(candidate if candidate.is_absolute() else (path.parent / candidate).resolve())
        for task_value in tasks:
            task = validate_task(str(task_value))
            base_name = str(entry.get("name") or f"model-{index}")
            name = base_name if len(tasks) == 1 else f"{base_name}__{task}"
            models.append(
                InferenceModel(
                    name=name,
                    family=family,
                    task=task,
                    path_or_id=str(source),
                    backend=backend,  # type: ignore[arg-type]
                    revision=entry.get("revision"),
                    base_model_path_or_id=base_source,
                    trust_remote_code=bool(entry.get("trust_remote_code", False)),
                    training_prompt_sha256=entry.get("training_prompt_sha256"),
                    training_premise_format=entry.get("training_premise_format"),
                )
            )
    return _validate_unique(models)


def _validate_unique(models: list[InferenceModel]) -> list[InferenceModel]:
    if not models:
        raise ValueError("No complete model artifacts were discovered")
    names = [model.name for model in models]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate inference model names: {duplicates}")
    return models


def resolve_models_source(source: str | Path) -> list[InferenceModel]:
    """Resolve a recursive artifact directory or explicit model JSON file."""
    path = Path(source).expanduser().resolve()
    if path.is_dir():
        return discover_models(path)
    if path.is_file() and path.suffix.casefold() == ".json":
        return _from_json(path)
    raise ValueError(f"models source must be a directory or JSON file: {path}")


def write_resolved_models(models: list[InferenceModel], path: str | Path) -> Path:
    """Persist the exact jobs selected for an umbrella run."""
    target = Path(path)
    target.write_text(
        json.dumps({"models": [asdict(model) for model in models]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target
