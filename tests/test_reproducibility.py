import json
import os
import random
import re
from inspect import signature

import numpy as np
import pytest

from LCD.shared.common import (
    DEFAULT_BERT_REVISION,
    DEFAULT_EMBEDDING_REVISION,
    DEFAULT_RAG_REVISION,
    MODEL_SPECS,
    announce_stage,
    configure_reproducibility,
    load_saved_artifact_manifest,
    resolve_huggingface_revision,
    saved_artifact_revision,
    source_tree_sha256,
)
from LCD.shared.bert import run_bert_ternary
from LCD.shared.lora import run as run_lora


def test_reproducibility_configuration_repeats_rng_sequences() -> None:
    configure_reproducibility(123, deterministic=True)
    first = (random.random(), np.random.random())
    configure_reproducibility(123, deterministic=True)
    second = (random.random(), np.random.random())

    assert first == second
    assert os.environ["PYTHONHASHSEED"] == "123"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"


def test_stage_message_is_consistent_and_flushed(capsys) -> None:
    announce_stage("lora/ministral/ternary", "training", "Starting now.")

    assert capsys.readouterr().out == (
        "[JURA][lora/ministral/ternary][TRAINING] Starting now.\n"
    )


def test_model_revisions_are_pinned_and_rag_tracks_main() -> None:
    assert DEFAULT_BERT_REVISION == DEFAULT_EMBEDDING_REVISION
    assert re.fullmatch(r"[0-9a-f]{40}", DEFAULT_BERT_REVISION)
    assert DEFAULT_RAG_REVISION == "main"
    for spec in MODEL_SPECS:
        if spec.alias != "llama":
            assert spec.revision is not None
            assert re.fullmatch(r"[0-9a-f]{40}", spec.revision)


def test_exact_huggingface_revision_does_not_need_network() -> None:
    assert (
        resolve_huggingface_revision("any/model", DEFAULT_BERT_REVISION)
        == DEFAULT_BERT_REVISION
    )


def test_source_tree_fingerprint_is_stable() -> None:
    first = source_tree_sha256()
    second = source_tree_sha256()
    assert first == second
    assert re.fullmatch(r"[0-9a-f]{64}", first)


def test_saved_artifact_manifest_checks_identity(tmp_path) -> None:
    artifact = tmp_path / "model"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}", encoding="utf-8")
    (artifact / "model.safetensors").write_bytes(b"weights")
    manifest = {"model_id": "model", "task": "binary", "train_sha256": "abc"}
    (artifact / "run_config.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    loaded = load_saved_artifact_manifest(
        artifact,
        required_files=("run_config.json", "config.json"),
        weight_files=("model.safetensors", "pytorch_model.bin"),
        expected=manifest,
    )

    assert loaded == manifest


def test_saved_artifact_manifest_rejects_mismatch(tmp_path) -> None:
    artifact = tmp_path / "adapter"
    artifact.mkdir()
    (artifact / "adapter_config.json").write_text("{}", encoding="utf-8")
    (artifact / "adapter_model.safetensors").write_bytes(b"weights")
    (artifact / "run_config.json").write_text(
        json.dumps({"task": "binary"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="incompatible"):
        load_saved_artifact_manifest(
            artifact,
            required_files=("run_config.json", "adapter_config.json"),
            weight_files=("adapter_model.safetensors",),
            expected={"task": "ternary"},
        )


def test_saved_artifact_request_never_falls_back_when_missing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="requested but is absent"):
        load_saved_artifact_manifest(
            tmp_path / "missing",
            required_files=("run_config.json",),
            weight_files=("model.safetensors",),
            expected={"task": "binary"},
        )


def test_training_workflows_expose_opt_in_model_reuse() -> None:
    assert signature(run_bert_ternary).parameters["use_existing_model"].default is False
    assert signature(run_lora).parameters["use_existing_model"].default is False


def test_saved_artifact_requires_immutable_revision() -> None:
    with pytest.raises(ValueError, match="immutable resolved revision"):
        saved_artifact_revision({"resolved_revision": "main"}, "/drive/model")
