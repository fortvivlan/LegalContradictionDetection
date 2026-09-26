"""Small, offline checks for full-context baseline retraining and comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from LCD.experiments.baselines.series_01_classifiers.compare_lora_retraining import (
    compile_lora_retraining_comparison,
)
from LCD.experiments.baselines.series_01_classifiers.retrain_lora import (
    MODELS,
    _check_adapter,
    _job_config,
    baseline_lora_parameters,
    run_baseline_lora_retraining,
)
from LCD.shared.common import resolve_model, slugify_model_id


def _score(**values):
    return {
        "task": "ternary", "evaluation_scope": "autotest_model",
        "test_dataset": "Full", "support": 1207, "accuracy": 0.8,
        "macro_f1": 0.7, "contradiction_precision": 0.6,
        "contradiction_recall": 0.5, "contradiction_f1": 0.55,
        "invalid_predictions": 0, "rag_misses": 0,
        **values,
    }


def test_baseline_retraining_dry_run_uses_full_context_and_isolated_recipe(tmp_path: Path) -> None:
    data = tmp_path / "local/data/classification"
    data.mkdir(parents=True)
    (data / "train.csv").write_text("train", encoding="utf-8")
    (data / "val.csv").write_text("val", encoding="utf-8")

    planned = run_baseline_lora_retraining(repo_root=tmp_path, dry_run=True)

    assert planned["model_alias"].tolist() == list(MODELS)
    assert all(planned["hyperparameters"].map(lambda x: x["max_seq_length"] == 2560))
    assert all(planned["hyperparameters"].map(lambda x: x["batch_size"] == 1))
    assert all(planned["hyperparameters"].map(lambda x: x["gradient_accumulation_steps"] == 16))
    assert set(planned["evaluation_scope"]) == {"autotest_model"}


def test_existing_adapter_must_match_full_training_recipe(tmp_path: Path) -> None:
    train, val = tmp_path / "train.csv", tmp_path / "val.csv"
    train.write_text("train", encoding="utf-8")
    val.write_text("val", encoding="utf-8")
    parameters = baseline_lora_parameters()
    config = _job_config(
        "qwen", train_path=train, val_path=val, revision="a" * 40,
        rag_revision="b" * 40, parameters=parameters,
    )
    target = tmp_path / "adapter"
    assert not _check_adapter(target, config)
    target.mkdir()
    (target / "adapter_config.json").write_text("{}", encoding="utf-8")
    (target / "adapter_model.safetensors").write_bytes(b"fixture")
    manifest = {
        "model_id": config["model_id"], "task": config["task"],
        "resolved_revision": config["revision"],
        "train_sha256": config["train_sha256"],
        "validation_sha256": config["validation_sha256"],
        "prompt_sha256": config["prompt_sha256"],
        "prompt_processing": config["prompt_processing"],
        "premise_format": config["premise_format"],
        "rag_requested_revision": config["rag_revision"],
        "hyperparameters": config["hyperparameters"],
    }
    (target / "run_config.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert _check_adapter(target, config)
    changed = dict(config)
    changed["hyperparameters"] = dict(parameters, max_seq_length=1024)
    with pytest.raises(ValueError, match="different recipe"):
        _check_adapter(target, changed)


def test_comparison_selects_only_balanced_grid_winners_and_labels_support(tmp_path: Path) -> None:
    new_root = tmp_path / "new"
    legacy_rows, campaign_rows, grid_rows = [], [], []
    winners = {}
    for alias in MODELS:
        model_id = resolve_model(alias).model_id
        slug = slugify_model_id(model_id)
        job = new_root / alias
        job.mkdir(parents=True)
        pd.DataFrame([_score()]).to_csv(job / "scores.csv", index=False)
        (job / "job_config.json").write_text(json.dumps({
            "model_alias": alias, "model_id": model_id,
            "hyperparameters": {"max_seq_length": 2560},
        }), encoding="utf-8")
        legacy_rows.append(_score(model_family="lora", model_name=f"models__lora__{slug}__ternary__ternary"))
        campaign_rows.append(_score(experiment_family="lora", model_alias=alias, support=1607))
        winner_id = f"winner-{alias}"
        winners[f"{alias}:ternary"] = {
            "recipe_id": winner_id, "parameters": {"max_seq_length": 1024},
        }
        grid_rows.extend([
            _score(search_id="grid-fixture", model_alias=alias, recipe_id=winner_id, contradiction_f1=0.9),
            _score(search_id="grid-fixture", model_alias=alias, recipe_id=f"loser-{alias}", contradiction_f1=0.1),
        ])
    paths = {name: tmp_path / f"{name}.csv" for name in ("legacy", "campaign", "grid")}
    pd.DataFrame(legacy_rows).to_csv(paths["legacy"], index=False)
    pd.DataFrame(campaign_rows).to_csv(paths["campaign"], index=False)
    pd.DataFrame(grid_rows).to_csv(paths["grid"], index=False)
    state_path = tmp_path / "grid_state.json"
    state_path.write_text(json.dumps({
        "search_id": "grid-fixture",
        "stages": {"dropout": {"winners": winners}},
    }), encoding="utf-8")

    result = compile_lora_retraining_comparison(
        new_results=new_root, legacy_scores=paths["legacy"],
        campaign_scores=paths["campaign"], grid_scores=paths["grid"],
        grid_state=state_path,
    )

    assert len(result) == 16
    assert set(result.loc[result["cohort"].eq("campaign_full_pipeline_v1"), "support"]) == {1607}
    assert set(result.loc[result["cohort"].eq("balanced_grid_winner"), "contradiction_f1"]) == {0.9}
    assert (new_root / "comparison/lora_comparison.csv").is_file()
    assert (new_root / "comparison/lora_comparison.xlsx").is_file()
