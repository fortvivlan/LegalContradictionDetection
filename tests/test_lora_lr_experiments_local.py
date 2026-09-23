import json
from inspect import signature
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from LCD.shared.common import merge_parameters, resolve_model
from LCD.shared.lora import DEFAULT_LORA_HYPERPARAMETERS
from LCD.experiments.lora.series_01_coordinate_search import engine as sweep
from LCD.experiments.lora.series_01_coordinate_search.alpha import run_lora_alpha_experiments
from LCD.experiments.lora.series_01_coordinate_search.dropout import run_lora_dropout_experiments
from LCD.experiments.lora.series_01_coordinate_search.learning_rate import run_lora_lr_experiments
from LCD.experiments.lora.series_01_coordinate_search.rank import run_lora_rank_experiments
from LCD.experiments.lora.series_01_coordinate_search.target_modules import run_lora_target_module_experiments
from LCD.experiments.lora.series_01_coordinate_search.compare_llm import run_lora_vs_llm_comparison


def _state(models=("qwen", "llama", "ministral", "t-lite"), tasks=("ternary",)):
    base = merge_parameters(
        DEFAULT_LORA_HYPERPARAMETERS, sweep.HISTORICAL_LORA_OVERRIDES
    )
    return {
        "configuration": {
            "models": list(models),
            "tasks": list(tasks),
            "base_hyperparameters": base,
            "dataset_sha256": {
                f"{split}_{task}": f"{split}-{task}"
                for split in ("train", "val")
                for task in tasks
            },
            "prompt_sha256": {task: f"prompt-{task}" for task in tasks},
        },
        "pins": {
            "model_revisions": {model: model * 8 for model in models},
            "rag_revision": "b" * 40,
        },
        "stages": {},
        "experiments": {},
    }


def _complete_with_label(state, stage, label):
    candidates = sweep.build_stage_candidates(stage, state)
    stage_state = sweep._register_stage(state, stage, candidates)
    winners = {}
    for candidate in stage_state["candidates"].values():
        if candidate["label"] == label:
            winners[f"{candidate['model_alias']}:{candidate['task']}"] = candidate
    stage_state["winners"] = winners
    stage_state["status"] = "completed"
    return candidates


def test_coordinate_search_has_60_logical_and_44_unique_recipes() -> None:
    state = _state()
    logical_counts = []
    new_counts = []
    labels = {
        "target_modules": "qv",
        "rank": "r_16",
        "learning_rate": "lr_2e-5",
        "alpha": "alpha_2r",
        "dropout": "dropout_0.1",
    }
    previous_total = 0

    for stage in sweep.STAGE_ORDER:
        candidates = _complete_with_label(state, stage, labels[stage])
        logical_counts.append(len(candidates))
        new_counts.append(len(state["experiments"]) - previous_total)
        previous_total = len(state["experiments"])

    assert logical_counts == [12, 12, 16, 8, 12]
    assert new_counts == [12, 8, 12, 4, 8]
    assert sum(logical_counts) == 60
    assert len(state["experiments"]) == 44


def test_all_stage_entrypoints_default_to_four_models_and_ternary_only() -> None:
    runners = (
        run_lora_target_module_experiments,
        run_lora_rank_experiments,
        run_lora_lr_experiments,
        run_lora_alpha_experiments,
        run_lora_dropout_experiments,
    )

    for runner in runners:
        parameters = signature(runner).parameters
        assert parameters["models"].default == sweep.ALL_MODELS
        assert parameters["tasks"].default == ("ternary",)
        assert parameters["search_id"].default == "lora_coordinate_imbalanced_val"
        assert parameters["nruns"].default == 4
    assert (
        signature(run_lora_vs_llm_comparison).parameters["search_id"].default
        == "lora_coordinate_imbalanced_val"
    )
    assert signature(run_lora_vs_llm_comparison).parameters["nruns"].default == 4


def test_ready_score_validation_accepts_ternary_only() -> None:
    rows = [
        {
            "model_alias": "qwen",
            "task": "ternary",
            "evaluation_scope": "validation",
            "test_dataset": None,
            "contradiction_f1": 0.5,
        }
    ]
    rows.extend(
        {
            "model_alias": "qwen",
            "task": "ternary",
            "evaluation_scope": scope,
            "test_dataset": dataset,
            "contradiction_f1": 0.5,
        }
        for dataset in sweep.DATASETS
        for scope in sweep.BENCHMARK_SCOPES
    )
    scores = pd.DataFrame(rows)

    validated = sweep._validate_ready_scores("qwen", scores, ("ternary",))

    pd.testing.assert_frame_equal(validated.reset_index(drop=True), scores)


def test_score_validation_requires_contradiction_f1_and_full_test_only() -> None:
    experiment = {"recipe_id": "recipe", "task": "ternary"}
    scores = pd.DataFrame(
        [
            {"task": "ternary", "evaluation_scope": "validation", "test_dataset": None},
            {"task": "ternary", "evaluation_scope": "autotest_model", "test_dataset": "Full"},
        ]
    )
    with pytest.raises(ValueError, match="Incomplete score columns"):
        sweep._validate_experiment_scores(experiment, scores)

    scores["contradiction_f1"] = [0.7, 0.8]
    scores.loc[1, "test_dataset"] = "Dialogue"
    with pytest.raises(ValueError, match="Incomplete score rows"):
        sweep._validate_experiment_scores(experiment, scores)


def test_stage_grids_and_inheritance_are_exact() -> None:
    state = _state(models=("qwen",), tasks=("ternary",))
    targets = _complete_with_label(state, "target_modules", "qkv")
    assert [candidate.parameters["target_modules"] for candidate in targets] == [
        ["q_proj", "v_proj"],
        ["q_proj", "k_proj", "v_proj"],
        "all-linear",
    ]

    ranks = _complete_with_label(state, "rank", "r_32")
    assert [candidate.parameters["lora_rank"] for candidate in ranks] == [8, 16, 32]
    assert [candidate.parameters["lora_alpha"] for candidate in ranks] == [16, 32, 64]
    assert all(
        candidate.parameters["target_modules"] == ["q_proj", "k_proj", "v_proj"]
        for candidate in ranks
    )

    rates = _complete_with_label(state, "learning_rate", "lr_1e-4")
    assert [candidate.parameters["learning_rate"] for candidate in rates] == [
        2e-5,
        1e-4,
        2e-4,
        1e-5,
    ]

    alphas = _complete_with_label(state, "alpha", "alpha_1r")
    assert [candidate.parameters["lora_alpha"] for candidate in alphas] == [32, 64]

    dropouts = sweep.build_stage_candidates("dropout", state)
    assert [candidate.parameters["lora_dropout"] for candidate in dropouts] == [
        0.0,
        0.05,
        0.1,
    ]


def test_winner_uses_validation_contradiction_f1_not_macro_or_benchmark() -> None:
    state = _state(models=("qwen",), tasks=("ternary",))
    candidates = sweep.build_stage_candidates("target_modules", state)
    sweep._register_stage(state, "target_modules", candidates)
    rows = []
    for index, candidate in enumerate(candidates):
        recipe_id = sweep._recipe_id(
            candidate.model_alias, candidate.task, candidate.parameters, state
        )
        rows.extend(
            [
                {
                    "recipe_id": recipe_id,
                    "model_alias": candidate.model_alias,
                    "task": candidate.task,
                    "evaluation_scope": "validation",
                    "test_dataset": None,
                    "macro_f1": 0.9 - index * 0.1,
                    "contradiction_f1": 0.2 + index * 0.3,
                    "invalid_predictions": 0,
                },
                {
                    "recipe_id": recipe_id,
                    "model_alias": candidate.model_alias,
                    "task": candidate.task,
                    "evaluation_scope": "autotest_model",
                    "test_dataset": "Full",
                    "macro_f1": 0.1 + index * 0.4,
                    "contradiction_f1": 0.9 - index * 0.3,
                    "invalid_predictions": 0,
                },
            ]
        )

    sweep._rank_and_finalize(state, "target_modules", pd.DataFrame(rows))

    assert state["stages"]["target_modules"]["winners"]["qwen:ternary"]["label"] == "all_linear"
    validation = sweep._stage_frames(state, "target_modules", pd.DataFrame(rows))[1]
    assert validation.iloc[0]["label"] == "all_linear"


def test_adapter_reuse_requires_complete_matching_manifest(tmp_path: Path) -> None:
    state = _state(models=("qwen",), tasks=("ternary",))
    candidate = sweep.build_stage_candidates("target_modules", state)[0]
    sweep._register_stage(state, "target_modules", [candidate])
    experiment = next(iter(state["experiments"].values()))
    target = sweep._adapter_target(tmp_path, experiment)
    target.mkdir(parents=True)
    for name in ("adapter_config.json", "tokenizer_config.json"):
        (target / name).write_text("{}", encoding="utf-8")
    (target / "adapter_model.safetensors").write_bytes(b"adapter")
    manifest = {
        "model_id": resolve_model("qwen").model_id,
        "task": "ternary",
        "resolved_revision": state["pins"]["model_revisions"]["qwen"],
        "train_sha256": "train-ternary",
        "validation_sha256": "val-ternary",
        "prompt_sha256": "prompt-ternary",
        "prompt_processing": "standard_chat_template_v1",
        "premise_format": "source_prefixed_v1",
        "rag_revision": "b" * 40,
        "hyperparameters": dict(experiment["parameters"]),
    }
    (target / "run_config.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert sweep._adapter_matches(tmp_path, experiment, state)
    manifest["hyperparameters"]["learning_rate"] = 1e-6
    (target / "run_config.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert not sweep._adapter_matches(tmp_path, experiment, state)


def test_latest_checkpoint_uses_highest_numeric_step(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-9").mkdir()
    (tmp_path / "checkpoint-100").mkdir()
    (tmp_path / "checkpoint-final").mkdir()

    assert sweep._latest_checkpoint(tmp_path) == tmp_path / "checkpoint-100"


def test_each_lora_model_uses_imbalanced_val_and_full_only(monkeypatch, tmp_path: Path) -> None:
    state = _state()
    candidates = sweep.build_stage_candidates("target_modules", state)
    sweep._register_stage(state, "target_modules", candidates)
    run_lora = Mock(return_value=pd.DataFrame())
    monkeypatch.setattr("LCD.shared.lora.run", run_lora)
    monkeypatch.setattr(sweep, "_adapter_matches", lambda *args: False)
    monkeypatch.setattr(sweep, "_latest_checkpoint", lambda *args: None)

    for model_alias in sweep.ALL_MODELS:
        experiment = next(
            record for record in state["experiments"].values()
            if record["model_alias"] == model_alias
        )
        sweep._run_experiment(
            experiment,
            root=tmp_path,
            artifact_root=tmp_path / "artifacts",
            result_root=tmp_path / "results",
            state=state,
        )

    assert [call.args[0] for call in run_lora.call_args_list] == list(sweep.ALL_MODELS)
    for call in run_lora.call_args_list:
        assert call.kwargs["train_path"] == tmp_path / "local/data/classification/train.csv"
        assert call.kwargs["val_path"] == tmp_path / "local/data/classification/val.csv"
        assert call.kwargs["selected_datasets"] == ("Full",)
        assert call.kwargs["benchmark_scopes"] == ("autotest_model",)


def test_empty_scores_do_not_complete_a_stage() -> None:
    state = _state(models=("qwen",), tasks=("ternary",))
    candidates = sweep.build_stage_candidates("target_modules", state)
    sweep._register_stage(state, "target_modules", candidates)

    assert not sweep._stage_complete(state, "target_modules", pd.DataFrame())


def test_completed_stage_writes_all_result_sheets(tmp_path: Path) -> None:
    state = _state(models=("qwen",), tasks=("ternary",))
    candidates = sweep.build_stage_candidates("target_modules", state)
    sweep._register_stage(state, "target_modules", candidates)
    rows = []
    for index, candidate in enumerate(candidates):
        recipe_id = sweep._recipe_id(
            candidate.model_alias, candidate.task, candidate.parameters, state
        )
        state["experiments"][recipe_id]["status"] = "completed"
        for dataset, scope in ((None, "validation"), ("Full", "autotest_model")):
            rows.append(
                {
                    "recipe_id": recipe_id,
                    "model_alias": "qwen",
                    "task": "ternary",
                    "evaluation_scope": scope,
                    "test_dataset": dataset,
                    "macro_f1": 0.8 - index * 0.1,
                    "contradiction_f1": 0.7,
                    "invalid_predictions": 0,
                }
            )
    scores = pd.DataFrame(rows)
    sweep._rank_and_finalize(state, "target_modules", scores)

    sweep._write_stage_artifacts(state, "target_modules", scores, tmp_path)

    stage_dir = tmp_path / "stages/target_modules"
    assert (stage_dir / "scores.csv").is_file()
    workbook = pd.ExcelFile(stage_dir / "results.xlsx")
    assert workbook.sheet_names == [
        "scores",
        "validation_ranking",
        "benchmark_ranking",
        "winners",
        "candidates",
        "experiments",
        "failures",
    ]


def test_invocation_attempt_cap_is_enforced(monkeypatch, tmp_path: Path) -> None:
    base_configuration = {}

    def fake_configuration(root, models, tasks, base):
        del root
        configuration = {
            "models": list(models),
            "tasks": list(tasks),
            "base_hyperparameters": dict(base),
            "dataset_sha256": {
                f"{split}_{task}": f"{split}-{task}"
                for split in ("train", "val")
                for task in tasks
            },
            "prompt_sha256": {task: f"prompt-{task}" for task in tasks},
        }
        base_configuration.update(configuration)
        return configuration

    monkeypatch.setattr(sweep, "_preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr(sweep, "_configuration", fake_configuration)
    monkeypatch.setattr(
        sweep,
        "_resolve_pins",
        lambda root, models: {
            "model_revisions": {model: "a" * 40 for model in models},
            "rag_revision": "b" * 40,
        },
    )
    monkeypatch.setattr(
        sweep, "_run_experiment", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    monkeypatch.setattr(sweep, "_write_stage_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(sweep, "_cleanup_cuda", lambda logger: None)

    sweep.run_sweep_stage(
        "target_modules",
        search_id="attempt-cap",
        repo_root=tmp_path,
        models=("qwen",),
        tasks=("ternary",),
        nruns=2,
        max_retries=0,
    )

    state_path = tmp_path / "local/experiments/lora/series_01_coordinate_search/attempt-cap/results/search_state.json"
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert sum(record["attempts"] for record in saved["experiments"].values()) == 2
    assert sum(record["status"] == "failed" for record in saved["experiments"].values()) == 2
    assert sum(record["status"] == "pending" for record in saved["experiments"].values()) == 1

    executed = []

    def successful_run(experiment, **kwargs):
        executed.append(experiment["recipe_id"])
        return pd.DataFrame(
            [
                {
                    "task": "ternary",
                    "evaluation_scope": "validation",
                    "test_dataset": None,
                    "macro_f1": 0.3,
                    "contradiction_f1": 0.7,
                    "invalid_predictions": 0,
                },
                {
                    "task": "ternary",
                    "evaluation_scope": "autotest_model",
                    "test_dataset": "Full",
                    "macro_f1": 0.9,
                    "contradiction_f1": 0.8,
                    "invalid_predictions": 0,
                },
            ]
        )

    monkeypatch.setattr(sweep, "_run_experiment", successful_run)
    for _ in range(2):
        sweep.run_sweep_stage(
            "target_modules",
            search_id="attempt-cap",
            repo_root=tmp_path,
            models=("qwen",),
            tasks=("ternary",),
            nruns=2,
            max_retries=0,
        )
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["stages"]["target_modules"]["status"] == "completed"
    assert len(executed) == 3
    assert {record["status"] for record in saved["experiments"].values()} == {"completed"}
    assert sorted(record["attempts"] for record in saved["experiments"].values()) == [1, 2, 2]

    sweep.run_sweep_stage(
        "target_modules",
        search_id="attempt-cap",
        repo_root=tmp_path,
        models=("qwen",),
        tasks=("ternary",),
        nruns=2,
    )
    assert len(executed) == 3
