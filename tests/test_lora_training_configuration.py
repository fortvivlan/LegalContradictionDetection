import importlib.util
import pickle
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

from LCD.shared import lora
from LCD.shared.common import merge_parameters


def test_existing_lora_defaults_remain_the_current_recipe() -> None:
    defaults = lora.DEFAULT_LORA_HYPERPARAMETERS

    assert defaults["target_modules"] == "all-linear"
    assert defaults["lora_dropout"] == 0.05
    assert defaults["epochs"] == 3
    assert defaults["learning_rate"] == 2e-4
    assert defaults["lr_scheduler_type"] == "cosine"
    assert defaults["optimizer"] == "auto"
    assert defaults["eval_strategy"] == "no"
    assert defaults["save_strategy"] == "no"


def test_validation_mismatch_reuse_rejects_changed_training_settings(
    monkeypatch, tmp_path: Path
) -> None:
    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    train.write_text("training", encoding="utf-8")
    val.write_text("validation", encoding="utf-8")
    manifest = {
        "hyperparameters": dict(lora.DEFAULT_LORA_HYPERPARAMETERS),
        "validation_sha256": "previous-validation",
    }
    monkeypatch.setattr(lora, "load_saved_artifact_manifest", lambda *args, **kwargs: manifest)
    with pytest.raises(ValueError, match="unchanged adapter settings"):
        lora.run(
            "qwen", "ternary", {"lora_rank": 32},
            train_path=train, val_path=val, artifact_root=tmp_path,
            use_existing_model=True, allow_validation_mismatch=True,
        )
    manifest["hyperparameters"]["load_best_model_at_end"] = True
    with pytest.raises(ValueError, match="checkpoint that was not selected"):
        lora.run(
            "qwen", "ternary",
            train_path=train, val_path=val, artifact_root=tmp_path,
            use_existing_model=True, allow_validation_mismatch=True,
        )


def test_tokenized_rows_dataset_is_pickleable() -> None:
    dataset = lora._TokenizedRowsDataset(
        [{"input_ids": [1], "attention_mask": [1], "labels": [1]}]
    )

    restored = pickle.loads(pickle.dumps(dataset))

    assert len(restored) == 1
    assert restored[0]["labels"] == [1]
    assert "<locals>" not in restored.__class__.__qualname__


def test_lora_training_text_always_includes_source_prefix(monkeypatch) -> None:
    captured = {}

    class FakeTokenizer:
        eos_token = ""

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(character) for character in text]}

    def fake_build(tokenizer, premise, hypothesis, label, task):
        captured["premise"] = premise
        return "prompt", "prompt label"

    monkeypatch.setattr(lora, "build_training_texts", fake_build)
    dataframe = pd.DataFrame(
        [
            {
                "premise": "Provision body.",
                "hypothesis": "Decision sentence.",
                "source": "КоАП Статья 18.8 Часть 3.1",
                "tag": "entailment",
            }
        ]
    )

    rows = lora._tokenize_training_rows(
        dataframe,
        FakeTokenizer(),
        "ternary",
        128,
        model_alias="qwen",
    )

    assert captured["premise"] == (
        "КоАП Статья 18.8 Часть 3.1 Provision body."
    )
    assert rows[0]["labels"][-6:] == [ord(character) for character in " label"]


def test_historical_settings_reach_lora_and_training_arguments(monkeypatch) -> None:
    captured = {}

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(use_cache=True)

        def gradient_checkpointing_disable(self):
            captured["checkpointing_disabled"] = True

        def eval(self):
            captured["model_eval"] = True

    class FakeTokenizer:
        padding_side = "left"

    fake_model = FakeModel()
    fake_tokenizer = FakeTokenizer()
    monkeypatch.setattr(
        lora,
        "load_causal_model",
        lambda *args, **kwargs: SimpleNamespace(
            model=fake_model,
            tokenizer=fake_tokenizer,
            compute_dtype="float16",
        ),
    )
    monkeypatch.setattr(
        lora,
        "_tokenize_training_rows",
        lambda dataframe, *args, **kwargs: [
            {"input_ids": [1], "attention_mask": [1], "labels": [1]}
            for _ in dataframe
        ],
    )

    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(empty_cache=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_peft = ModuleType("peft")

    def lora_config(**kwargs):
        captured["lora_config"] = kwargs
        return kwargs

    fake_peft.LoraConfig = lora_config
    fake_peft.prepare_model_for_kbit_training = lambda model, **kwargs: model
    fake_peft.get_peft_model = lambda model, config: model
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            captured["training_args"] = kwargs

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured["trainer"] = kwargs
            self.state = SimpleNamespace(log_history=[{"eval_loss": 0.25}])

        def train(self, **kwargs):
            captured["train_kwargs"] = kwargs
            return SimpleNamespace(metrics={"train_loss": 0.1})

    fake_transformers = ModuleType("transformers")
    fake_transformers.DataCollatorForSeq2Seq = lambda **kwargs: kwargs
    fake_transformers.TrainingArguments = FakeTrainingArguments
    fake_transformers.Trainer = FakeTrainer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    script_path = Path(__file__).resolve().parents[1] / "LCD/experiments/lora/series_01_coordinate_search/learning_rate.py"
    spec = importlib.util.spec_from_file_location(
        "LCD/experiments/lora/series_01_coordinate_search/learning_rate", script_path
    )
    assert spec is not None and spec.loader is not None
    campaign = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = campaign
    spec.loader.exec_module(campaign)

    parameters = merge_parameters(
        lora.DEFAULT_LORA_HYPERPARAMETERS,
        {**campaign.HISTORICAL_LORA_OVERRIDES, "learning_rate": 2e-5},
    )
    train_rows = [object(), object()]
    validation_rows = [object()]

    lora._train_adapter(
        train_rows,
        validation_dataframe=validation_rows,
        spec=SimpleNamespace(alias="qwen"),
        task="ternary",
        revision="a" * 40,
        token=None,
        parameters=parameters,
        trainer_output_dir="trainer-output",
        resume_from_checkpoint="trainer-output/checkpoint-100",
    )

    assert captured["lora_config"] == {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    training_args = captured["training_args"]
    assert training_args["output_dir"] == "trainer-output"
    assert training_args["num_train_epochs"] == 5
    assert training_args["per_device_train_batch_size"] == 2
    assert training_args["gradient_accumulation_steps"] == 4
    assert training_args["learning_rate"] == 2e-5
    assert training_args["optim"] == "adamw_torch_fused"
    assert training_args["lr_scheduler_type"] == "linear"
    assert training_args["warmup_ratio"] == 0.0
    assert training_args["logging_steps"] == 10
    assert training_args["eval_strategy"] == "epoch"
    assert training_args["save_strategy"] == "epoch"
    assert training_args["fp16"] is True
    assert training_args["bf16"] is False
    assert len(captured["trainer"]["train_dataset"]) == 2
    assert len(captured["trainer"]["eval_dataset"]) == 1
    assert captured["train_kwargs"] == {
        "resume_from_checkpoint": "trainer-output/checkpoint-100"
    }
