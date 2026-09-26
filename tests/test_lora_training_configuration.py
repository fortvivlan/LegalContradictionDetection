import pickle
from pathlib import Path

import pandas as pd
import pytest

from LCD.shared import lora


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
    assert defaults["max_seq_length"] == 2560
    assert defaults["batch_size"] == 1
    assert defaults["gradient_accumulation_steps"] == 16


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


def test_lora_tokenization_keeps_complete_prompt_and_response(monkeypatch) -> None:
    class CharacterTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(character) for character in text]}

    def build(tokenizer, premise, hypothesis, label, task):
        prompt = f"instructions|{premise}|{hypothesis}|assistant:"
        return prompt, prompt + label

    monkeypatch.setattr(lora, "build_training_texts", build)
    example = pd.DataFrame([{
        "example_id": "train:000001", "premise": "premise body", "source": "citation",
        "hypothesis": "hypothesis body", "tag": "entailment",
    }])
    prompt = "instructions|citation premise body|hypothesis body|assistant:"
    full = prompt + "entailment"
    rows = lora._tokenize_training_rows(
        example, CharacterTokenizer(), "ternary", len(full), model_alias="qwen"
    )

    assert rows[0]["input_ids"] == [ord(char) for char in full]
    assert rows[0]["labels"] == [-100] * len(prompt) + [ord(char) for char in "entailment"]
    with pytest.raises(ValueError, match=f"train:000001 needs {len(full)} tokens"):
        lora._tokenize_training_rows(
            example, CharacterTokenizer(), "ternary", len(full) - 1,
            model_alias="qwen",
        )


def test_lora_tokenization_rejects_changed_prompt_boundary(monkeypatch) -> None:
    class CharacterTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(character) for character in text]}

    monkeypatch.setattr(lora, "build_training_texts", lambda *args: ("prompt", "different label"))
    example = pd.DataFrame([{
        "premise": "p", "source": "", "hypothesis": "h", "tag": "entailment",
    }])
    with pytest.raises(ValueError, match="not an exact prompt continuation"):
        lora._tokenize_training_rows(
            example, CharacterTokenizer(), "ternary", 2560, model_alias="qwen"
        )
