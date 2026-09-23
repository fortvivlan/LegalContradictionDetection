from pathlib import Path

import pandas as pd
import pytest

from LCD.shared.common import (
    evaluate_predictions,
    load_dataset,
    merge_parameters,
    resolve_model,
)


def test_resolve_model_accepts_alias_and_full_id() -> None:
    assert resolve_model("qwen").model_id == "Qwen/Qwen3-8B"
    assert resolve_model("Qwen/Qwen3-8B").alias == "qwen"


def test_resolve_model_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unsupported model"):
        resolve_model("unknown/model")


def test_merge_parameters_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="Unknown hyperparameter"):
        merge_parameters({"epochs": 3}, {"epoch": 2})


def test_load_dataset_uses_explicit_task_labels(tmp_path: Path) -> None:
    path = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "premise": ["p"],
            "hypothesis": ["h"],
            "source": ["s"],
            "tag": ["no"],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="invalid ternary labels"):
        load_dataset(path, "ternary")


def test_load_dataset_accepts_missing_optional_source(tmp_path: Path) -> None:
    path = tmp_path / "ternary.csv"
    pd.DataFrame(
        {
            "premise": ["p"],
            "hypothesis": ["h"],
            "source": [None],
            "tag": ["entailment"],
        }
    ).to_csv(path, index=False)

    dataframe = load_dataset(path, "ternary")

    assert len(dataframe) == 1
    assert dataframe.loc[0, "source"] == ""


@pytest.mark.parametrize("missing_column", ["premise", "hypothesis", "tag"])
def test_load_dataset_rejects_missing_training_values(
    tmp_path: Path, missing_column: str
) -> None:
    path = tmp_path / f"missing_{missing_column}.csv"
    row = {
        "premise": "p",
        "hypothesis": "h",
        "source": "s",
        "tag": "entailment",
    }
    row[missing_column] = None
    pd.DataFrame([row]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing premise, hypothesis, or tag"):
        load_dataset(path, "ternary")


def test_evaluate_predictions_counts_invalid_as_wrong() -> None:
    dataframe = pd.DataFrame(
        {
            "example_id": ["v:0", "v:1", "v:2"],
            "premise": ["p0", "p1", "p2"],
            "hypothesis": ["h0", "h1", "h2"],
            "source": ["s0", "s1", "s2"],
            "tag": ["contradiction", "entailment", "not mentioned"],
        }
    )

    tables = evaluate_predictions(
        dataframe,
        ["contradiction", None, "not mentioned"],
        ["contradiction", "unparseable", "not mentioned"],
        model_id="model",
        task="ternary",
    )

    score = tables.scores.iloc[0]
    assert score["accuracy"] == pytest.approx(2 / 3)
    assert score["invalid_predictions"] == 1
    assert set(tables.per_class["label"]) == {"contradiction", "entailment", "not mentioned"}
    invalid_column = tables.confusion_matrix[
        tables.confusion_matrix["predicted_label"] == "invalid"
    ]
    assert invalid_column["count"].sum() == 1
