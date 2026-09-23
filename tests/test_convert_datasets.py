from pathlib import Path

import pandas as pd
import pytest

from LCD.shared.conversion import run_conversion


def _write_dataset(path: Path, labels: list[str]) -> None:
    pd.DataFrame({
        "premise": [f"premise {i}" for i in range(len(labels))],
        "hypothesis": [f"hypothesis {i}" for i in range(len(labels))],
        "source": [f"source {i}" for i in range(len(labels))],
        "tag": labels,
    }).to_excel(path, index=False, engine="openpyxl")


def test_conversion_preserves_three_class_labels_and_columns(tmp_path: Path) -> None:
    train, val, output = tmp_path / "train.xlsx", tmp_path / "val.xlsx", tmp_path / "csv"
    _write_dataset(train, ["contradiction", "entailment", "not mentioned"])
    _write_dataset(val, ["not mentioned", "contradiction"])

    paths = run_conversion(train, val, output)

    assert set(paths) == {"train", "val"}
    assert {p.name for p in paths.values()} == {"train.csv", "val.csv"}
    converted = pd.read_csv(paths["train"])
    assert converted.columns.tolist() == ["premise", "hypothesis", "source", "tag"]
    assert converted["tag"].tolist() == ["contradiction", "entailment", "not mentioned"]


def test_conversion_validates_both_inputs_before_writing(tmp_path: Path) -> None:
    train, val, output = tmp_path / "train.xlsx", tmp_path / "val.xlsx", tmp_path / "csv"
    _write_dataset(train, ["contradiction"])
    _write_dataset(val, ["invalid"])
    with pytest.raises(ValueError, match="unexpected labels"):
        run_conversion(train, val, output)
    assert not output.exists()
