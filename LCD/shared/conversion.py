"""Convert validated three-class Excel datasets to CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


DATASET_COLUMNS = ("premise", "hypothesis", "source", "tag")
TERNARY_LABELS = ("contradiction", "entailment", "not mentioned")
def _read_and_validate_dataset(path: Path) -> "pd.DataFrame":
    import pandas as pd

    if not path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {path}")

    dataframe = pd.read_excel(path, engine="openpyxl")
    missing_columns = [
        column for column in DATASET_COLUMNS if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            f"{path} is missing required columns: {', '.join(missing_columns)}"
        )

    dataframe = dataframe.loc[:, DATASET_COLUMNS].copy()
    unexpected_labels = sorted(
        {
            str(label)
            for label in dataframe["tag"].dropna().unique()
            if label not in TERNARY_LABELS
        }
    )
    if dataframe["tag"].isna().any() or unexpected_labels:
        details = []
        if dataframe["tag"].isna().any():
            details.append("missing labels")
        if unexpected_labels:
            details.append(f"unexpected labels: {unexpected_labels}")
        raise ValueError(f"Invalid tags in {path}: {'; '.join(details)}")

    return dataframe


def run_conversion(
    train_path: str | Path = "local/data/classification/train.xlsx",
    val_path: str | Path = "local/data/classification/val.xlsx",
    output_dir: str | Path = "local/data/classification",
) -> dict[str, Path]:
    """Validate both XLSX inputs before writing three-class CSV files."""
    train_path = Path(train_path)
    val_path = Path(val_path)
    output_dir = Path(output_dir)

    # Validate both inputs before writing any output files.
    datasets = {
        "train": _read_and_validate_dataset(train_path),
        "val": _read_and_validate_dataset(val_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: dict[str, Path] = {}
    for split, source_dataframe in datasets.items():
        output_path = output_dir / f"{split}.csv"
        source_dataframe.to_csv(output_path, index=False, encoding="utf-8")
        output_paths[split] = output_path

    return output_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert three-class train.xlsx and val.xlsx to CSV files."
    )
    parser.add_argument("--train", type=Path, default=Path("local/data/classification/train.xlsx"))
    parser.add_argument("--val", type=Path, default=Path("local/data/classification/val.xlsx"))
    parser.add_argument("--output-dir", type=Path, default=Path("local/data/classification"))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    written_paths = run_conversion(
        train_path=arguments.train,
        val_path=arguments.val,
        output_dir=arguments.output_dir,
    )
    for output_name, output_path in written_paths.items():
        print(f"{output_name}: {output_path}")
