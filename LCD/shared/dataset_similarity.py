"""Audit lexical nearest neighbours from validation and Full to training.

The module has no import-time I/O. Similarity is character 3–5 gram TF-IDF
cosine, with vocabulary and IDF fitted on training text only. Pair similarity
averages premise and hypothesis cosine against the *same* training row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from LCD.shared.common import REPOSITORY_ROOT


DEFAULT_TRAIN = REPOSITORY_ROOT / "local/data/classification/train.csv"
DEFAULT_VAL = REPOSITORY_ROOT / "local/data/classification/val.csv"
DEFAULT_FULL = REPOSITORY_ROOT / "local/data/benchmarks/autotest/Full"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "local/analysis/train_val_full_neighbors"
FIELDS = ("premise", "hypothesis")


def normalize_text(value: str) -> str:
    """Make a conservative exact key, preserving punctuation and wording."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()


def file_sha256(path: Path) -> str:
    """Hash an input file for reproducible local audit metadata."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(train_path: Path, val_path: Path, full_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and each reviewed Full workbook with source rows."""
    groups = []
    for split, path in (("train", train_path), ("val", val_path)):
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        missing = set((*FIELDS, "source", "tag")) - set(frame)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        frame = frame.loc[:, [*FIELDS, "source", "tag"]].copy()
        frame["file"] = path.name
        frame["row"] = np.arange(len(frame)) + 2
        frame["split"] = split
        groups.append(frame)

    files = sorted(full_dir.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No Full workbooks found in {full_dir}")
    for path in files:
        frame = pd.read_excel(path, dtype=str, keep_default_na=False)
        missing = set((*FIELDS, "article_number", "expert_label")) - set(frame)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        frame = frame.rename(columns={"article_number": "source", "expert_label": "tag"})
        frame = frame.loc[:, [*FIELDS, "source", "tag"]].copy()
        frame["file"] = path.name
        frame["row"] = np.arange(len(frame)) + 2
        frame["split"] = "Full"
        groups.append(frame)

    rows = pd.concat(groups, ignore_index=True)
    for field in FIELDS:
        rows[f"{field}_key"] = rows[field].map(normalize_text)
        if rows[f"{field}_key"].eq("").any():
            raise ValueError(f"Blank {field} found in input")
    rows["tag"] = rows["tag"].str.strip().str.casefold()
    allowed = {"contradiction", "entailment", "not mentioned"}
    if not set(rows["tag"]).issubset(allowed):
        raise ValueError(f"Unexpected labels: {sorted(set(rows['tag']) - allowed)}")
    rows["reference"] = rows["file"] + ":" + rows["row"].astype(str)
    train = rows.loc[rows.split.eq("train")].reset_index(drop=True)
    queries = rows.loc[rows.split.ne("train")].reset_index(drop=True)
    return train, queries


def nearest_neighbours(train: pd.DataFrame, queries: pd.DataFrame, *, batch_size: int = 128) -> pd.DataFrame:
    """Return every query's best train premise, hypothesis, and complete pair.

    The three maxima are independent. A pair maximum always uses both fields
    from one training row. Ties resolve to the earliest training row.
    """
    if train.empty or queries.empty or batch_size < 1:
        raise ValueError("Train and query sets must be nonempty; batch_size must be positive")
    train_vectors = {}
    query_vectors = {}
    for field in FIELDS:
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), sublinear_tf=True, dtype=np.float32)
        train_vectors[field] = vectorizer.fit_transform(train[f"{field}_key"])
        query_vectors[field] = vectorizer.transform(queries[f"{field}_key"])

    result = queries.loc[:, ["split", "file", "row", "reference", "tag", "source", *FIELDS]].copy()
    for field in FIELDS:
        result[f"{field}_score"] = 0.0
        result[f"{field}_train_index"] = 0
        result[f"{field}_exact"] = False
    result["pair_score"] = 0.0
    result["pair_train_index"] = 0
    result["pair_exact"] = False

    for start in range(0, len(queries), batch_size):
        stop = min(start + batch_size, len(queries))
        scores = {
            field: (query_vectors[field][start:stop] @ train_vectors[field].T).toarray()
            for field in FIELDS
        }
        for field in FIELDS:
            indices = scores[field].argmax(axis=1)
            result.loc[start:stop - 1, f"{field}_train_index"] = indices
            result.loc[start:stop - 1, f"{field}_score"] = np.clip(
                scores[field][np.arange(stop - start), indices], 0, 1
            )
        pair_scores = (scores["premise"] + scores["hypothesis"]) / 2
        indices = pair_scores.argmax(axis=1)
        result.loc[start:stop - 1, "pair_train_index"] = indices
        result.loc[start:stop - 1, "pair_score"] = np.clip(
            pair_scores[np.arange(stop - start), indices], 0, 1
        )

    for field in FIELDS:
        indices = result[f"{field}_train_index"].to_numpy(dtype=int)
        result[f"{field}_train_ref"] = train.iloc[indices]["reference"].to_numpy()
        result[f"{field}_exact"] = (
            queries[f"{field}_key"].to_numpy() == train.iloc[indices][f"{field}_key"].to_numpy()
        )
    indices = result["pair_train_index"].to_numpy(dtype=int)
    neighbours = train.iloc[indices]
    result["pair_train_ref"] = neighbours["reference"].to_numpy()
    result["pair_train_tag"] = neighbours["tag"].to_numpy()
    result["pair_train_source"] = neighbours["source"].to_numpy()
    for field in FIELDS:
        result[f"pair_train_{field}"] = neighbours[field].to_numpy()
    result["pair_exact"] = (
        (queries["premise_key"].to_numpy() == neighbours["premise_key"].to_numpy())
        & (queries["hypothesis_key"].to_numpy() == neighbours["hypothesis_key"].to_numpy())
    )
    return result.drop(columns=["premise_train_index", "hypothesis_train_index", "pair_train_index"])


def summarize(neighbours: pd.DataFrame) -> list[dict]:
    """Summarize row and unique-pair distributions, including label strata."""
    records = []
    for split in ("val", "Full"):
        split_rows = neighbours.loc[neighbours.split.eq(split)].copy()
        for field in FIELDS:
            split_rows[f"{field}_key"] = split_rows[field].map(normalize_text)
        for weighting, data in (
            ("rows", split_rows),
            ("unique_pairs", split_rows.drop_duplicates(["premise_key", "hypothesis_key"])),
        ):
            for label, group in [("all", data), *list(data.groupby("tag", sort=True))]:
                if group.empty:
                    continue
                record = {"split": split, "weighting": weighting, "label": label, "n": len(group)}
                for field in (*FIELDS, "pair"):
                    values = group[f"{field}_score"].to_numpy()
                    record.update({
                        f"{field}_mean": float(values.mean()),
                        f"{field}_median": float(np.median(values)),
                        f"{field}_p10": float(np.quantile(values, 0.1)),
                        f"{field}_p90": float(np.quantile(values, 0.9)),
                        f"{field}_exact_n": int(group[f"{field}_exact"].sum()),
                        f"{field}_ge_90_n": int((values >= 0.90).sum()),
                        f"{field}_ge_95_n": int((values >= 0.95).sum()),
                    })
                records.append(record)
    return records


def main(argv: list[str] | None = None) -> None:
    """Write row-level neighbours and distribution summaries under local/."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--full-dir", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(argv)
    train, queries = load_rows(args.train, args.val, args.full_dir)
    neighbours = nearest_neighbours(train, queries, batch_size=args.batch_size)
    summary = summarize(neighbours)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    neighbours.to_csv(args.output_dir / "nearest_train_neighbors.csv", index=False)
    pd.DataFrame(summary).to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "method.json").write_text(json.dumps({
        "train": str(args.train), "val": str(args.val), "full_dir": str(args.full_dir),
        "input_sha256": {
            str(path): file_sha256(path)
            for path in (args.train, args.val, *sorted(args.full_dir.glob("*.xlsx")))
        },
        "train_rows": len(train), "query_rows": len(queries),
        "normalization": "Unicode NFKC, casefold, collapse whitespace; punctuation retained",
        "similarity": "character 3-5 gram TF-IDF cosine; fit on train only; sublinear_tf",
        "pair_similarity": "mean premise and hypothesis cosine to the same training row",
        "high_similarity_thresholds": [0.90, 0.95],
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
