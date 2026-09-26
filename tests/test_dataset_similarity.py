"""Small, model-free checks for the dataset nearest-neighbour audit."""

import pandas as pd

from LCD.shared.dataset_similarity import nearest_neighbours, normalize_text, summarize


def test_normalized_exact_and_joint_pair_neighbour() -> None:
    train = pd.DataFrame([
        {"premise": "A long statute about a fine", "hypothesis": "An unrelated ruling about detention", "tag": "contradiction", "source": "A", "reference": "train.csv:2"},
        {"premise": "A different statute about appeal", "hypothesis": "A long ruling about a fine", "tag": "entailment", "source": "B", "reference": "train.csv:3"},
    ])
    queries = pd.DataFrame([
        {"split": "val", "file": "val.csv", "row": 2, "reference": "val.csv:2", "tag": "entailment", "source": "A", "premise": "a  long statute about a fine", "hypothesis": "a long ruling about a fine"},
    ])
    for frame in (train, queries):
        for field in ("premise", "hypothesis"):
            frame[f"{field}_key"] = frame[field].map(normalize_text)
    result = nearest_neighbours(train, queries, batch_size=1).iloc[0]
    assert result.premise_exact and result.premise_train_ref == "train.csv:2"
    assert result.hypothesis_exact and result.hypothesis_train_ref == "train.csv:3"
    assert not result.pair_exact
    assert result.pair_score < 1


def test_summary_deduplicates_normalized_pairs() -> None:
    row = {
        "split": "val", "tag": "entailment", "premise": "Same premise",
        "hypothesis": "Same hypothesis", "premise_score": 1.0,
        "hypothesis_score": 1.0, "pair_score": 1.0,
        "premise_exact": True, "hypothesis_exact": True, "pair_exact": True,
    }
    repeated = {**row, "premise": "same   premise", "hypothesis": "SAME HYPOTHESIS"}
    records = summarize(pd.DataFrame([row, repeated]))
    assert next(item for item in records if item["weighting"] == "rows" and item["label"] == "all")["n"] == 2
    assert next(item for item in records if item["weighting"] == "unique_pairs" and item["label"] == "all")["n"] == 1
