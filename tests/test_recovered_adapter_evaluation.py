from pathlib import Path

import pandas as pd
import pytest

from LCD.experiments.lora.series_00_legacy_adapters.recovered_adapter_evaluation import (
    extract_legacy_label,
    load_legacy_autotest_pairs,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("contradiction", "contradiction"),
        ("Answer: entailment", "entailment"),
        ("<think>ignored</think> Not Mentioned", "not mentioned"),
        ("not mentioned, not contradiction", "not mentioned"),
        ("unknown", None),
    ],
)
def test_extract_legacy_label_uses_notebook_precedence(raw, expected) -> None:
    assert extract_legacy_label(raw) == expected


def test_load_legacy_autotest_pairs_preserves_exact_premise(tmp_path: Path) -> None:
    article = "КоАП РФ: Статья 20.20. п. 3. "
    premise = article + "Exact provision text"
    pd.DataFrame(
        [
            {
                "sentence": "Exact hypothesis",
                "article": article,
                "premise": premise,
                "answer": "contradiction",
            }
        ]
    ).to_excel(tmp_path / "result_Тест_Иванов_old.xlsx", index=False)

    pairs = load_legacy_autotest_pairs(tmp_path)

    assert len(pairs) == 1
    assert pairs.loc[0, "premise"] == premise
    assert pairs.loc[0, "hypothesis"] == "Exact hypothesis"
    assert pairs.loc[0, "gold_label"] == "contradiction"
    assert pairs.loc[0, "subject_key"] == "иванов"
