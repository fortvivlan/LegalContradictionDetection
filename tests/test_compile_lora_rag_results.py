from pathlib import Path

import pandas as pd

from LCD.experiments.baselines.series_02_lora_rag_comparison.compile_results import (
    LORA_ORDER,
    SCORE_COLUMNS,
    VARIANT_ORDER,
    _short_lora_name,
    _svg_barplot,
)


def test_short_lora_name_is_explicit() -> None:
    assert (
        _short_lora_name(
            "models__lora__meta-llama_Llama-3.1-8B__ternary__ternary"
        )
        == "Llama-3.1-8B"
    )


def test_svg_barplot_contains_every_model_and_variant(tmp_path: Path) -> None:
    rows = []
    for model in LORA_ORDER:
        for variant in VARIANT_ORDER:
            for dataset in ("FULL",):
                row = {
                    "rag_variant": variant,
                    "lora_name": model,
                    "task": "ternary",
                    "evaluation_scope": "autotest_total",
                    "dataset": dataset,
                }
                row.update({column: 0.5 for column in SCORE_COLUMNS})
                rows.append(row)
    output = tmp_path / "plot.svg"

    _svg_barplot(pd.DataFrame(rows), "ternary", output)

    svg = output.read_text(encoding="utf-8")
    assert "Ternary — contradiction metrics" in svg
    assert all(model in svg for model in LORA_ORDER)
    assert all(variant in svg for variant in VARIANT_ORDER)
    assert svg.count('fill="#4C78A8"') == 9
