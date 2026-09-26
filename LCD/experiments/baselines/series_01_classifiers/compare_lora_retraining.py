"""Compare full-context baseline LoRAs with three saved historical cohorts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from LCD.shared.common import REPOSITORY_ROOT, resolve_model, slugify_model_id
from .retrain_lora import MODELS, RUN_ID

SERIES = REPOSITORY_ROOT / "local/experiments/baselines/series_01_classifiers"
LEGACY_SCORES = SERIES / "full_pipeline_evaluation_baseline/results/all_scores.csv"
CAMPAIGN_SCORES = SERIES / "campaigns/results/full_pipeline_v1/all_experiment_scores.csv"
GRID_ROOT = REPOSITORY_ROOT / "local/experiments/lora/series_01_coordinate_search/lora_coordinate_balanced_val/results"
GRID_SCORES = GRID_ROOT / "all_experiment_scores.csv"
GRID_STATE = GRID_ROOT / "search_state.json"
NEW_RESULTS = SERIES / "lora_retraining" / RUN_ID / "results"
METRICS = (
    "accuracy", "macro_f1", "contradiction_precision", "contradiction_recall",
    "contradiction_f1", "invalid_predictions", "rag_misses",
)
COHORTS = (
    "retrained_baseline", "legacy_incorrect_loss", "campaign_full_pipeline_v1",
    "balanced_grid_winner",
)


def _selected_row(frame, *, source: Path, cohort: str, alias: str):
    """Require exactly one Full/model ternary score for a cohort and model."""
    selected = frame.loc[
        frame["task"].eq("ternary")
        & frame["evaluation_scope"].eq("autotest_model")
        & frame["test_dataset"].eq("Full")
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one ternary Full/autotest_model score for "
            f"{cohort}/{alias} in {source}; found {len(selected)}"
        )
    return selected.iloc[0]


def compile_lora_retraining_comparison(
    *,
    new_results: str | Path = NEW_RESULTS,
    legacy_scores: str | Path = LEGACY_SCORES,
    campaign_scores: str | Path = CAMPAIGN_SCORES,
    grid_scores: str | Path = GRID_SCORES,
    grid_state: str | Path = GRID_STATE,
    output_dir: str | Path | None = None,
):
    """Write a descriptive four-cohort comparison without cross-cohort deltas.

    Historical Full supports differ (1,207 versus 1,607 in the saved runs).
    Equal supports alone do not prove identical gold rows or retrieval settings.
    """
    import pandas as pd

    new_root = Path(new_results).expanduser().resolve()
    sources = {
        "legacy_incorrect_loss": Path(legacy_scores).expanduser().resolve(),
        "campaign_full_pipeline_v1": Path(campaign_scores).expanduser().resolve(),
        "balanced_grid_winner": Path(grid_scores).expanduser().resolve(),
    }
    old = {name: pd.read_csv(path) for name, path in sources.items()}
    state_path = Path(grid_state).expanduser().resolve()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    search_id = state.get("search_id")
    if not search_id or "search_id" not in old["balanced_grid_winner"].columns or set(old["balanced_grid_winner"]["search_id"]) != {search_id}:
        raise ValueError(f"Grid scores and winner state do not match: {state_path}")
    winners = state["stages"]["dropout"]["winners"]
    records = []
    for alias in MODELS:
        model_id = resolve_model(alias).model_id
        slug = slugify_model_id(model_id)
        new_path = new_root / alias / "scores.csv"
        config_path = new_root / alias / "job_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"New LoRA job configuration is missing: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("model_alias") != alias or config.get("model_id") != model_id:
            raise ValueError(f"New LoRA job configuration does not match {alias}")
        new_frame = pd.read_csv(new_path)
        legacy = old["legacy_incorrect_loss"]
        legacy = legacy.loc[
            legacy["model_family"].eq("lora")
            & legacy["model_name"].eq(f"models__lora__{slug}__ternary__ternary")
        ]
        campaign = old["campaign_full_pipeline_v1"]
        campaign = campaign.loc[
            campaign["experiment_family"].eq("lora")
            & campaign["model_alias"].eq(alias)
        ]
        winner = winners[f"{alias}:ternary"]
        grid = old["balanced_grid_winner"]
        grid = grid.loc[
            grid["model_alias"].eq(alias)
            & grid["recipe_id"].eq(winner["recipe_id"])
        ]
        for cohort, frame, source, recipe_id, context in (
            ("retrained_baseline", new_frame, new_path, "baseline_full_context", config["hyperparameters"]["max_seq_length"]),
            ("legacy_incorrect_loss", legacy, sources["legacy_incorrect_loss"], "legacy_adapter", None),
            ("campaign_full_pipeline_v1", campaign, sources["campaign_full_pipeline_v1"], "full_pipeline_v1", None),
            ("balanced_grid_winner", grid, sources["balanced_grid_winner"], winner["recipe_id"], winner["parameters"]["max_seq_length"]),
        ):
            row = _selected_row(frame, source=source, cohort=cohort, alias=alias)
            record = {
                "model_alias": alias,
                "model_id": model_id,
                "cohort": cohort,
                "recipe_id": recipe_id,
                "task": "ternary",
                "test_dataset": "Full",
                "evaluation_scope": "autotest_model",
                "support": int(row["support"]),
                "training_context_tokens": context,
                "source_file": str(source),
                "comparison_note": (
                    "Saved campaign used a different Full pair set; compare descriptively only."
                    if cohort == "campaign_full_pipeline_v1"
                    else "Historical settings and benchmark identity differ; compare descriptively only."
                ),
            }
            record.update({metric: float(row[metric]) for metric in METRICS})
            records.append(record)
    scores = pd.DataFrame.from_records(records)
    scores["cohort"] = pd.Categorical(scores["cohort"], categories=COHORTS, ordered=True)
    scores.sort_values(["model_alias", "cohort"], inplace=True)
    scores["cohort"] = scores["cohort"].astype(str)
    scores.reset_index(drop=True, inplace=True)
    if len(scores) != len(MODELS) * len(COHORTS):
        raise ValueError("Comparison is missing a model or cohort")
    summary = scores.pivot(index="model_alias", columns="cohort", values=["support", "macro_f1", "contradiction_f1"])
    summary.columns = [f"{metric}__{cohort}" for metric, cohort in summary.columns]
    summary = summary.reset_index()
    provenance = pd.DataFrame([
        {"cohort": cohort, "source_file": str(path), "interpretation": note}
        for cohort, path, note in (
            ("retrained_baseline", new_root, "Corrected full-context baseline; current Full/model score."),
            ("legacy_incorrect_loss", sources["legacy_incorrect_loss"], "Historical adapter trained with incorrect loss computation."),
            ("campaign_full_pipeline_v1", sources["campaign_full_pipeline_v1"], "Historical campaign has 1,607 Full/model pairs versus 1,207 in the later reports."),
            ("balanced_grid_winner", sources["balanced_grid_winner"], "Winner IDs come from the saved dropout-stage state."),
        )
    ])
    destination = Path(output_dir).expanduser().resolve() if output_dir else new_root / "comparison"
    destination.mkdir(parents=True, exist_ok=True)
    scores.to_csv(destination / "lora_comparison.csv", index=False)
    with pd.ExcelWriter(destination / "lora_comparison.xlsx", engine="openpyxl") as writer:
        scores.to_excel(writer, sheet_name="scores", index=False)
        summary.to_excel(writer, sheet_name="by_model", index=False)
        provenance.to_excel(writer, sheet_name="provenance", index=False)
    return scores


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-results", type=Path, default=NEW_RESULTS)
    parser.add_argument("--legacy-scores", type=Path, default=LEGACY_SCORES)
    parser.add_argument("--campaign-scores", type=Path, default=CAMPAIGN_SCORES)
    parser.add_argument("--grid-scores", type=Path, default=GRID_SCORES)
    parser.add_argument("--grid-state", type=Path, default=GRID_STATE)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = compile_lora_retraining_comparison(
        new_results=args.new_results,
        legacy_scores=args.legacy_scores,
        campaign_scores=args.campaign_scores,
        grid_scores=args.grid_scores,
        grid_state=args.grid_state,
        output_dir=args.output_dir,
    )
    print(result[["model_alias", "cohort", "support", "macro_f1", "contradiction_f1"]].to_string(index=False))
