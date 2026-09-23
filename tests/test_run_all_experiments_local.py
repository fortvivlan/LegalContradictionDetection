import importlib.util
from pathlib import Path
import sys

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "LCD/experiments/baselines/series_01_classifiers/run_campaign.py"
SPEC = importlib.util.spec_from_file_location("LCD.experiments.baselines.series_01_classifiers.run_campaign", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
_load_score_artifact = MODULE._load_score_artifact


def test_load_score_artifact_accepts_headerless_empty_csv(tmp_path: Path) -> None:
    path = tmp_path / "all_experiment_scores.csv"
    path.write_text("\n", encoding="utf-8")

    scores = _load_score_artifact(path)

    assert scores.empty
    assert list(scores.columns) == []


def test_load_score_artifact_preserves_saved_scores(tmp_path: Path) -> None:
    path = tmp_path / "all_experiment_scores.csv"
    expected = pd.DataFrame({"job_id": ["bert:bert:ternary"], "macro_f1": [0.75]})
    expected.to_csv(path, index=False)

    scores = _load_score_artifact(path)

    pd.testing.assert_frame_equal(scores, expected)
