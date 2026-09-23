import json
from pathlib import Path

import pandas as pd
import pytest

from LCD.shared.rag.artifacts import RagBundle, RerankerBundle
from LCD.shared.rag.depth_sweep import (
    DEFAULT_STAGE_TWO_DEPTH_RUNS,
    FOCUSED_VARIANT,
    RagDepthRun,
    _validate_manifests,
    run_rag_depth_sweep,
)
from LCD.shared.rag.evaluation import SUMMARY_COLUMNS
from LCD.experiments.lawbook.series_01_embedding_reranking.run_depth_sweep import _parse_depth_run


def _write_manifest(root: Path, name: str, *, rag_commit: str = "commit") -> Path:
    target = root / name
    target.mkdir(parents=True)
    manifest = target / "rag_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": name,
                "codex_sha256": "codex-hash",
                "rag_commit": rag_commit,
                "model_id": "encoder",
                "resolved_revision": "encoder-revision",
                "reranker": {
                    "mode": "finetuned",
                    "local": True,
                    "model": "reranker_model",
                    "base_model_id": "reranker",
                    "base_model_revision": "reranker-revision",
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_stage_two_depth_run_cli_parser() -> None:
    parsed = _parse_depth_run("sbert_legal_40:60:40")
    assert (parsed.artifact_run, parsed.candidate_top_k, parsed.final_top_k) == (
        "sbert_legal_40",
        60,
        40,
    )


def test_incompatible_sweep_manifests_fail(tmp_path: Path) -> None:
    _write_manifest(tmp_path, "first")
    _write_manifest(tmp_path, "second", rag_commit="other")

    with pytest.raises(ValueError, match="incompatible"):
        _validate_manifests(
            [RagDepthRun("first", 20, 10), RagDepthRun("second", 40, 20)],
            tmp_path,
        )


def test_multi_artifact_depth_sweep_preserves_mapping_and_ranks_full_recall(
    monkeypatch, tmp_path: Path
) -> None:
    artifact_root = tmp_path / "artifacts"
    for name in ("sbert_legal_v1", "sbert_legal_40", "sbert_legal_60"):
        _write_manifest(artifact_root, name)
    rag_dir = tmp_path / "rag"
    rag_dir.mkdir()
    (rag_dir / "codex.csv").write_text("text,source\n", encoding="utf-8")
    results = tmp_path / "results" / "sbert_legal_v1" / "top_k_stage2"
    stage_one = results.parent / "rag_recall.csv"
    stage_one.parent.mkdir(parents=True, exist_ok=True)
    stage_one.write_text("stage-one", encoding="utf-8")
    loaded_rerankers = []
    calls = []

    monkeypatch.setattr(
        "LCD.shared.rag.depth_sweep.ensure_rag_repository",
        lambda path, revision: (Path(path), revision),
    )

    def fake_load(source, *, name=None, codex_override=None):
        path = Path(source)
        if path == rag_dir:
            return RagBundle(
                "baseline", rag_dir / "codex.csv", rag_dir / "index", "base", None, False
            )
        artifact_name = path.parent.name
        return RagBundle(
            artifact_name,
            rag_dir / "codex.csv",
            path.parent / "index",
            str(path.parent / "embedding"),
            None,
            True,
            reranker=RerankerBundle(
                "finetuned",
                str(path.parent / "reranker_model"),
                None,
                True,
                1024,
            ),
        )

    monkeypatch.setattr(
        "LCD.shared.rag.depth_sweep.load_rag_bundle", fake_load
    )

    class FakeReranker:
        def __init__(self, model_id, **kwargs):
            self.model_id = model_id
            loaded_rerankers.append(Path(model_id).parent.name)

    monkeypatch.setattr(
        "LCD.shared.rag.depth_sweep.CrossEncoderReranker", FakeReranker
    )
    monkeypatch.setattr(
        "LCD.shared.rag.depth_sweep._release_gpu", lambda: None
    )
    scores_by_depth = {
        (100, 80): 0.90,
        (80, 60): 0.95,
        (60, 40): 0.95,
        (40, 20): 0.80,
        (20, 10): 0.70,
        (100, 60): 0.92,
        (100, 40): 0.88,
        (100, 20): 0.82,
        (100, 10): 0.75,
    }

    def fake_evaluation(sources, *, retrieval_depths, results_dir, **kwargs):
        artifact_name = sources[1].name
        calls.append((artifact_name, tuple(retrieval_depths), kwargs))
        rows = []
        for candidate, final in retrieval_depths:
            row = {column: 0.0 for column in SUMMARY_COLUMNS}
            row.update(
                {
                    "variant": FOCUSED_VARIANT,
                    "candidate_top_k": candidate,
                    "final_top_k": final,
                    "full_total_recall": scores_by_depth[(candidate, final)],
                }
            )
            rows.append(row)
        frame = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
        target = Path(results_dir)
        target.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(target / "rag_recall.xlsx", engine="openpyxl") as writer:
            frame.to_excel(writer, sheet_name="recall", index=False)
            pd.DataFrame(columns=["dataset"]).to_excel(
                writer, sheet_name="missing_hypotheses", index=False
            )
        return frame

    monkeypatch.setattr(
        "LCD.shared.rag.depth_sweep.run_rag_evaluation", fake_evaluation
    )

    report = run_rag_depth_sweep(
        artifact_root=artifact_root,
        rag_dir=rag_dir,
        results_dir=results,
    )

    assert report == (results / "rag_recall.xlsx").resolve()
    combined = pd.read_excel(report, sheet_name="recall")
    assert list(
        zip(
            combined["artifact_run"],
            combined["candidate_top_k"],
            combined["final_top_k"],
        )
    ) == [
        (run.artifact_run, run.candidate_top_k, run.final_top_k)
        for run in DEFAULT_STAGE_TWO_DEPTH_RUNS
    ]
    assert loaded_rerankers == [
        "sbert_legal_v1",
        "sbert_legal_60",
        "sbert_legal_40",
    ]
    assert [call[0] for call in calls] == loaded_rerankers
    assert calls[0][1] == (
        (100, 80),
        (40, 20),
        (20, 10),
        (100, 60),
        (100, 40),
        (100, 20),
        (100, 10),
    )
    assert all(
        call[2]["evaluation_variants"] == (FOCUSED_VARIANT,) for call in calls
    )
    best = combined[combined["is_best_full"]]
    assert set(zip(best["candidate_top_k"], best["final_top_k"])) == {
        (80, 60),
        (60, 40),
    }
    assert set(best["full_total_recall_rank"]) == {1}
    assert stage_one.read_text(encoding="utf-8") == "stage-one"
    config = json.loads((results / "evaluation_config.json").read_text("utf-8"))
    assert config["depth_runs"] == [
        {
            "artifact_run": run.artifact_run,
            "candidate_top_k": run.candidate_top_k,
            "final_top_k": run.final_top_k,
        }
        for run in DEFAULT_STAGE_TWO_DEPTH_RUNS
    ]
