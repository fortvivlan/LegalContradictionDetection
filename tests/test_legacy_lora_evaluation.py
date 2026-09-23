from pathlib import Path

from LCD.experiments.baselines.series_02_lora_rag_comparison import legacy_lora_evaluation


def test_legacy_evaluation_passes_explicit_local_sources(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return "complete"

    monkeypatch.setattr(legacy_lora_evaluation, "run_full_pipeline_evaluation", fake_run)
    result = legacy_lora_evaluation.run_legacy_lora_evaluation(
        models_source=tmp_path / "models.json",
        rag_source=tmp_path / "rag",
        results_dir=tmp_path / "results",
        candidate_top_k=40,
        final_top_k=20,
    )
    assert result == "complete"
    assert captured["models_source"] == tmp_path / "models.json"
    assert captured["rag_source"] == tmp_path / "rag"
    assert captured["reranker_mode"] == "bundle"
    assert captured["inference_parameters"] == {
        "candidate_top_k": 40,
        "final_top_k": 20,
    }
