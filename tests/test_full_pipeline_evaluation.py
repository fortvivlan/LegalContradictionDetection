import json
from pathlib import Path

import pandas as pd

from LCD.shared.full_pipeline import (
    _evict_huggingface_model_revision,
    _include_source_prefix,
    _load_adapter_tokenizer,
    run_full_pipeline_evaluation,
)
from LCD.shared.rag.artifacts import RagBundle, RerankerBundle


def test_source_prefix_policy_is_limited_to_generative_models() -> None:
    assert not _include_source_prefix("bert")
    assert _include_source_prefix("base_llm")
    assert _include_source_prefix("lora")


def test_huggingface_cache_eviction_targets_only_exact_revision(monkeypatch) -> None:
    class Revision:
        def __init__(self, commit_hash: str) -> None:
            self.commit_hash = commit_hash

    class Repo:
        repo_type = "model"

        def __init__(self, repo_id: str, revisions: tuple[Revision, ...]) -> None:
            self.repo_id = repo_id
            self.revisions = revisions

    class Strategy:
        expected_freed_size = 123
        executed = False

        def execute(self) -> None:
            self.executed = True

    strategy = Strategy()

    class Cache:
        repos = {
            Repo("meta-llama/Llama-3.1-8B", (Revision("wanted"),)),
            Repo("Qwen/Qwen3-8B", (Revision("other"),)),
        }
        deleted: tuple[str, ...] = ()

        def delete_revisions(self, *revisions: str):
            self.deleted = revisions
            return strategy

    cache = Cache()
    monkeypatch.setattr("huggingface_hub.scan_cache_dir", lambda: cache)

    freed = _evict_huggingface_model_revision(
        "meta-llama/Llama-3.1-8B", "wanted"
    )

    assert freed == 123
    assert cache.deleted == ("wanted",)
    assert strategy.executed is True


def test_legacy_adapter_tokenizer_falls_back_to_base(
    monkeypatch, tmp_path: Path
) -> None:
    class Tokenizer:
        pad_token = None
        eos_token = "</s>"
        padding_side = "right"

    fallback = Tokenizer()

    def incompatible_tokenizer(*args, **kwargs):
        raise ValueError("Tokenizer class TokenizersBackend does not exist")

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", incompatible_tokenizer
    )
    loaded = _load_adapter_tokenizer(
        tmp_path,
        base_tokenizer=fallback,
        trust_remote_code=False,
    )

    assert loaded is fallback
    assert loaded.pad_token == "</s>"
    assert loaded.padding_side == "left"


def test_full_pipeline_recovers_corrupt_state_and_resumes_completed_jobs(
    monkeypatch, tmp_path: Path
) -> None:
    manifest = tmp_path / "models.json"
    manifest.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "name": "bert-ternary",
                        "family": "bert",
                        "task": "ternary",
                        "path": "artifact",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "LCD" / "shared").mkdir(parents=True, exist_ok=True)
    (tmp_path / "LCD" / "shared" / "prompt.py").write_text('PROMPT_TEXT = "t"', encoding="utf-8")
    bundle = RagBundle(
        "test-rag", tmp_path / "codex.csv", tmp_path / "index", "encoder", None, False
    )
    monkeypatch.setattr("LCD.shared.full_pipeline.load_rag_bundle", lambda path: bundle)
    monkeypatch.setattr(
        "LCD.shared.full_pipeline.PremiseRetriever.from_components",
        lambda *args, **kwargs: object(),
    )
    calls = []
    monkeypatch.setattr(
        "LCD.shared.full_pipeline._load_predictor",
        lambda entry, prompt, parameters: (object(), object(), object()),
    )

    def fake_evaluate(entry, **kwargs):
        calls.append(entry.name)
        return {
            "scores": pd.DataFrame(
                [{"model": entry.name, "task": entry.task, "evaluation_scope": "validation"}]
            ),
            "details": pd.DataFrame([{"value": 1}]),
        }

    monkeypatch.setattr("LCD.shared.full_pipeline._evaluate_one", fake_evaluate)
    results = tmp_path / "results"

    first = run_full_pipeline_evaluation(
        models_source=manifest,
        rag_source=tmp_path / "rag.json",
        repo_root=tmp_path,
        results_dir=results,
    )
    (results / "state.json").write_text("", encoding="utf-8")
    second = run_full_pipeline_evaluation(
        models_source=manifest,
        rag_source=tmp_path / "rag.json",
        repo_root=tmp_path,
        results_dir=results,
    )
    third = run_full_pipeline_evaluation(
        models_source=manifest,
        rag_source=tmp_path / "rag.json",
        repo_root=tmp_path,
        results_dir=results,
    )

    assert calls == ["bert-ternary"]
    assert first["model_name"].tolist() == ["bert-ternary"]
    assert second["model_name"].tolist() == ["bert-ternary"]
    assert third["model_name"].tolist() == ["bert-ternary"]
    assert list(results.glob("state.corrupt-*.json"))
    recovered_state = json.loads((results / "state.json").read_text(encoding="utf-8"))
    assert recovered_state["jobs"]["bert-ternary"]["status"] == "completed"
    assert recovered_state["jobs"]["bert-ternary"]["recovered_from_outputs"] is True


def test_legacy_retrieval_top_k_override_sets_both_depths(
    monkeypatch, tmp_path: Path
) -> None:
    manifest = tmp_path / "models.json"
    manifest.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "name": "bert-ternary",
                        "family": "bert",
                        "task": "ternary",
                        "path": "artifact",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "LCD" / "shared").mkdir(parents=True, exist_ok=True)
    (tmp_path / "LCD" / "shared" / "prompt.py").write_text('PROMPT_TEXT = "t"', encoding="utf-8")
    bundle = RagBundle(
        "test-rag", tmp_path / "codex.csv", tmp_path / "index", "encoder", None, False
    )
    monkeypatch.setattr("LCD.shared.full_pipeline.load_rag_bundle", lambda path: bundle)
    monkeypatch.setattr(
        "LCD.shared.full_pipeline.PremiseRetriever.from_components",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "LCD.shared.full_pipeline._load_predictor",
        lambda entry, prompt, parameters: (object(), object(), object()),
    )
    observed = {}

    def fake_evaluate(entry, **kwargs):
        observed.update(kwargs["parameters"])
        return {"scores": pd.DataFrame([{"model": entry.name}])}

    monkeypatch.setattr("LCD.shared.full_pipeline._evaluate_one", fake_evaluate)

    run_full_pipeline_evaluation(
        models_source=manifest,
        rag_source=tmp_path / "rag.json",
        repo_root=tmp_path,
        results_dir=tmp_path / "results",
        inference_parameters={"retrieval_top_k": 37},
    )

    assert observed["candidate_top_k"] == 37
    assert observed["final_top_k"] == 37


def test_full_pipeline_forwards_asymmetric_embedding_bundle_options(
    monkeypatch, tmp_path: Path
) -> None:
    manifest = tmp_path / "models.json"
    manifest.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "name": "bert-ternary",
                        "family": "bert",
                        "task": "ternary",
                        "path": "artifact",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "LCD" / "shared").mkdir(parents=True, exist_ok=True)
    (tmp_path / "LCD" / "shared" / "prompt.py").write_text('PROMPT_TEXT = "t"', encoding="utf-8")
    reranker_path = tmp_path / "reranker"
    reranker_path.mkdir()
    bundle = RagBundle(
        "qwen-rag",
        tmp_path / "codex.csv",
        tmp_path / "index",
        "qwen-embedding",
        None,
        True,
        embedding_query_prefix="query:",
        embedding_document_prefix="passage:",
        embedding_trust_remote_code=True,
        embedding_precision="bfloat16",
        embedding_batch_size=16,
        reranker=RerankerBundle(
            "finetuned", str(reranker_path), None, True, 1024
        ),
    )
    monkeypatch.setattr(
        "LCD.shared.full_pipeline.load_rag_bundle", lambda path: bundle
    )
    observed = {}

    def fake_retriever(*args, **kwargs):
        observed.update(kwargs)
        return object()

    monkeypatch.setattr(
        "LCD.shared.full_pipeline.PremiseRetriever.from_components",
        fake_retriever,
    )
    monkeypatch.setattr(
        "LCD.shared.full_pipeline._load_predictor",
        lambda entry, prompt, parameters: (object(), object(), object()),
    )
    monkeypatch.setattr(
        "LCD.shared.full_pipeline._evaluate_one",
        lambda entry, **kwargs: {"scores": pd.DataFrame([{"model": entry.name}])},
    )

    class FakeReranker:
        def __init__(self, model_id, **kwargs):
            self.model_id = model_id
            self.kwargs = kwargs

    monkeypatch.setattr(
        "LCD.shared.full_pipeline.CrossEncoderReranker", FakeReranker
    )

    run_full_pipeline_evaluation(
        models_source=manifest,
        rag_source=tmp_path / "rag-qwen",
        reranker_mode="bundle",
        repo_root=tmp_path,
        results_dir=tmp_path / "results",
    )

    assert observed["embedding_query_prefix"] == "query:"
    assert observed["embedding_document_prefix"] == "passage:"
    assert observed["embedding_trust_remote_code"] is True
    assert observed["embedding_precision"] == "bfloat16"
    assert observed["embedding_batch_size"] == 16
    assert observed["reranker"].model_id == str(reranker_path)
    assert observed["reranker"].kwargs["trust_remote_code"] is True
