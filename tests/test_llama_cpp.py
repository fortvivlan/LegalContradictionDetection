import json
from pathlib import Path

import pytest

from LCD.shared.llama_cpp import LlamaCppServerError, LlamaCppServerPredictor
from LCD.shared.model_discovery import resolve_models_source


def test_llama_cpp_predictor_sends_chat_messages_and_disables_thinking(
    monkeypatch, tmp_path: Path
) -> None:
    predictor = LlamaCppServerPredictor(
        "Qwen/Qwen3.8-27B",
        "ternary",
        base_url="http://localhost:8080/v1",
        prompt_text="instruction",
        seed=17,
        cache_dir=tmp_path / "cache",
    )
    observed = {}

    def fake_request(method, route, payload=None):
        observed.update({"method": method, "route": route, "payload": payload})
        return {
            "choices": [
                {"message": {"content": "<think>hidden</think> contradiction"}}
            ]
        }

    monkeypatch.setattr(predictor, "_request_json", fake_request)

    prediction = predictor.predict_examples(["norm"], ["decision"])[0]
    cached_prediction = predictor.predict_examples(["norm"], ["decision"])[0]

    assert predictor.base_url == "http://localhost:8080"
    assert prediction.label == "contradiction"
    assert cached_prediction == prediction
    assert observed["method"] == "POST"
    assert observed["route"] == "/v1/chat/completions"
    payload = observed["payload"]
    assert payload["model"] == "Qwen/Qwen3.8-27B"
    assert payload["temperature"] == 0
    assert payload["seed"] == 17
    assert payload["chat_template_kwargs"] == {
        "enable_thinking": False,
        "preserve_thinking": False,
    }
    assert payload["messages"] == [
        {"role": "system", "content": "instruction"},
        {
            "role": "user",
            "content": "Предпосылка: norm\nГипотеза: decision",
        },
    ]
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1


def test_llama_cpp_predictor_rejects_malformed_response(monkeypatch) -> None:
    predictor = LlamaCppServerPredictor("model", "ternary")
    monkeypatch.setattr(predictor, "_request_json", lambda *args, **kwargs: {})

    with pytest.raises(LlamaCppServerError, match="assistant content"):
        predictor.predict_pairs(["premise"], "hypothesis")


def test_llama_cpp_server_check_requires_requested_alias(monkeypatch) -> None:
    predictor = LlamaCppServerPredictor("expected-model", "ternary")
    monkeypatch.setattr(
        predictor,
        "_request_json",
        lambda *args, **kwargs: {"data": [{"id": "another-model"}]},
    )

    with pytest.raises(LlamaCppServerError, match="requested alias"):
        predictor.check_server()


def test_llama_cpp_manifest_and_full_dataset_selection(tmp_path: Path) -> None:
    manifest = tmp_path / "models.json"
    manifest.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "name": "qwen",
                        "family": "base_llm",
                        "backend": "llama_cpp",
                        "tasks": ["ternary"],
                        "model_id": "Qwen/Qwen3.8-27B",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    models = resolve_models_source(manifest)
    assert models[0].backend == "llama_cpp"
    assert models[0].task == "ternary"



def test_llama_cpp_backend_is_base_llm_only(tmp_path: Path) -> None:
    manifest = tmp_path / "models.json"
    manifest.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "family": "bert",
                        "backend": "llama_cpp",
                        "task": "ternary",
                        "path": "artifact",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="family='base_llm'"):
        resolve_models_source(manifest)
