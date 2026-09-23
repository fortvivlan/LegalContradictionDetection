"""OpenAI-compatible llama.cpp server prediction support."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

from .common import Task
from .inference import ModelPrediction
from .prompting import build_messages, parse_generated_label


class LlamaCppServerError(RuntimeError):
    """Raised when a llama.cpp server request or response is invalid."""


class LlamaCppServerPredictor:
    """Predict labels through llama-server's OpenAI-compatible chat endpoint.

    The server owns the GGUF model and GPU. This client sends the same system
    and user messages as the Transformers predictor and never logs prompt text.
    """

    def __init__(
        self,
        model_id: str,
        task: Task,
        *,
        base_url: str = "http://127.0.0.1:8080",
        max_new_tokens: int = 16,
        prompt_text: str | None = None,
        seed: int = 42,
        timeout: float = 300.0,
        request_retries: int = 2,
        api_key_env: str = "LLAMA_CPP_API_KEY",
        cache_dir: str | Path | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("model_id cannot be empty")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if request_retries < 0:
            raise ValueError("request_retries cannot be negative")
        self.model_id = model_id
        self.task = task
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        self.max_new_tokens = max_new_tokens
        self.prompt_text = prompt_text
        self.seed = seed
        self.timeout = timeout
        self.request_retries = request_retries
        self.api_key_env = api_key_env
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    def _cached_content(
        self, payload: Mapping[str, Any]
    ) -> tuple[Path | None, str | None]:
        if self.cache_dir is None:
            return None, None
        digest = sha256(
            json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        path = self.cache_dir / f"{digest}.json"
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            content = cached.get("content")
            if isinstance(content, str):
                return path, content
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            AttributeError,
        ):
            pass
        return path, None

    @staticmethod
    def _write_cached_content(path: Path | None, content: str) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"content": content}, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(path)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.api_key_env) if self.api_key_env else None
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _request_json(
        self,
        method: str,
        route: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = (
            json.dumps(payload, ensure_ascii=False).encode("utf-8")
            if payload is not None
            else None
        )
        request = urllib.request.Request(
            f"{self.base_url}{route}",
            data=body,
            headers=self._headers(),
            method=method,
        )
        for attempt in range(self.request_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                if not isinstance(result, dict):
                    raise LlamaCppServerError(
                        f"llama.cpp returned a non-object response from {route}"
                    )
                return result
            except urllib.error.HTTPError as error:
                retryable = error.code == 429 or error.code >= 500
                message = f"llama.cpp HTTP {error.code} from {route}"
                caught: Exception = LlamaCppServerError(message)
            except (urllib.error.URLError, TimeoutError) as error:
                retryable = True
                detail = (
                    error.reason
                    if isinstance(error, urllib.error.URLError)
                    else error
                )
                caught = LlamaCppServerError(
                    f"Could not reach llama.cpp at {self.base_url}: {detail}"
                )
            except (UnicodeError, json.JSONDecodeError):
                retryable = False
                caught = LlamaCppServerError(
                    f"llama.cpp returned invalid JSON from {route}"
                )
            if not retryable or attempt == self.request_retries:
                raise caught
            time.sleep(min(2**attempt, 4))
        raise AssertionError("unreachable")

    def check_server(self) -> None:
        """Fail early unless an OpenAI-compatible model endpoint is ready."""
        result = self._request_json("GET", "/v1/models")
        models = result.get("data")
        if not isinstance(models, list):
            raise LlamaCppServerError(
                "llama.cpp /v1/models response does not contain a model list"
            )
        model_ids = {
            item.get("id") for item in models if isinstance(item, dict)
        }
        if self.model_id not in model_ids:
            raise LlamaCppServerError(
                f"llama.cpp is not serving the requested alias {self.model_id!r}; "
                f"available aliases: {sorted(str(value) for value in model_ids)}"
            )

    def _predict_one(self, premise: str, hypothesis: str) -> ModelPrediction:
        payload = {
            "model": self.model_id,
            "messages": build_messages(
                premise,
                hypothesis,
                self.task,
                prompt_text=self.prompt_text,
            ),
            "max_tokens": self.max_new_tokens,
            "temperature": 0,
            "seed": self.seed,
            "stream": False,
            "chat_template_kwargs": {
                "enable_thinking": False,
                "preserve_thinking": False,
            },
        }
        cache_path, text = self._cached_content(payload)
        if text is None:
            response = self._request_json("POST", "/v1/chat/completions", payload)
            try:
                text = response["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as error:
                raise LlamaCppServerError(
                    "llama.cpp chat response has no assistant content"
                ) from error
            if not isinstance(text, str):
                raise LlamaCppServerError(
                    "llama.cpp assistant content is not plain text"
                )
            self._write_cached_content(cache_path, text)
        return ModelPrediction(parse_generated_label(text, self.task), text)

    def predict_examples(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
        *,
        progress_description: str | None = None,
    ) -> list[ModelPrediction]:
        """Predict aligned examples sequentially through one server slot."""
        if len(premises) != len(hypotheses):
            raise ValueError("Premises and hypotheses must have equal length")
        examples: Any = zip(premises, hypotheses)
        if progress_description is not None:
            from tqdm.auto import tqdm

            examples = tqdm(
                examples,
                total=len(premises),
                desc=progress_description,
                unit="request",
            )
        return [
            self._predict_one(premise, hypothesis)
            for premise, hypothesis in examples
        ]

    def predict_pairs(
        self, premises: Sequence[str], hypothesis: str
    ) -> list[ModelPrediction]:
        """Predict several retrieved premises against one hypothesis."""
        return self.predict_examples(premises, [hypothesis] * len(premises))
