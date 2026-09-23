"""Model-aware sentence embeddings for asymmetric legal retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from LCD.shared.common import REPOSITORY_ROOT, DEFAULT_RAG_DIR
from typing import Any

try:
    from langchain_core.embeddings import Embeddings as _LangChainEmbeddings
except ModuleNotFoundError:  # Optional RAG dependencies are loaded only at runtime.
    class _LangChainEmbeddings:  # type: ignore[no-redef]
        pass


LEGAL_RETRIEVAL_INSTRUCTION = (
    "Given a Russian legal statement from the operative section of a court "
    "decision, retrieve the relevant provision of Russian law."
)


@dataclass(frozen=True)
class EmbeddingModelSpec:
    """One pretrained embedding model and its asymmetric retrieval prompts."""

    alias: str
    model_id: str
    revision: str | None = None
    query_prefix: str = ""
    document_prefix: str = ""
    trust_remote_code: bool = False


DEFAULT_STAGE_THREE_MODELS: tuple[EmbeddingModelSpec, ...] = (
    EmbeddingModelSpec("bge_m3", "BAAI/bge-m3"),
    EmbeddingModelSpec(
        "qwen3_embedding_0_6b",
        "Qwen/Qwen3-Embedding-0.6B",
        query_prefix=f"Instruct: {LEGAL_RETRIEVAL_INSTRUCTION}\nQuery:",
    ),
    EmbeddingModelSpec(
        "multilingual_e5_large_instruct",
        "intfloat/multilingual-e5-large-instruct",
        query_prefix=f"Instruct: {LEGAL_RETRIEVAL_INSTRUCTION}\nQuery: ",
    ),
)


def _validate_precision(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"auto", "float16", "bfloat16", "float32"}:
        raise ValueError(
            "embedding precision must be auto, float16, bfloat16, or float32"
        )
    return normalized


class SentenceTransformerEmbeddings(_LangChainEmbeddings):
    """LangChain-compatible embeddings with distinct query/document inputs."""

    def __init__(
        self,
        model_name: str | Path,
        *,
        revision: str | None = None,
        token: str | None = None,
        trust_remote_code: bool = False,
        device: str = "cpu",
        precision: str = "float32",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        query_prefix: str = "",
        document_prefix: str = "",
        show_progress: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("embedding batch_size must be positive")
        normalized_precision = _validate_precision(precision)
        import torch
        from sentence_transformers import SentenceTransformer

        if normalized_precision == "bfloat16":
            dtype = torch.bfloat16
        elif normalized_precision == "float16":
            dtype = torch.float16
        elif normalized_precision == "float32":
            dtype = torch.float32
        elif (
            str(device).startswith("cuda")
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        ):
            dtype = torch.bfloat16
        elif str(device).startswith("cuda") and torch.cuda.is_available():
            dtype = torch.float16
        else:
            dtype = torch.float32

        model_path = Path(model_name).expanduser()
        local = model_path.exists()
        model_kwargs: dict[str, Any] = {"torch_dtype": dtype}
        self.model = SentenceTransformer(
            str(model_path.resolve()) if local else str(model_name),
            revision=None if local else revision,
            token=token,
            trust_remote_code=trust_remote_code,
            device=device,
            model_kwargs=model_kwargs,
        )
        self.model_name = str(model_name)
        self.revision = None if local else revision
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.show_progress = show_progress
        self.resolved_dtype = str(dtype).removeprefix("torch.")

    def _embed(self, texts: list[str], prefix: str) -> list[list[float]]:
        if not texts:
            return []
        values = self.model.encode(
            [f"{prefix}{text}" for text in texts],
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=self.show_progress,
            convert_to_numpy=True,
        )
        return values.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed corpus chunks with the configured document prefix."""
        return self._embed(texts, self.document_prefix)

    def embed_query(self, text: str) -> list[float]:
        """Embed one retrieval query with the model-specific instruction."""
        return self._embed([text], self.query_prefix)[0]

    def save_pretrained(self, output_dir: str | Path) -> Path:
        """Save the resolved Sentence Transformers model for offline reuse."""
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(target), safe_serialization=True)
        return target.resolve()
