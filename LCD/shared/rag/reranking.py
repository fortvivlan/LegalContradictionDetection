"""Cross-encoder scoring for second-stage semantic retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any
from typing import Protocol


class Reranker(Protocol):
    """Minimal interface accepted by the premise retriever."""

    model_id: str
    revision: str | None

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Return one relevance score per query/document pair."""


class CrossEncoderReranker:
    """Batched Sentence Transformers cross-encoder reranker."""

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        token: str | None = None,
        trust_remote_code: bool = False,
        device: str | None = None,
        precision: str = "auto",
        batch_size: int = 8,
        max_length: int = 1024,
    ) -> None:
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("reranker batch_size and max_length must be positive")
        import torch
        from sentence_transformers import CrossEncoder

        normalized_precision = precision.strip().lower()
        if normalized_precision not in {"auto", "float16", "bfloat16", "float32"}:
            raise ValueError(
                "reranker precision must be auto, float16, bfloat16, or float32"
            )
        if normalized_precision == "bfloat16":
            dtype = torch.bfloat16
        elif normalized_precision == "float16":
            dtype = torch.float16
        elif normalized_precision == "float32":
            dtype = torch.float32
        elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif torch.cuda.is_available():
            dtype = torch.float16
        else:
            dtype = torch.float32
        self.model_id = model_id
        self.revision = revision
        self.batch_size = batch_size
        self.model = CrossEncoder(
            model_id,
            revision=revision,
            token=token,
            trust_remote_code=trust_remote_code,
            device=device,
            max_length=max_length,
            model_kwargs={"dtype": dtype},
        )

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Score hypothesis-as-query against premise-as-document pairs."""
        if not documents:
            return []
        values = self.model.predict(
            [(query, document) for document in documents],
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [float(value) for value in values]


def score_and_sort_records(
    query: str,
    records: Sequence[Any],
    reranker: Reranker,
) -> tuple[Any, ...]:
    """Score and stably reorder every candidate without discarding audit data."""
    scores = reranker.score(query, [record.premise for record in records])
    if len(scores) != len(records):
        raise ValueError("Reranker returned a different number of scores than candidates")
    scored = [
        replace(record, reranker_score=float(score))
        for record, score in zip(records, scores)
    ]
    scored.sort(
        key=lambda record: (
            -float(record.reranker_score),
            int(record.initial_rank or record.rank),
        )
    )
    return tuple(
        replace(record, rank=rank)
        for rank, record in enumerate(scored, start=1)
    )


def rerank_records(
    query: str,
    records: Sequence[Any],
    reranker: Reranker,
    *,
    final_top_k: int,
) -> tuple[Any, ...]:
    """Score, stably reorder, and truncate retrieval records."""
    if final_top_k <= 0:
        raise ValueError("final_top_k must be positive")
    return score_and_sort_records(query, records, reranker)[:final_top_k]
