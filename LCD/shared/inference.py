"""Shared full-document RAG inference and aggregation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from .common import LABELS_BY_TASK, Task, file_sha256
from .documents import (
    extract_operative_section,
    read_docx_text,
    split_russian_sentences,
    textcheck,
)
from .retrieval import PremiseRetriever, citation_dict, citations_json

BODY_ONLY_PREMISE_FORMAT = "body_only_v1"
SOURCE_PREFIXED_PREMISE_FORMAT = "source_prefixed_v1"


@dataclass(frozen=True)
class ModelPrediction:
    """A parsed prediction and the corresponding raw model output."""

    label: str | None
    raw_output: str = ""


class PairPredictor(Protocol):
    """Minimal predictor interface required by document inference."""

    def predict_pairs(
        self, premises: Sequence[str], hypothesis: str
    ) -> list[ModelPrediction]: ...


def aggregate_pair_labels(labels: Sequence[str | None], task: Task) -> str:
    """Aggregate premise-pair labels with contradiction-first semantics."""
    valid = [label for label in labels if label in LABELS_BY_TASK[task]]
    if "contradiction" in valid:
        return "contradiction"
    if "entailment" in valid:
        return "entailment"
    if "not mentioned" in valid:
        return "not mentioned"
    return "invalid"


def format_model_premise(premise: str, source: str) -> str:
    """Prefix a retrieved provision with its model-visible citation source."""
    body = premise.strip()
    citation = source.strip()
    if not citation or body.startswith(citation):
        return body
    return f"{citation} {body}".strip()


def model_premise_format(include_source_prefix: bool) -> str:
    """Return the stable audit label for a classifier premise policy."""
    return (
        SOURCE_PREFIXED_PREMISE_FORMAT
        if include_source_prefix
        else BODY_ONLY_PREMISE_FORMAT
    )


@dataclass
class DocumentInferenceTables:
    """Aggregate, pair-level, and error tables for uploaded decisions."""

    aggregates: object
    pairs: object
    errors: object


def run_document_inference(
    document_paths: Sequence[Path],
    *,
    predictor: PairPredictor,
    retriever: PremiseRetriever,
    model_id: str,
    task: Task,
    top_k: int = 20,
    final_top_k: int | None = None,
    include_source_prefix: bool = True,
) -> DocumentInferenceTables:
    """Run retrieval and classification, optionally exposing sources to the model."""
    import pandas as pd
    from tqdm.auto import tqdm

    aggregate_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    for document_path in document_paths:
        document_hash = "unavailable"
        try:
            document_hash = file_sha256(document_path)
            full_text = read_docx_text(document_path)
            operative = extract_operative_section(full_text)
            if operative is None:
                error_rows.append(
                    {
                        "model": model_id,
                        "task": task,
                        "document": document_path.name,
                        "document_sha256": document_hash,
                        "stage": "operative_section",
                        "error": "ПОСТАНОВИЛ section was not found; document skipped",
                    }
                )
                continue
            hypotheses = [
                (sentence_index, hypothesis)
                for sentence_index, hypothesis in enumerate(
                    split_russian_sentences(operative)
                )
                if not textcheck(hypothesis)
            ]
            for sentence_index, hypothesis in tqdm(
                hypotheses,
                desc=f"{document_path.name} [{task}]",
                leave=False,
            ):
                hypothesis_id = f"{document_path.name}:{sentence_index:05d}"
                if final_top_k is None:
                    retrieved = retriever.retrieve(hypothesis, top_k=top_k)
                else:
                    retrieved = retriever.retrieve(
                        hypothesis, top_k=top_k, final_top_k=final_top_k
                    )
                if not retrieved:
                    error_rows.append(
                        {
                            "model": model_id,
                            "task": task,
                            "document": document_path.name,
                            "document_sha256": document_hash,
                            "hypothesis_id": hypothesis_id,
                            "stage": "retrieval",
                            "error": "No premises were retrieved",
                        }
                    )
                    continue
                model_premises = [
                    (
                        format_model_premise(record.premise, record.source)
                        if include_source_prefix
                        else record.premise
                    )
                    for record in retrieved
                ]
                predictions = predictor.predict_pairs(model_premises, hypothesis)
                if len(predictions) != len(retrieved):
                    raise RuntimeError("Predictor returned the wrong number of results")
                contradiction_sources: list[str] = []
                for record, model_premise, prediction in zip(
                    retrieved, model_premises, predictions
                ):
                    if prediction.label == "contradiction":
                        contradiction_sources.append(record.source)
                    pair_rows.append(
                        {
                            "model": model_id,
                            "task": task,
                            "document": document_path.name,
                            "document_sha256": document_hash,
                            "hypothesis_id": hypothesis_id,
                            "sentence_index": sentence_index,
                            "hypothesis": hypothesis,
                            "premise": record.premise,
                            "model_premise": model_premise,
                            "source": record.source,
                            "retrieval_method": record.method,
                            "retrieval_rank": record.rank,
                            "retrieval_initial_rank": record.initial_rank,
                            "retrieval_score": record.score,
                            "reranker_score": record.reranker_score,
                            **citation_dict(record.citation),
                            "detected_citations": citations_json(
                                record.detected_citations
                            ),
                            "unresolved_citations": citations_json(
                                record.unresolved_citations
                            ),
                            "prediction": prediction.label or "invalid",
                            "raw_output": prediction.raw_output,
                        }
                    )
                aggregate_rows.append(
                    {
                        "model": model_id,
                        "task": task,
                        "document": document_path.name,
                        "document_sha256": document_hash,
                        "hypothesis_id": hypothesis_id,
                        "sentence_index": sentence_index,
                        "hypothesis": hypothesis,
                        "prediction": aggregate_pair_labels(
                            [prediction.label for prediction in predictions], task
                        ),
                        "retrieved_premises": len(retrieved),
                        "detected_citations": citations_json(
                            retrieved[0].detected_citations
                        ),
                        "unresolved_citations": citations_json(
                            retrieved[0].unresolved_citations
                        ),
                        "contradiction_sources": json.dumps(
                            contradiction_sources, ensure_ascii=False
                        ),
                    }
                )
        except Exception as error:  # keep other uploaded files processable
            error_rows.append(
                {
                    "model": model_id,
                    "task": task,
                    "document": document_path.name,
                    "document_sha256": document_hash,
                    "stage": "document_processing",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return DocumentInferenceTables(
        aggregates=pd.DataFrame(aggregate_rows),
        pairs=pd.DataFrame(pair_rows),
        errors=pd.DataFrame(error_rows),
    )
