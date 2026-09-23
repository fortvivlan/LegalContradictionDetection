"""Sentence-embedding training and retrieval evaluation workflows."""

from .citation_audit import run_citation_audit
from .depth_sweep import run_rag_depth_sweep
from .embedding_sweep import run_rag_embedding_sweep
from .embeddings import DEFAULT_STAGE_THREE_MODELS, EmbeddingModelSpec
from .evaluation import run_rag_evaluation
from .training import run_rag_experiment

__all__ = [
    "DEFAULT_STAGE_THREE_MODELS",
    "EmbeddingModelSpec",
    "run_citation_audit",
    "run_rag_depth_sweep",
    "run_rag_embedding_sweep",
    "run_rag_evaluation",
    "run_rag_experiment",
]
