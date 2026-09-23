"""Dataset conversion helpers for legal sentence-embedding training."""

from __future__ import annotations

import hashlib
from pathlib import Path
from LCD.shared.common import REPOSITORY_ROOT, DEFAULT_RAG_DIR

EMBEDDING_LABELS = {
    "contradiction": ("similar", 1),
    "entailment": ("similar", 1),
    "not mentioned": ("not mentioned", 0),
}


def convert_embedding_dataset(path: str | Path, split: str):
    """Load an XLSX split and map three-class tags to retrieval relevance."""
    import pandas as pd

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Embedding dataset does not exist: {source}")
    frame = pd.read_excel(source, engine="openpyxl", dtype=str, keep_default_na=False)
    required = ("premise", "hypothesis", "source", "tag")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing columns: {', '.join(missing)}")
    frame = frame.loc[:, required].copy()
    unknown = sorted(set(frame["tag"]) - set(EMBEDDING_LABELS))
    if unknown:
        raise ValueError(f"Unsupported source labels in {source}: {unknown}")
    if frame[["premise", "hypothesis", "tag"]].eq("").any().any():
        raise ValueError(f"{source} contains blank model inputs or labels")
    frame.insert(0, "example_id", [f"{split}:{index:06d}" for index in range(len(frame))])
    mapped = frame["tag"].map(EMBEDDING_LABELS)
    frame["embedding_tag"] = mapped.map(lambda value: value[0])
    frame["label"] = mapped.map(lambda value: value[1]).astype(int)
    return frame


def dataframe_sha256(frame) -> str:
    """Return a stable UTF-8 hash of a converted dataframe."""
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
