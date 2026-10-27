"""One-vector-per-provision FAISS retrieval with JSON metadata and no pickle."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import tempfile

from .corpus import UNIT_FIELDS, sha256, write_json

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QUERY_INSTRUCTION = (
    "Given a statement from a Russian court decision, retrieve provisions of Russian law "
    "that regulate the same legal issue, including rules that may support or contradict "
    "the statement, while prioritizing matching actors, actions, conditions, and procedural stages."
)


def query_text(statement: str, instruction: str = QUERY_INSTRUCTION) -> str:
    """Apply exactly one Qwen instruction to a plain statement."""
    if not statement.strip():
        raise ValueError("Query statement must not be empty")
    return f"Instruct: {instruction}\nQuery: {statement}"


class QwenEncoder:
    """Local Sentence Transformers encoder with explicit token-length accounting."""

    def __init__(self, model: str = DEFAULT_MODEL, *, revision: str | None = None,
                 device: str = "cuda", precision: str = "auto", batch_size: int = 8,
                 seed: int = 42, cache_dir: str | Path | None = None,
                 instruction: str = QUERY_INSTRUCTION, local_files_only: bool = False):
        import torch
        from sentence_transformers import SentenceTransformer
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if precision not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("Unknown precision")
        torch.manual_seed(seed)
        if precision == "auto":
            precision = ("bfloat16" if torch.cuda.is_available() and device.startswith("cuda")
                         and torch.cuda.is_bf16_supported() else
                         "float16" if torch.cuda.is_available() and device.startswith("cuda") else "float32")
        path = Path(model)
        local = path.is_dir()
        if local:
            model = str(path.resolve())
            digest = hashlib.sha256()
            for file in sorted(p for p in path.rglob("*") if p.is_file()):
                digest.update(file.relative_to(path).as_posix().encode())
                digest.update(sha256(file).encode())
            resolved_revision = "local:" + digest.hexdigest()
        elif revision and re.fullmatch(r"[a-fA-F0-9]{40}", revision):
            resolved_revision = revision
        elif local_files_only:
            from huggingface_hub import snapshot_download
            snapshot = snapshot_download(model, revision=revision, cache_dir=cache_dir,
                                         local_files_only=True)
            resolved_revision = Path(snapshot).name
        else:
            from huggingface_hub import HfApi
            resolved_revision = HfApi().model_info(model, revision=revision).sha
        if local:
            load_path = model
        else:
            # Load a complete local snapshot so secondary tokenizer/config reads
            # cannot fall back to a machine-wide Hugging Face cache.
            from huggingface_hub import snapshot_download
            load_path = snapshot_download(
                model, revision=resolved_revision, cache_dir=cache_dir,
                local_files_only=local_files_only,
                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"],
            )
        self.model = SentenceTransformer(
            load_path, device=device,
            cache_folder=str(cache_dir) if cache_dir else None,
            local_files_only=True,
            model_kwargs={"torch_dtype": getattr(torch, precision)},
            tokenizer_kwargs={"padding_side": "left"},
        )
        self.model.eval()
        self.batch_size, self.instruction = batch_size, instruction
        self.max_tokens = int(self.model.max_seq_length)
        self.configuration = {"model": model, "revision": resolved_revision,
                              "device": device, "precision": precision, "seed": seed,
                              "batch_size": batch_size, "query_instruction": instruction,
                              "document_instruction": "", "max_tokens": self.max_tokens,
                              "normalize_embeddings": True, "dimension": 1024}

    def token_lengths(self, texts: list[str]) -> list[int]:
        """Count actual model inputs without tokenizer truncation."""
        return [len(tokens) for tokens in self.model.tokenizer(
            texts, add_special_tokens=True, truncation=False, padding=False,
            return_attention_mask=False)["input_ids"]]

    def encode_documents(self, texts: list[str]):
        """Embed untouched document text; suppress any model-level default prompt."""
        return self.model.encode(texts, prompt="", batch_size=self.batch_size,
                                 normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=True)

    def encode_query(self, statement: str):
        """Embed a statement using the instruction recorded in the corpus manifest."""
        prepared = query_text(statement, self.instruction)
        _check_lengths(self, [prepared], ["query"])
        return self.model.encode([prepared], prompt="", batch_size=1,
                                 normalize_embeddings=True, convert_to_numpy=True)[0]


def _check_lengths(encoder, texts, citations):
    lengths = encoder.token_lengths(texts)
    if len(lengths) != len(texts):
        raise ValueError("Tokenizer returned an unexpected number of lengths")
    oversized = [(source, length) for source, length in zip(citations, lengths)
                 if length > encoder.max_tokens]
    if oversized:
        raise ValueError(f"Inputs exceed {encoder.max_tokens} tokens; no truncation allowed: {oversized}")
    return lengths


def _read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _validate_vectors(vectors, count: int):
    import numpy as np
    array = np.asarray(vectors, dtype="float32")
    if array.ndim != 2 or array.shape[0] != count or not array.shape[1]:
        raise ValueError(f"Unexpected embedding shape: {array.shape}")
    if not np.isfinite(array).all() or (np.linalg.norm(array, axis=1) == 0).any():
        raise ValueError("Embedding vectors must be finite and nonzero")
    return np.ascontiguousarray(array)


def build_index(corpus_dir: str | Path, *, encoder=None, **encoder_options) -> dict:
    """Validate exported CSV rows, embed each once and save an exact cosine index.

    ``encoder`` is injectable for offline tests and must expose configuration,
    max_tokens, token_lengths, encode_documents and encode_query. Public artifacts
    are committed only after encoding and FAISS validation complete.
    """
    import faiss
    target = Path(corpus_dir)
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    if (target / "faiss_index").exists():
        raise ValueError("Index already exists; select a fresh corpus output directory")
    for name, digest in manifest["files"].items():
        if sha256(target / name) != digest:
            raise ValueError(f"Corpus hash mismatch: {name}")
    records = _read_records(target / "units.jsonl")
    with (target / "koap_units.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(UNIT_FIELDS):
            raise ValueError("Unexpected main CSV columns")
        rows = list(reader)
    if not rows or len(rows) != len(records):
        raise ValueError("Empty or mismatched main CSV/metadata")
    if any(any(row[k] != record[k] for k in UNIT_FIELDS) for row, record in zip(rows, records)):
        raise ValueError("CSV and metadata contents differ")
    encoder = encoder or QwenEncoder(**encoder_options)
    texts = [r["text"] for r in records]
    lengths = _check_lengths(encoder, texts, [r["source"] for r in records])
    vectors = _validate_vectors(encoder.encode_documents(texts), len(records))
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    for record, length in zip(records, lengths):
        record["token_count"] = length
    with tempfile.TemporaryDirectory(prefix=".index-build-", dir=target) as temporary:
        stage = Path(temporary) / "faiss_index"
        stage.mkdir()
        faiss.write_index(index, str(stage / "index.faiss"))
        with (stage / "units.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        if faiss.read_index(str(stage / "index.faiss")).ntotal != len(rows):
            raise ValueError("FAISS round-trip count mismatch")
        stage.rename(target / "faiss_index")
    versions = {}
    for name in ("faiss-cpu", "sentence-transformers", "transformers", "torch", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    manifest["index"] = {**encoder.configuration, "dimension": int(vectors.shape[1]),
                         "vectors": len(records), "metric": "cosine", "index_type": "IndexFlatIP",
                         "max_observed_tokens": max(lengths), "packages": versions}
    manifest["files"].update({name: sha256(target / name) for name in
                              ("faiss_index/index.faiss", "faiss_index/units.jsonl")})
    write_json(target / "manifest.json", manifest)
    return manifest


class CodeRagStore:
    """Read-only legal-unit lookup and dense search; JSON metadata needs no pickle."""

    def __init__(self, corpus_dir: str | Path, *, encoder=None, device: str = "cpu",
                 cache_dir: str | Path | None = None, local_files_only: bool = False):
        import faiss
        target = Path(corpus_dir)
        self.manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        configuration = self.manifest["index"]
        for name in ("faiss_index/index.faiss", "faiss_index/units.jsonl"):
            if sha256(target / name) != self.manifest["files"][name]:
                raise ValueError(f"Index hash mismatch: {name}")
        self.records = _read_records(target / "faiss_index/units.jsonl")
        self.index = faiss.read_index(str(target / "faiss_index/index.faiss"))
        if (self.index.ntotal != len(self.records) or self.index.ntotal != configuration["vectors"]
                or self.index.d != configuration["dimension"]
                or self.index.metric_type != faiss.METRIC_INNER_PRODUCT):
            raise ValueError("Index/metadata dimensions, counts or metric differ")
        if len({r["unit_id"] for r in self.records}) != len(self.records):
            raise ValueError("Duplicate legal-unit IDs in index metadata")
        if any(r["source_sha256"] != self.manifest["source_sha256"] for r in self.records):
            raise ValueError("Mixed source editions in index metadata")
        self._encoder = encoder
        self._encoder_options = {"model": configuration["model"],
                                 "revision": configuration["revision"], "device": device,
                                 "precision": configuration["precision"],
                                 "instruction": configuration["query_instruction"],
                                 "cache_dir": cache_dir, "local_files_only": local_files_only}
        if encoder is not None:
            self._validate_encoder(encoder)

    def _validate_encoder(self, encoder):
        for key in ("model", "revision", "query_instruction"):
            if encoder.configuration[key] != self.manifest["index"][key]:
                raise ValueError(f"Query encoder differs from index: {key}")

    def exact_lookup(self, article: str, *, part: str | None = None,
                     point: str | None = None) -> list[dict]:
        """Return an article's leaves, narrowed by any explicitly supplied part/point."""
        return [{**r, "method": "exact", "score": None} for r in self.records
                if r["article"] == article and (part is None or r["part"] == part)
                and (point is None or r["point"] == point)]

    def search(self, statement: str, *, top_k: int = 10) -> list[dict]:
        """Return complete legal units with cosine scores and source provenance."""
        import faiss
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_text(statement)  # Validate even when an encoder is injected.
        if self._encoder is None:
            encoder = QwenEncoder(**self._encoder_options)
            self._validate_encoder(encoder)
            self._encoder = encoder
        vector = _validate_vectors([self._encoder.encode_query(statement)], 1)
        if vector.shape[1] != self.index.d:
            raise ValueError("Query vector dimension differs from index")
        faiss.normalize_L2(vector)
        scores, ids = self.index.search(vector, min(top_k, len(self.records)))
        return [{**self.records[int(index)], "method": "faiss", "rank": rank,
                 "score": float(score)}
                for rank, (score, index) in enumerate(zip(scores[0], ids[0]), 1) if index >= 0]
