"""Build a cosine-similarity FAISS index for a sentence encoder."""

from __future__ import annotations

from pathlib import Path
from LCD.shared.common import REPOSITORY_ROOT, DEFAULT_RAG_DIR


def build_faiss_index(
    codex_path: str | Path,
    embedding_model: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cpu",
    batch_size: int = 32,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    revision: str | None = None,
    token: str | None = None,
    trust_remote_code: bool = False,
    precision: str = "float32",
    query_prefix: str = "",
    document_prefix: str = "",
    embeddings=None,
) -> dict[str, int]:
    """Embed codex chunks and save a LangChain-compatible FAISS index."""
    import faiss
    import numpy as np
    import pandas as pd
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from .embeddings import SentenceTransformerEmbeddings

    if batch_size <= 0 or chunk_size <= 0:
        raise ValueError("batch_size and chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
    corpus = pd.read_csv(Path(codex_path))
    if not {"text", "source"}.issubset(corpus.columns):
        raise ValueError("codex.csv must contain text and source columns")
    if corpus[["text", "source"]].isna().any().any():
        raise ValueError("codex.csv contains missing text or source values")
    corpus = corpus.drop_duplicates(subset=["text", "source"], keep="first")
    documents = [
        Document(page_content=str(row.text), metadata={"source": str(row.source)})
        for row in corpus.itertuples(index=False)
    ]
    split_chunks = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    ).split_documents(documents)
    chunks = []
    seen_chunks: set[tuple[str, str]] = set()
    for chunk in split_chunks:
        key = (
            str(chunk.page_content),
            str(chunk.metadata.get("source", "")),
        )
        if key in seen_chunks:
            continue
        seen_chunks.add(key)
        chunks.append(chunk)
    if embeddings is None:
        embeddings = SentenceTransformerEmbeddings(
            embedding_model,
            revision=revision,
            token=token,
            trust_remote_code=trust_remote_code,
            device=device,
            precision=precision,
            batch_size=batch_size,
            normalize_embeddings=True,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
            show_progress=True,
        )
    vectors = np.asarray(
        embeddings.embed_documents([chunk.page_content for chunk in chunks]),
        dtype=np.float32,
    )
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise ValueError(f"Unexpected embedding matrix shape: {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise ValueError("Embedding matrix contains non-finite values")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    ids = {position: str(position) for position in range(len(chunks))}
    store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(
            {str(position): chunk for position, chunk in enumerate(chunks)}
        ),
        index_to_docstore_id=ids,
    )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    store.save_local(str(target))
    return {"corpus_rows": len(documents), "chunks": len(chunks), "dimension": vectors.shape[1]}
