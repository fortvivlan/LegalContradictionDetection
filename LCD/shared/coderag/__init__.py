"""Local, provision-preserving КоАП extraction and retrieval; no import side effects."""

from .corpus import extract_docx, export_corpus, read_paragraphs
from .index import QUERY_INSTRUCTION, CodeRagStore, build_index

__all__ = ["extract_docx", "export_corpus", "read_paragraphs", "QUERY_INSTRUCTION",
           "CodeRagStore", "build_index"]
