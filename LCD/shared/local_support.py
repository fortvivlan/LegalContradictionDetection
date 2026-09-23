"""Local paths, tokens, and document selection for workflow entrypoints."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence


def get_huggingface_token() -> str | None:
    """Read an optional Hugging Face token from the local environment."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def prepare_artifact_root(root: str | Path) -> Path:
    """Create an explicitly selected local artifact directory."""
    path = Path(root).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def selected_docx_files(
    document_paths: Sequence[str | Path] | None = None,
) -> Iterator[list[Path]]:
    """Validate explicit DOCX inputs without taking ownership of their files."""
    paths = [Path(path).expanduser().resolve() for path in document_paths or ()]
    invalid = [path for path in paths if path.suffix.lower() != ".docx" or not path.is_file()]
    if invalid:
        raise FileNotFoundError("Invalid or missing DOCX input(s): " + ", ".join(map(str, invalid)))
    yield sorted(paths, key=lambda item: item.name.lower())


def deliver_file(path: str | Path) -> None:
    """Report the resolved path of a written local artifact."""
    print(f"[LCD][results] Saved artifact: {Path(path).resolve()}", flush=True)
