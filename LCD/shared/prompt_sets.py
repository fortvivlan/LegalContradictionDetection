"""Load paired prompt constants without executing prompt modules."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptSet:
    """Resolved three-class prompt with its source hash."""

    name: str
    ternary: str
    ternary_path: Path
    ternary_sha256: str


def _literal_assignment(path: Path, variable: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == variable
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, str) and value.strip():
                return value
    raise ValueError(f"{path} must assign a nonempty string to {variable}")


def load_prompt_set(name: str = "base", *, root: str | Path = ".") -> PromptSet:
    """Load `base` or `<suffix>_prompt.py` without executing it."""
    if not name or any(character in name for character in ("/", "\\", ".")):
        raise ValueError("prompt set must be a simple filename suffix")
    directory = Path(root).resolve()
    if name == "base":
        ternary_path = directory / "LCD" / "shared" / "prompt.py"
    else:
        ternary_path = directory / f"{name}_prompt.py"
    for path in (ternary_path,):
        if not path.is_file():
            raise FileNotFoundError(f"Prompt file does not exist: {path}")
    ternary_bytes = ternary_path.read_bytes()
    return PromptSet(
        name=name,
        ternary=_literal_assignment(ternary_path, "PROMPT_TEXT"),
        ternary_path=ternary_path,
        ternary_sha256=hashlib.sha256(ternary_bytes).hexdigest(),
    )
