from pathlib import Path

import pytest

from LCD.shared.prompt_sets import load_prompt_set


def test_base_prompt_is_read_without_execution(tmp_path: Path) -> None:
    prompt_dir = tmp_path / "LCD" / "shared"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "prompt.py").write_text(
        'PROMPT_TEXT = "three class"\nraise RuntimeError()', encoding="utf-8"
    )
    prompt = load_prompt_set("base", root=tmp_path)
    assert prompt.ternary == "three class"
    assert len(prompt.ternary_sha256) == 64


def test_named_prompt_and_path_validation(tmp_path: Path) -> None:
    (tmp_path / "legal_prompt.py").write_text('PROMPT_TEXT = "legal"', encoding="utf-8")
    assert load_prompt_set("legal", root=tmp_path).ternary == "legal"
    with pytest.raises(ValueError):
        load_prompt_set("../bad", root=tmp_path)
