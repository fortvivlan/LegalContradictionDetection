from pathlib import Path

import pytest

from LCD.shared import local_support


def test_local_documents_are_validated_sorted_and_preserved(tmp_path: Path) -> None:
    first, second = tmp_path / "b.docx", tmp_path / "a.DOCX"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    with local_support.selected_docx_files([first, second]) as paths:
        assert [path.name for path in paths] == ["a.DOCX", "b.docx"]
    assert first.read_bytes() == b"first"
    assert second.read_bytes() == b"second"
    with pytest.raises(FileNotFoundError):
        with local_support.selected_docx_files([tmp_path / "missing.docx"]):
            pass


def test_local_artifact_root_and_delivery(tmp_path: Path, capsys) -> None:
    root = local_support.prepare_artifact_root(tmp_path / "artifacts")
    artifact = root / "scores.xlsx"
    artifact.write_bytes(b"scores")
    local_support.deliver_file(artifact)
    assert root.is_dir()
    assert artifact.read_bytes() == b"scores"
    assert str(artifact.resolve()) in capsys.readouterr().out
