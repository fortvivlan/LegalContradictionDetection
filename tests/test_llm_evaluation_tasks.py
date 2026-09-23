import pytest

from LCD.shared.llm_evaluation import _normalize_tasks


def test_normalize_tasks_preserves_selected_order() -> None:
    assert _normalize_tasks(("ternary",)) == ("ternary",)
    with pytest.raises(ValueError, match="ternary"):
        _normalize_tasks(("binary", "ternary"))


@pytest.mark.parametrize("tasks", [(), ("ternary", "ternary")])
def test_normalize_tasks_rejects_empty_or_duplicate_selection(tasks) -> None:
    with pytest.raises(ValueError, match="tasks cannot"):
        _normalize_tasks(tasks)
