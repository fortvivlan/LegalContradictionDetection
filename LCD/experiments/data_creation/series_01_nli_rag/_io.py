"""Cross-platform filesystem helpers with no import side effects."""
from __future__ import annotations

from pathlib import Path
import time


def replace_checkpoint(source: Path, target: Path, *, attempts: int = 8) -> None:
    """Retry transient Windows sharing locks while retaining atomic replacement."""
    for attempt in range(attempts):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(min(0.05 * 2 ** attempt, 0.8))
