"""Audit saved dataset0929 candidates and their provenance without modifying them.

Run from the repository root::

    python -m LCD.experiments.data_creation.audit_collection
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Sequence
from zipfile import BadZipFile, ZipFile


DEFAULT_ROOT = Path("local/data/classification/dataset0929")
MAIN_ARTICLES = (
    "18.8", "18.9", "18.10", "18.11", "18.12", "18.15",
    "18.16", "18.17", "18.18", "18.19", "18.20", "19.27",
)
DOMAIN_SHIFT_ARTICLES = ("20.20", "20.21", "6.9")
GROUPS = {"main": MAIN_ARTICLES, "domain_shift": DOMAIN_SHIFT_ARTICLES}


def _checked_path(root: Path, value: str) -> Path:
    """Resolve a manifest path while rejecting paths outside its pool."""
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"path outside pool: {value}")
    return path


def audit_collection(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    """Check file integrity, hashes, uniqueness, and article-folder placement."""
    counts: Counter[str] = Counter()
    issues: list[str] = []
    seen_urls: dict[str, str] = {}
    seen_hashes: dict[str, str] = {}

    for pool in ("decisions", "appeals"):
        pool_root = root / pool
        manifest = pool_root / "provenance.csv"
        if not manifest.exists():
            issues.append(f"missing manifest: {manifest}")
            continue
        with manifest.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        for number, row in enumerate(rows, start=2):
            label = f"{pool}/provenance.csv:{number}"
            group, article = row.get("group", ""), row.get("article", "")
            if group not in GROUPS or article not in GROUPS.get(group, ()):
                issues.append(f"{label}: invalid group/article {group}/{article}")
            counts[f"{pool}/{group}/{article}"] += 1

            url = row.get("source_url", "")
            digest = row.get("content_sha256", "")
            if not url.startswith("https://"):
                issues.append(f"{label}: missing HTTPS source URL")
            elif url in seen_urls:
                issues.append(f"{label}: duplicate source URL; first at {seen_urls[url]}")
            else:
                seen_urls[url] = label
            if not digest:
                issues.append(f"{label}: missing content hash")
            elif digest in seen_hashes:
                issues.append(f"{label}: duplicate content hash; first at {seen_hashes[digest]}")
            else:
                seen_hashes[digest] = label

            expected_folder = (group, article)
            for field in ("filename", "source_html", "source_text"):
                value = row.get(field, "")
                if not value:
                    issues.append(f"{label}: missing {field}")
                    continue
                parts = Path(value).parts
                if len(parts) < 3 or parts[:2] != expected_folder:
                    issues.append(f"{label}: {field} outside {group}/{article}")
                try:
                    path = _checked_path(pool_root, value)
                except ValueError as exc:
                    issues.append(f"{label}: {exc}")
                    continue
                if not path.is_file():
                    issues.append(f"{label}: missing {field} file: {value}")
                    continue
                if field == "source_text":
                    actual = hashlib.sha256(
                        path.read_text(encoding="utf-8").rstrip("\n").encode("utf-8")
                    ).hexdigest()
                    if digest and actual != digest:
                        issues.append(f"{label}: source_text hash mismatch")
                elif field == "filename":
                    try:
                        with ZipFile(path) as archive:
                            if archive.testzip() or "word/document.xml" not in archive.namelist():
                                issues.append(f"{label}: invalid DOCX archive")
                    except BadZipFile:
                        issues.append(f"{label}: invalid DOCX archive")

    return {
        "root": str(root),
        "counts": dict(sorted(counts.items())),
        "decision_main_total": sum(
            counts[f"decisions/main/{article}"] for article in MAIN_ARTICLES
        ),
        "decision_domain_shift_total": sum(
            counts[f"decisions/domain_shift/{article}"] for article in DOMAIN_SHIFT_ARTICLES
        ),
        "appeal_total": sum(value for key, value in counts.items() if key.startswith("appeals/")),
        "issues": issues,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Print a JSON audit report; return nonzero only for integrity issues."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    report = audit_collection(args.root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
