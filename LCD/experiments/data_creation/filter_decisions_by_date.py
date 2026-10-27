"""Plan and archive original rulings before a configurable date.

Use the published opening date before archive-title metadata: SudAct titles
can differ from the actual ruling by a day. Case numbers and search years are
never used as decision dates. Importing this module does not change files.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Sequence

from LCD.experiments.data_creation.review_decisions import apply_review
from LCD.shared.documents import read_docx_text

MONTHS = dict(zip(
    "января февраля марта апреля мая июня июля августа сентября октября ноября декабря".split(),
    range(1, 13),
))
DATE = re.compile(
    r"(?<!\d)(?P<day>\d{1,2})\s+(?P<month>" + "|".join(MONTHS) +
    r")\s+(?P<year>20\d{2})(?!\d)|"
    r"(?<!\d)(?P<nday>\d{1,2})[./](?P<nmonth>\d{1,2})[./](?P<nyear>20\d{2})(?!\d)",
    re.IGNORECASE,
)


def _dates(text: str) -> list[tuple[date, str]]:
    found = []
    for match in DATE.finditer(text):
        try:
            if match['day']:
                value = date(int(match['year']), MONTHS[match['month'].lower()], int(match['day']))
            else:
                value = date(int(match['nyear']), int(match['nmonth']), int(match['nday']))
        except ValueError:
            continue
        found.append((value, text[max(0, match.start()-80):match.end()+80]))
    return found


def decision_date(title: str, text: str) -> dict:
    """Return an opening date and evidence, falling back to a dated title.

    Only text before the judge/consideration opening is eligible. Ambiguous
    openings and disagreement about the year require review instead of guessing.
    An anonymized opening may use the archive's explicit title date.
    """
    opening = re.split(r"\bсудья\b|\bсудьи\b|\bрассмотрев\b|\bустановил\b", text[:1800],
                       maxsplit=1, flags=re.IGNORECASE)[0]
    body_dates = _dates(opening)
    title_part = re.split(r"\bот\s+", title, maxsplit=1, flags=re.IGNORECASE)
    title_dates = _dates(title_part[1]) if len(title_part) == 2 else []
    body_unique = {value for value, _ in body_dates}
    if len(body_unique) > 1:
        return {"date": None, "method": "ambiguous_opening", "evidence": body_dates_as_json(body_dates)}
    chosen = body_dates or title_dates
    if not chosen:
        return {"date": None, "method": "unresolved", "evidence": []}
    value = chosen[0][0]
    return {"date": value.isoformat(), "method": "published_opening" if body_dates else "archive_title",
            "evidence": body_dates_as_json(chosen),
            "archive_title_dates": body_dates_as_json(title_dates),
            "year_disagreement": bool(body_dates and title_dates and value.year != title_dates[0][0].year)}


def body_dates_as_json(values: list[tuple[date, str]]) -> list[dict]:
    """Serialize date passages without discarding the original wording."""
    return [{"date": value.isoformat(), "passage": passage} for value, passage in values]


def create_date_plan(root: Path, output: Path, cutoff: date) -> dict:
    """Write a date-only, hash-checked archive plan for main original rulings."""
    root, output = root.resolve(), output.resolve()
    if output.is_relative_to(root):
        raise ValueError("Output must be outside the active decisions tree")
    if output.exists():
        raise FileExistsError(output)
    manifest = root / 'provenance.csv'
    with manifest.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        fields, rows = list(reader.fieldnames or []), list(reader)
    names = {r['filename'] for r in rows if r['group'] == 'main'}
    actual = {p.relative_to(root).as_posix() for p in (root/'main').rglob('*.docx')}
    if names != actual:
        raise ValueError(f'Manifest/DOCX mismatch: {names ^ actual}')
    records = []
    for row in rows:
        if row['group'] != 'main':
            continue
        files = {}
        for key in ('filename', 'source_html', 'source_text'):
            path = (root/row[key]).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f'Unsafe source: {path}')
            files[key] = {'path': row[key], 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        original_doc = Path(row['source_text']).with_suffix('.doc')
        if (root/original_doc).is_file():
            files['published_original'] = {
                'path': original_doc.as_posix(),
                'sha256': hashlib.sha256((root/original_doc).read_bytes()).hexdigest(),
            }
        result = decision_date(row['title'], read_docx_text(root/row['filename']))
        pending = [] if result['date'] else ['unresolved_decision_date']
        if result.get('year_disagreement'):
            pending.append('date_year_disagreement')
        exclude = bool(result['date'] and date.fromisoformat(result['date']) < cutoff)
        records.append({'source': row['filename'], 'provenance': row, 'date_review': result,
                        'action': 'exclude' if exclude else 'keep',
                        'pending_review_flags': pending, 'files': files})
    plan = {'root': str(root), 'output': str(output), 'groups': ['main'],
            'cutoff': cutoff.isoformat(), 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
            'manifest_fields': fields, 'records': records}
    output.mkdir(parents=True)
    (output/'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {'reviewed': len(records), 'excluded': sum(r['action']=='exclude' for r in records),
               'retained': sum(r['action']=='keep' for r in records),
               'pending': [r['source'] for r in records if r['pending_review_flags']], 'cutoff': cutoff.isoformat()}
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Create a date review or apply its verified archive plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('local/data/classification/dataset0929/decisions'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cutoff', type=date.fromisoformat, default=date(2024, 1, 1))
    parser.add_argument('--apply', type=Path)
    args = parser.parse_args(argv)
    if args.apply:
        result = apply_review(args.apply)
    else:
        if not args.output:
            parser.error('--output is required when creating a plan')
        result = create_date_plan(args.root, args.output, args.cutoff)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
