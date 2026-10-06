"""Inventory supplied law editions and export federal-law mentions for review.

Imports are side-effect free. The audit reads complete decision DOCX files,
including table cells, once per file; source HTML/TXT mirrors are not scanned.
It does not build or change the retrieval database.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from difflib import SequenceMatcher
from pathlib import Path
import re

from LCD.shared.documents import read_docx_text, split_russian_sentences


MENTION = re.compile(
    r"федеральн[а-яё]*\s+закон[а-яё]*|(?<!\w)Ф\s*З(?!\w)", re.IGNORECASE
)
LAW_NUMBER = re.compile(
    r"(?<![\w.])(?:(?:№|[NН](?:o|о)?\.?)\s*(\d+)\s*[-‐‑‒–—−]?\s*"
    r"|(\d+)\s*[-‐‑‒–—−]\s*)Ф\s*З(?!\w)", re.IGNORECASE
)
EDITION = re.compile(r"редакция\s+(\d{2}\.\d{2}\.\d{4})", re.IGNORECASE)
MONTHS = "января февраля марта апреля мая июня июля августа сентября октября ноября декабря".split()
DATE = re.compile(
    r"(?<!\d)(\d{1,2})\.(\d{1,2})\.(\d{4})(?!\d)|"
    r"(?<!\d)(\d{1,2})\s+(" + "|".join(MONTHS) + r")\s+(\d{4})(?!\d)",
    re.IGNORECASE,
)
TITLE = re.compile(r'^\s*(?:(?:года|г)\.?\s*)?[«"]([^»"]+)[»"]')
LAW_WORD = re.compile(r"закон[а-яё]*|(?<!\w)ФЗ(?!\w)", re.IGNORECASE)
NAMED_LAW = re.compile(
    r'(?:федеральн[а-яё]*\s+закон[а-яё]*|(?<!\w)ФЗ(?!\w))'
    r'(?:(?!приказ|указ|постановлен)[^«"\n]){0,100}[«"](О[б]?\s+[^»"\n]+)[»"]', re.IGNORECASE
)


def _normalize_title(title: str) -> str:
    """Normalize typography and common abbreviations in quoted law titles."""
    title = title.casefold().replace("ё", "е")
    title = re.sub(r"\bрф\b", "российской федерации", title)
    return re.sub(r"\s+", " ", re.sub(r"[^а-я0-9 ]", " ", title)).strip()


def representative_mentions(mentions: list[dict]) -> list[dict]:
    """Choose one sentence per identifiable law, distinguishing citation dates.

    Prefer a dated citation to a bare number. Matching descriptive titles merge
    date variants for the same number; distinct laws retain separate dates.
    Undated numbered citations merge when they identify a single law. Titles without a
    number are matched to numbered titles by normalized spelling similarity;
    otherwise one representative per distinct normalized title is retained.
    Generic references without a number or title cannot identify a law and are
    omitted from this compact export, but remain in the full export.
    """
    citations = []
    dated = {}
    for record in mentions:
        sentence = record["sentence"]
        for match in LAW_NUMBER.finditer(sentence):
            number = str(int(match.group(1) or match.group(2)))
            # The nearest preceding law mention bounds the citation's date.
            prefix = sentence[:match.start()]
            markers = list(LAW_WORD.finditer(prefix))
            prefix = prefix[markers[-1].start():] if markers else ""
            dates = list(DATE.finditer(prefix))
            suffix = sentence[match.end():]
            if re.match(r"\s*от\s+", suffix, re.IGNORECASE):
                following = DATE.search(suffix[:70])
                if following:
                    dates = [following]
                    suffix = suffix[following.end():]
            date = ""
            if dates:
                value = dates[-1]
                day, month, year = value.group(1, 2, 3)
                if day is None:
                    day, month_name, year = value.group(4, 5, 6)
                    month = str(MONTHS.index(month_name.lower()) + 1)
                date = f"{int(day):02}.{int(month):02}.{year}"
            title = TITLE.match(suffix)
            if title is None:
                title = re.search(r'[«"]([^»"]+)[»"]', prefix)
            title = _normalize_title(title.group(1)) if title else ""
            citations.append((number, date, title, record))
            if date:
                dated.setdefault(number, set()).add(date)
    # Equal descriptive titles identify the same law despite citation-date typos.
    # Generic amendment titles cannot establish this equivalence.
    titles_by_key = {}
    for number, date, title, record in citations:
        if title and not title.startswith("о внесении"):
            titles_by_key.setdefault((number, date), set()).add(title)
    canonical = {}
    for key, titles in sorted(titles_by_key.items(), key=lambda item: not bool(item[0][1])):
        equivalent = next((
            other for other, known_titles in titles_by_key.items()
            if other in canonical and other[0] == key[0]
            and any(SequenceMatcher(None, a, b).ratio() >= 0.8 or a in b or b in a
                    for a in titles for b in known_titles)
        ), None)
        canonical[key] = canonical[equivalent] if equivalent else key
    selected = {}
    aliases = {}
    # Prefer the first fully dated occurrence, then retain otherwise unseen laws.
    for number, date, title, record in sorted(citations, key=lambda item: not bool(item[1])):
        possibilities = dated.get(number, set())
        named_keys = {canonical[key] for key in canonical if key[0] == number}
        if (number, date) in canonical:
            key = canonical[(number, date)]
        elif len(named_keys) == 1 and not title:
            key = next(iter(named_keys))
        else:
            key = None
        if not date and len(possibilities) > 1:
            # A bare number cannot distinguish several dated laws already kept.
            if key is None:
                continue
        date = date or (next(iter(possibilities)) if len(possibilities) == 1 else "")
        key = key or (number, date)
        if key[1]:
            # Federal-law numbers distinguish laws within the citation year.
            key = next((known for known in selected
                        if known[0] == key[0] and known[1][-4:] == key[1][-4:]), key)
        selected.setdefault(key, record)
        if title:
            aliases.setdefault(title, set()).add(key)
    for record in mentions:
        for match in NAMED_LAW.finditer(record["sentence"]):
            title = _normalize_title(match.group(1))
            if title in aliases:
                continue
            inverted = re.search(r"ФЗ\s*[-№]*\s*(\d+)", match.group(0), re.IGNORECASE)
            if inverted and any(key[0] == str(int(inverted.group(1))) for key in selected):
                # Informal references such as ФЗ-114 already have a numbered
                # representative; do not add a second row for their title.
                continue
            # Similarity handles abbreviated/inflected titles, but never merges
            # generic amendment titles, which can name many different laws.
            scores = [
                (SequenceMatcher(None, title, known).ratio(), known)
                for known in aliases
                if not title.startswith("о внесении") and not known.startswith("о внесении")
            ]
            if scores and (max(scores)[0] >= 0.8 or any(
                len(title) >= 20 and (title in known or known in title)
                for _, known in scores
            )):
                continue
            key = ("title", title)
            selected.setdefault(key, record)
            aliases.setdefault(title, set()).add(key)
    result = {}
    for (number, date), record in selected.items():
        identity = f"{number}-ФЗ" + (f" от {date}" if date else "") if number != "title" else date
        source_key = (record["source"], record["sentence_index"])
        result.setdefault(source_key, {**record, "law_labels": []})["law_labels"].append(identity)
    return list(result.values())


def federal_law_numbers(text: str) -> list[str]:
    """Return distinct explicit numbered ФЗ references, without resolving titles."""
    return sorted({str(int(m.group(1) or m.group(2))) for m in LAW_NUMBER.finditer(text)}, key=int)


def inventory_codes(codes_dir: Path) -> list[dict]:
    """Record edition labels from filenames and SHA-256 of each supplied file.

    Edition labels are expert-supplied metadata, not verified effective dates.
    Missing labels remain null rather than being inferred from modification time.
    """
    files = sorted(p for p in codes_dir.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"No supplied code files in {codes_dir}")
    records = []
    for path in files:
        edition = EDITION.search(path.stem)
        records.append({
            "source": path.relative_to(codes_dir).as_posix(),
            "edition_label": edition.group(1) if edition else None,
            "edition_source": "filename",
            "federal_law_numbers": federal_law_numbers(path.stem.replace("_", " ")),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    return records


def extract_mentions(text: str) -> list[dict]:
    """Extract every matching sentence from all paragraphs, preserving repeats.

    This is a corpus coverage audit, so no operative-section or irrelevant-text
    filtering is applied. Whitespace within sentences is normalized for review.
    """
    records = []
    sentence_index = 0
    for paragraph in text.splitlines():
        for sentence in split_russian_sentences(paragraph):
            sentence_index += 1
            if MENTION.search(sentence):
                records.append({
                    "sentence_index": sentence_index,
                    "sentence": re.sub(r"\s+", " ", sentence).strip(),
                    "federal_law_numbers": federal_law_numbers(sentence),
                })
    return records


def audit_codes(codes_dir: Path, decisions_dir: Path, output_dir: Path) -> dict:
    """Write the inventory, compact/full expert TXT exports, and coverage summary.

    Only decision DOCX files are read; appeal directories are excluded even if
    accidentally included below the configured input root. Fail on unreadable
    documents instead of silently producing an incomplete expert report.
    """
    codes = inventory_codes(codes_dir)
    supplied = {number for code in codes for number in code["federal_law_numbers"]}
    documents = sorted(
        path for path in decisions_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".docx"
        and not path.name.startswith("~$")
        and not any("appeal" in part.lower() for part in path.parts)
    )
    if not documents:
        raise ValueError(f"No decision DOCX files in {decisions_dir}")
    mentions = []
    document_hashes = []
    for path in documents:
        source = path.relative_to(decisions_dir).as_posix()
        document_hashes.append({
            "source": source,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
        for record in extract_mentions(read_docx_text(path)):
            mentions.append({"source": source, **record})
    counts = Counter(number for record in mentions for number in record["federal_law_numbers"])
    absent = sorted(set(counts) - supplied, key=int)
    representatives = representative_mentions(mentions)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "codes_dir": str(codes_dir.resolve()),
        "decisions_dir": str(decisions_dir.resolve()),
        "documents_scanned": len(documents),
        "documents_with_mentions": len({record["source"] for record in mentions}),
        "sentences_with_mentions": len(mentions),
        "representative_sentences": len(representatives),
        "representative_law_groups": sum(len(record["law_labels"]) for record in representatives),
        "supplied_federal_law_numbers": sorted(supplied, key=int),
        "mentioned_federal_law_sentence_counts": dict(sorted(counts.items(), key=lambda item: int(item[0]))),
        "candidate_absent_federal_law_numbers": absent,
        "sentences_without_explicit_law_number": sum(not record["federal_law_numbers"] for record in mentions),
        "comparison_note": "Number-only candidates; dates, titles, redactions, indirect references and required editions need expert review.",
        "decision_sources": document_hashes,
    }
    lines = [
        "Все предложения решений с упоминаниями федеральных законов / ФЗ",
        f"Источник решений: {decisions_dir.resolve()}",
        f"Проверено документов: {len(documents)}; предложений: {len(mentions)}",
        "Сканируется полный текст решений; апелляции и зеркала HTML/TXT исключены.",
        "Поставленные ФЗ: " + ", ".join(f"{number}-ФЗ" for number in sorted(supplied, key=int)),
        "Кандидаты на добавление (по номеру; проверить эксперту): " + ", ".join(f"{number}-ФЗ" for number in absent),
        "Предложения без номера также включены. Повторы в разных местах сохранены.",
        "",
    ]
    for index, record in enumerate(mentions, 1):
        lines.extend([
            f"[{index}] {record['source']} — предложение {record['sentence_index']}",
            record["sentence"], "",
        ])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "federal_law_sentences_full.txt").write_text("\n".join(lines), encoding="utf-8")
    compact = [
        "Федеральные законы: одно предложение на каждый выявленный закон",
        f"Источник решений: {decisions_dir.resolve()}",
        f"Проверено документов: {len(documents)}; найдено предложений: {len(mentions)}; оставлено: {len(representatives)}",
        "Разные законы с одинаковым номером сохранены отдельно; одинаковые названия объединяют варианты дат.",
        "Предпочтение отдано первому упоминанию с датой; названия без номера сопоставлены по сходству текста.",
        "Общие ссылки без номера и названия исключены; полный список сохранён в federal_law_sentences_full.txt.",
        "Группировка предварительная: ошибки и скрытые даты в исходных решениях требуют проверки экспертом.",
        "",
    ]
    for index, record in enumerate(representatives, 1):
        compact.extend([
            f"[{index}] " + "; ".join(record["law_labels"]),
            f"Источник: {record['source']} — предложение {record['sentence_index']}",
            record["sentence"], "",
        ])
    (output_dir / "federal_law_sentences.txt").write_text("\n".join(compact), encoding="utf-8")
    (output_dir / "code_versions.json").write_text(
        json.dumps({"codes": codes}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "coverage_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    """Run the local expert-review audit with configurable input/output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codes-dir", type=Path, default=Path("local/data/codes"))
    parser.add_argument("--decisions-dir", type=Path, default=Path("local/data/classification/dataset0929/decisions"))
    parser.add_argument("--output-dir", type=Path, default=Path("local/experiments/lawbook/series_03_code_update"))
    args = parser.parse_args()
    summary = audit_codes(args.codes_dir, args.decisions_dir, args.output_dir)
    print(json.dumps({key: value for key, value in summary.items() if key != "decision_sources"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
