"""Extract the smallest numbered legal units, keeping publisher material separate.

Only the explicitly supplied DOCX is read. Paragraph positions are one-based XML
positions, including tables and empty paragraphs. Raw text is retained for notes;
normalized copies are used only to recognize structure. No model is imported.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
NUMBER = r"\d+(?:[.-]\d+)*"
ARTICLE = re.compile(rf"^Статья\s+({NUMBER}(?:\s+\d+)*)\.\s*(.*)$", re.I)
PART = re.compile(rf"^({NUMBER})\.\s+", re.I)
POINT = re.compile(rf"^({NUMBER})\)\s+", re.I)
NOTE = re.compile(r"^Примечани(е|я)\s*[.:]\s*", re.I)
INACTIVE = re.compile(r"^(?:утратил[аои]?\s+силу|утратили\s+силу|не\s+применяется\b)", re.I)
EDITORIAL = {"Комментарий", "Информация о версии", "Информация об изменениях",
             "Подзаголовок для информации об изменениях"}
UNIT_FIELDS = ("text", "source", "hierarchy", "changes")
NOTE_FIELDS = ("text", "position")
PARSER_VERSION = 1


def clean(text: str) -> str:
    """Normalize a structural label without altering the original paragraph."""
    return " ".join(text.split())


def sha256(path: Path) -> str:
    """Hash a local artifact without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    """Write an inspectable UTF-8 JSON artifact."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class Paragraph:
    index: int
    text: str
    style: str = "Normal"


def read_paragraphs(source: str | Path) -> list[Paragraph]:
    """Read raw DOCX text and resolved style names in body/table XML order."""
    with ZipFile(source) as archive:
        styles = ET.fromstring(archive.read("word/styles.xml"))
        names = {node.get(W + "styleId"): node.find(W + "name").get(W + "val")
                 for node in styles.findall(W + "style") if node.find(W + "name") is not None}
        body = ET.fromstring(archive.read("word/document.xml")).find(W + "body")
    if body is None:
        raise ValueError("DOCX has no document body")
    result = []
    for index, node in enumerate(body.iter(W + "p"), 1):
        text = "".join((item.text or "") if item.tag == W + "t" else
                       "\t" if item.tag == W + "tab" else "\n"
                       for item in node.iter()
                       if item.tag in {W + "t", W + "tab", W + "br", W + "cr"})
        style = node.find(W + "pPr/" + W + "pStyle")
        result.append(Paragraph(index, text, names.get(style.get(W + "val"), "Normal")
                                if style is not None else "Normal"))
    return result


@dataclass
class Node:
    level: str
    number: str
    paragraphs: list[Paragraph] = field(default_factory=list)
    children: list["Node"] = field(default_factory=list)
    notices: list[dict] = field(default_factory=list)


def section_number(value: str) -> str:
    """Convert a Roman section number to the normalized Arabic location label."""
    if value.isdigit():
        return value
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    try:
        numbers = [values[c] for c in value.upper()]
    except KeyError as error:
        raise ValueError(f"Unknown section number: {value}") from error
    return str(sum(-n if i + 1 < len(numbers) and n < numbers[i + 1] else n
                   for i, n in enumerate(numbers)))


def _without_marker(paragraphs: list[Paragraph], level: str) -> str:
    texts = [p.text.strip() for p in paragraphs]
    if texts and level in {"part", "point"}:
        texts[0] = (PART if level == "part" else POINT).sub("", texts[0], count=1)
    return "\n".join(texts)


def _inactive(node: Node) -> bool:
    return bool(INACTIVE.match(clean(_without_marker(node.paragraphs, node.level))))


def extract_paragraphs(paragraphs: list[Paragraph]) -> dict:
    """Parse headings, parts, points and notes, returning records and a full audit.

    General post-list text beginning «Кроме того» is shared by the parent list.
    Other post-point paragraphs remain with their point (including «данное
    обстоятельство» in 4.3). Unknown free-standing post-list prose fails closed.
    """
    articles = []
    article = part = point = None
    section = chapter = ""
    section_id = chapter_id = ""
    section_notices, chapter_notices = [], []
    pending = []
    active_notice = None
    note_node = None
    plural_notes = in_notes = footer = False
    audit = {p.index: {"paragraph": p.index, "style": p.style, "text": p.text,
                       "kind": "blank" if not p.text.strip() else "unassigned", "unit_ids": []}
             for p in paragraphs}

    def kind(p, value):
        audit[p.index]["kind"] = value

    def consume_notices(target):
        nonlocal pending
        target.extend(pending)
        pending = []

    for paragraph in paragraphs:
        text = clean(paragraph.text)
        if not text:
            continue
        if footer or text == "Президент":
            footer = True
            kind(paragraph, "signature")
            continue
        if text.lower().startswith("информация об изменениях"):
            active_notice = {"paragraph_ids": [paragraph.index], "texts": [paragraph.text]}
            pending.append(active_notice)
            kind(paragraph, "amendment")
            continue
        if paragraph.style in EDITORIAL:
            if paragraph.style in {"Информация о версии", "Информация об изменениях"} and active_notice:
                active_notice["texts"].append(paragraph.text)
                active_notice["paragraph_ids"].append(paragraph.index)
                kind(paragraph, "amendment")
            else:
                kind(paragraph, "commentary" if article or section else "front_matter")
            continue
        active_notice = None
        heading = re.match(r"^Раздел\s+([IVXLC]+|\d+)\.", text, re.I)
        if heading:
            section, section_id = text, section_number(heading[1])
            chapter = chapter_id = ""
            section_notices, chapter_notices = [], []
            consume_notices(section_notices)
            article = part = point = None
            kind(paragraph, "section_heading")
            continue
        heading = re.match(rf"^Глава\s+({NUMBER})\.", text, re.I)
        if heading:
            chapter, chapter_id = text, heading[1]
            chapter_notices = []
            consume_notices(chapter_notices)
            article = part = point = None
            kind(paragraph, "chapter_heading")
            continue
        heading = ARTICLE.match(text)
        if heading:
            article = {"number": re.sub(r"\s+", ".", heading[1]), "title": heading[2],
                       "heading_id": paragraph.index, "section": section, "chapter": chapter,
                       "section_number": section_id, "chapter_number": chapter_id,
                       "node": Node("article", ""), "notes": [],
                       "ancestor_notices": list(section_notices) + list(chapter_notices)}
            consume_notices(article["node"].notices)
            articles.append(article)
            part = point = note_node = None
            in_notes = plural_notes = False
            kind(paragraph, "article_heading")
            continue
        if article is None:
            kind(paragraph, "front_matter")
            continue
        marker = NOTE.match(text)
        if marker:
            in_notes = True
            plural_notes = marker[1].lower() == "я"
            note_node = None
            body = text[marker.end():]
            if body:
                note_node = Node("note", "")
                if plural_notes and (numbered := PART.match(body)):
                    note_node.number = numbered[1]
                    # Drop only the plural block heading, preserving the note itself.
                    raw_marker = NOTE.match(paragraph.text.lstrip())
                    raw = paragraph.text.lstrip()[raw_marker.end():]
                    note_node.paragraphs.append(Paragraph(paragraph.index, raw, paragraph.style))
                else:
                    note_node.paragraphs.append(paragraph)
                consume_notices(note_node.notices)
                article["notes"].append(note_node)
            else:
                # Notices before a plural heading are metadata of the whole block.
                article.setdefault("note_block_notices", []).extend(pending)
                pending = []
            kind(paragraph, "note_heading" if not body else "note")
            continue
        if in_notes:
            numbered = PART.match(text) if plural_notes else None
            if numbered:
                note_node = Node("note", numbered[1])
                article["notes"].append(note_node)
            if note_node is None:
                raise ValueError(f"Expected a numbered note at paragraph {paragraph.index}: {text}")
            note_node.paragraphs.append(paragraph)
            consume_notices(note_node.notices)
            kind(paragraph, "note")
            continue
        numbered_part, numbered_point = PART.match(text), POINT.match(text)
        if numbered_part:
            part = Node("part", numbered_part[1], [paragraph])
            article["node"].children.append(part)
            point = None
            consume_notices(part.notices)
        elif numbered_point:
            point = Node("point", numbered_point[1], [paragraph])
            (part or article["node"]).children.append(point)
            consume_notices(point.notices)
        else:
            target = point or part or article["node"]
            if point and text.startswith("Кроме того,"):
                target = part or article["node"]
                audit[paragraph.index]["context_assignment"] = "shared_list_closing_rule"
            elif point and not re.match(r"^[а-я]\)\s", text, re.I):
                if "данное обстоятельство" not in text.lower():
                    raise ValueError(f"Ambiguous post-point paragraph {paragraph.index}: {text}")
                audit[paragraph.index]["context_assignment"] = "point_specific_continuation"
            target.paragraphs.append(paragraph)
            consume_notices(target.notices)
        kind(paragraph, "legal_text")

    units, notes = [], []
    excluded_ids = set()

    def exclude(node):
        excluded_ids.update(p.index for p in node.paragraphs)
        for child in node.children:
            exclude(child)

    def notice_text(notices):
        return "\n\n".join(dict.fromkeys("\n".join(n["texts"]) for n in notices))

    def emit(node, context, path, inherited_notices, a):
        if _inactive(node):
            exclude(node)
            return
        own = list(node.paragraphs)
        ancestors = context + own
        notices = inherited_notices + node.notices
        path = {**path, **({node.level: node.number} if node.number else {})}
        if node.children:
            for child in node.children:
                emit(child, ancestors, path, notices, a)
            return
        if not own:
            raise ValueError(f"Empty active article: {a['number']} {a['title']}")
        # Shared parent closing rules must follow the leaf, not precede it.
        first_index = own[0].index
        before = [p for p in context if p.index < first_index]
        after = [p for p in context if p.index > first_index]
        ordered = before + own + after
        texts = []
        for p in ordered:
            t = p.text.strip()
            if p is own[0] and node.level in {"part", "point"}:
                t = (PART if node.level == "part" else POINT).sub("", t, count=1)
            elif p in context:
                t = PART.sub("", t, count=1)
            texts.append(t)
        source = " ".join(label for label in
                          (f"п. {path['point']}" if path.get("point") else "",
                           f"ч. {path['part']}" if path.get("part") else "", f"ст. {a['number']}") if label)
        identity = f"koap_rf:article:{a['number']}"
        for level in ("part", "point"):
            if path.get(level):
                identity += f":{level}:{path[level]}"
        record = {"text": "\n".join(texts), "source": source,
                  "hierarchy": "\n".join(x for x in (a["chapter"], a["section"]) if x),
                  "changes": notice_text(notices), "unit_id": identity,
                  "code_id": "koap_rf", "article": a["number"], "part": path.get("part", ""),
                  "point": path.get("point", ""), "article_title": a["title"],
                  "chapter": a["chapter_number"], "section": a["section_number"],
                  "paragraph_ids": [p.index for p in ordered],
                  "amendment_paragraph_ids": sorted({i for n in notices for i in n["paragraph_ids"]})}
        units.append(record)

    for a in articles:
        if INACTIVE.match(a["title"]):
            exclude(a["node"])
            for note in a["notes"]:
                exclude(note)
            audit[a["heading_id"]]["kind"] = "inactive_article_heading"
            continue
        emit(a["node"], [], {}, a["ancestor_notices"], a)
        for note in a["notes"]:
            raw = "\n".join(p.text for p in note.paragraphs)
            content = NOTE.sub("", clean(raw), count=1)
            content = PART.sub("", content, count=1)
            if INACTIVE.match(content):
                exclude(note)
                continue
            suffix = note.number or "unnumbered"
            notes.append({"text": raw,
                          "position": f"ст. {a['number']} глава {a['chapter_number']} раздел {a['section_number']}",
                          "unit_id": f"koap_rf:article:{a['number']}:note:{suffix}",
                          "paragraph_ids": [p.index for p in note.paragraphs],
                          "changes": notice_text(a.get("note_block_notices", []) + note.notices)})
    for record in units + notes:
        for index in record["paragraph_ids"]:
            audit[index]["unit_ids"].append(record["unit_id"])
    for index in excluded_ids:
        audit[index]["kind"] = "inactive_placeholder"
    uncovered = [i for i, p in audit.items()
                 if p["kind"] in {"legal_text", "note", "unassigned"} and not p["unit_ids"]]
    if uncovered:
        raise ValueError(f"Unaccounted legal paragraphs: {uncovered}")
    for records in (units, notes):
        ids = [r["unit_id"] for r in records]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate legal-unit identities")
    return {"units": units, "notes": notes, "audit": list(audit.values()),
            "article_count": len(articles), "pending_amendments": pending}


def extract_docx(source: str | Path) -> dict:
    """Extract one explicit КоАП DOCX, recording its edition label and SHA-256."""
    source = Path(source)
    result = extract_paragraphs(read_paragraphs(source))
    match = re.search(r"редакция\s+(\d{2}\.\d{2}\.\d{4})", source.name)
    result["provenance"] = {"source_filename": source.name, "source_sha256": sha256(source),
                            "edition": match[1] if match else "unspecified"}
    for record in result["units"] + result["notes"]:
        record.update(result["provenance"])
    return result


def export_corpus(source: str | Path, output_dir: str | Path) -> dict:
    """Extract and export CSVs, audit and metadata; never load another code file."""
    result = extract_docx(source)
    target = Path(output_dir)
    if (target / "faiss_index").exists():
        raise ValueError("Use a fresh output directory when re-extracting an indexed corpus")
    target.mkdir(parents=True, exist_ok=True)
    for filename, records, fields in (("koap_units.csv", result["units"], UNIT_FIELDS),
                                      ("koap_notes.csv", result["notes"], NOTE_FIELDS)):
        with (target / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
    # Corpus metadata remains available before any embedding model is loaded.
    with (target / "units.jsonl").open("w", encoding="utf-8") as stream:
        for record in result["units"]:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_json(target / "extraction_audit.json", {"paragraphs": result["audit"],
               "note_metadata": [{k: v for k, v in n.items() if k != "text"} for n in result["notes"]],
               "pending_amendments": result["pending_amendments"]})
    from collections import Counter
    manifest = {"schema_version": 1, "parser_version": PARSER_VERSION, **result["provenance"],
                "configuration": {"smallest_unit": "point_then_part_then_article",
                                  "inactive_placeholders": "excluded", "notes": "separate_not_indexed",
                                  "shared_list_context": "repeated_per_point"},
                "counts": {"articles_seen": result["article_count"], "units": len(result["units"]),
                           "notes": len(result["notes"]),
                           "paragraph_kinds": dict(Counter(p["kind"] for p in result["audit"]))},
                "files": {name: sha256(target / name) for name in
                          ("koap_units.csv", "koap_notes.csv", "units.jsonl", "extraction_audit.json")}}
    write_json(target / "manifest.json", manifest)
    return manifest
