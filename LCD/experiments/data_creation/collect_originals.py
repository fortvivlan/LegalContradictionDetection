"""Collect original first-instance administrative-offence rulings.

The collector searches public Sudact result pages, validates the complete body
of every candidate, and stores a DOCX text copy together with the source HTML,
plain text, and CSV provenance. Importing this module has no side effects.

Run from the repository root, for example::

    python -m LCD.experiments.data_creation.collect_originals \
        --group all --count-main 1000 --count-domain-shift 30
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
from html.parser import HTMLParser
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


SUDAct_ROOT = "https://sudact.ru"
MAIN_ARTICLES = (
    "18.8", "18.9", "18.10", "18.11", "18.12", "18.15",
    "18.16", "18.17", "18.18", "18.19", "18.20", "19.27",
)
DOMAIN_SHIFT_ARTICLES = ("20.20", "20.21", "6.9")
PUBLIC_SEEDS = {
    "18.9": ("A3arD7Jy4tN8",),
    "18.11": (
        "VMFE0DHF5fHW", "9Q3qNZho3ZJt", "Dud3xNbTpa6b",
        "62C0gUOVxIoa", "brG5GQz2VZUb", "l8iTuScuMENK",
    ),
    "18.16": ("5IDtnqH0F51n",),
    "18.20": ("g5EN5TmlZjWT", "OhfetYd33KWW"),
    "19.27": (
        "StIhbhDKhgNV", "pdj2o0auwvbK", "zetf5qgVIexb", "4Bw9By98bRxu",
    ),
    "20.20": (
        "BLi3lP1sKV5i", "OpA4tgTr205L", "wrH1ml5dykoW", "Gd8QiwLknp1w",
        "48xndxp9KMHH", "EU4qwTjJVwmw", "oeo3mqDG7OVx", "v9nxuXForYbR",
        "bxAHA9dDDVgC", "nVsDt5eWNYdA", "f3xSJwicoS1R", "aZktvgrIU5Wn",
        "UR8k8EJ4WyWw", "chCK7oQeaBdn",
    ),
}
DEFAULT_YEARS = tuple(range(datetime.now().year, 2015, -1))
USER_AGENT = "LegalContradictionDetection/1.0 (public judicial research)"

PROVENANCE_FIELDS = (
    "filename", "source_html", "source_text", "source_url",
    "retrieved_at_utc", "court", "case_number", "article",
    "content_sha256", "source_sha256", "evidence", "validation", "group",
    "source", "title", "document_id", "search_year", "search_query",
)
FAILURE_FIELDS = (
    "source_url", "attempted_at_utc", "article", "group", "reason",
    "title", "document_id", "search_year", "search_query",
)

_RESULT_LINK = re.compile(
    r'<h4[^>]*>.*?<a\s+href="/(regular|magistrate)/doc/([A-Za-z0-9]+)/'
    r'(?:\?[^"#]*)?(?:#[^"]*)?"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_OPERATIVE = re.compile(
    r"п\s*о\s*с\s*т\s*а\s*н\s*о\s*в\s*и\s*л(?:а)?\s*:?",
    re.IGNORECASE,
)
_PENALTY = re.compile(
    r"назначить.{0,240}(?:административн\w*\s+)?наказан|"
    r"подвергнуть.{0,240}(?:административн\w*\s+)?наказан|"
    r"административн\w*\s+(?:штраф|арест|выдворен|приостановлен)|"
    r"(?:штраф|арест|выдворен|предупрежден|обязательн\w*\s+работ|"
    r"дисквалификац|конфискац)",
    re.IGNORECASE | re.DOTALL,
)
_ADJUDICATION = re.compile(
    r"признать.{0,500}?виновн|привлечь.{0,500}?к\s+административн\w*\s+"
    r"ответственност",
    re.IGNORECASE | re.DOTALL,
)
_REVIEW_OPENING = re.compile(
    r"рассмотрев.{0,1000}?(?:жалоб[уы]|протест).{0,600}?"
    r"(?:на\s+постановлен|постановлени[ея].{0,100}?по\s+делу)",
    re.IGNORECASE | re.DOTALL,
)
_REVIEW_OUTCOME = re.compile(
    r"(?:постановлени[ея]|решени[ея]).{0,500}?"
    r"(?:оставить\s+без\s+изменения|отменить|изменить)|"
    r"жалоб[уы].{0,300}?оставить\s+без\s+удовлетворения",
    re.IGNORECASE | re.DOTALL,
)


class DecisionText(HTMLParser):
    """Extract readable text while preserving judicial paragraph breaks."""

    BREAK_TAGS = {"br", "p", "center", "div", "h1", "h2", "h3", "li"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag in {"script", "style"}:
            self._ignored_depth += 1
        elif not self._ignored_depth and tag in self.BREAK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif not self._ignored_depth and tag in self.BREAK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)

    def paragraphs(self) -> list[str]:
        """Return nonempty normalized lines from the parsed fragment."""
        joined = "".join(self.parts).replace("\xa0", " ")
        return [
            re.sub(r"[ \t]+", " ", line).strip()
            for line in joined.splitlines()
            if line.strip()
        ]


@dataclass(frozen=True)
class Candidate:
    """A document identifier discovered through one article search."""

    document_id: str
    title: str
    article: str
    group: str
    year: int
    query: str
    corpus: str = "regular"

    @property
    def url(self) -> str:
        """Return the canonical public document URL."""
        return f"{SUDAct_ROOT}/{self.corpus}/doc/{self.document_id}/"


@dataclass(frozen=True)
class ParsedDecision:
    """Structured fields and full visible text extracted from source HTML."""

    title: str
    court: str
    paragraphs: tuple[str, ...]

    @property
    def text(self) -> str:
        """Return the body as paragraph-delimited plain text."""
        return "\n".join(self.paragraphs)


@dataclass(frozen=True)
class Validation:
    """Outcome and auditable evidence from body-level validation."""

    accepted: bool
    reason: str
    evidence: str = ""


class PublicFetcher:
    """Rate-limited HTTP client with retries for public archive pages."""

    def __init__(
        self,
        delay: float = 1.0,
        timeout: float = 30.0,
        retries: int = 5,
        opener: Callable[..., object] = urlopen,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.delay = max(0.0, delay)
        self.timeout = timeout
        self.retries = max(1, retries)
        self.opener = opener
        self.sleep = sleep
        self._last_request = 0.0

    def fetch_text(self, url: str) -> str:
        """Fetch UTF-8 text, retrying transient failures."""
        error: Exception | None = None
        for attempt in range(self.retries):
            elapsed = time.monotonic() - self._last_request
            if elapsed < self.delay:
                self.sleep(self.delay - elapsed)
            request = Request(url, headers={
                "User-Agent": USER_AGENT,
                "X-Requested-With": "XMLHttpRequest",
            })
            try:
                response = self.opener(request, timeout=self.timeout)
                self._last_request = time.monotonic()
                with response:  # type: ignore[attr-defined]
                    payload = response.read()  # type: ignore[attr-defined]
                return payload.decode("utf-8")
            except Exception as exc:  # urllib exposes several exception types
                error = exc
                self._last_request = time.monotonic()
                if attempt + 1 < self.retries:
                    self.sleep(min(2 ** attempt, 8))
        assert error is not None
        raise error

    def fetch_json(self, url: str) -> dict[str, object]:
        """Fetch and decode one JSON response."""
        return json.loads(self.fetch_text(url))


def _strip_tags(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", value))).strip()


def parse_search_results(
    content: str,
    article: str,
    group: str,
    year: int,
    query: str | None = None,
) -> list[Candidate]:
    """Parse original-ruling candidates from one Sudact result fragment."""
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for match in _RESULT_LINK.finditer(content):
        corpus, identifier, raw_title = match.groups()
        title = _strip_tags(raw_title)
        if identifier in seen or not title.casefold().startswith(("постановление", "решение")):
            continue
        # A stated non-5 case index is a cheap rejection. Some Sudact magistrate
        # headings omit the case number or call an original ruling "Решение".
        stated_case = re.search(r"(?:по\s+делу\s+)?№\s*([^\s]+)", title, re.I)
        if stated_case and not re.match(r"5(?:[-–/]|[а-я]-)", stated_case.group(1), re.I):
            continue
        seen.add(identifier)
        candidates.append(Candidate(
            identifier, title, article, group, year, query or article, corpus,
        ))
    return candidates


def search_candidates(
    fetcher: PublicFetcher,
    article: str,
    group: str,
    year: int,
    page: int,
    query: str | None = None,
    corpus: str = "regular",
    poll_attempts: int = 12,
) -> tuple[list[Candidate], int]:
    """Search one result page, polling Sudact's asynchronous search job."""
    search_query = query or article
    prefix = corpus
    params = {
        f"{prefix}-txt": search_query,
        f"{prefix}-date_from": f"01.01.{year}",
        f"{prefix}-date_to": f"31.12.{year}",
        "page": str(page),
    }
    if corpus == "regular":
        params["regular-workflow_stage"] = "10"
    url = f"{SUDAct_ROOT}/{corpus}/doc_ajax/?{urlencode(params)}"
    for _ in range(max(1, poll_attempts)):
        payload = fetcher.fetch_json(url)
        content = payload.get("content")
        if isinstance(content, str):
            raw_count = len(re.findall(
                rf'href="/{corpus}/doc/[A-Za-z0-9]+/', content, re.I,
            ))
            return parse_search_results(
                content, article, group, year, search_query,
            ), raw_count
        if payload.get("search_status") == "finished" or payload.get("status") == "finished":
            return [], 0
        fetcher.sleep(max(0.25, fetcher.delay))
    raise TimeoutError(
        f"Sudact search did not finish: article={article} year={year} "
        f"page={page} query={search_query!r}"
    )


def candidate_pages(
    fetcher: PublicFetcher,
    article: str,
    group: str,
    years: Sequence[int],
    pages_per_year: int,
    corpora: Sequence[str] = ("regular",),
) -> Iterator[list[Candidate]]:
    """Yield result pages for one article, newest year first."""
    queries = (f"{article} признать виновным", article)
    for corpus in corpora:
        for query in queries:
            closed_years: set[int] = set()
            for page in range(1, pages_per_year + 1):
                for year in years:
                    if year in closed_years:
                        continue
                    candidates, raw_count = search_candidates(
                        fetcher, article, group, year, page,
                        query=query, corpus=corpus,
                    )
                    if not raw_count:
                        closed_years.add(year)
                        continue
                    yield candidates
                    if raw_count < 10:
                        closed_years.add(year)


def parse_decision(page: str) -> ParsedDecision | None:
    """Extract title, court, and complete published body from a Sudact page."""
    heading = re.search(r"<h1[^>]*>(.*?)</h1>", page, re.I | re.S)
    court = re.search(r'<div\s+class="b-justice"[^>]*>(.*?)</div>', page, re.I | re.S)
    marker = re.search(r'<hr\s+class="hr-h1"[^>]*>', page, re.I)
    if not heading or not marker:
        return None
    title = _strip_tags(heading.group(1))
    court_name = _strip_tags(court.group(1)) if court else ""
    if not title.casefold().startswith(("постановление", "решение")):
        return None
    fragment = page[marker.end():]
    fragment = fragment.split("<!--AdFox START-->", 1)[0]
    # Published redaction tokens such as <адрес> are text, not markup.
    fragment = re.sub(r"<([А-Яа-яЁё][^<>]{0,80})>", r"&lt;\1&gt;", fragment)
    parser = DecisionText()
    parser.feed(fragment)
    paragraphs = tuple(parser.paragraphs())
    if not paragraphs:
        return None
    return ParsedDecision(title, court_name, paragraphs)


def article_pattern(article: str) -> re.Pattern[str]:
    """Build an exact Russian statutory citation pattern for an article."""
    whole, fraction = article.split(".", 1)
    number = rf"{re.escape(whole)}[.,]{re.escape(fraction)}(?!\d)"
    return re.compile(
        rf"(?:ст(?:\.|атья|атьи|атью|атье|атьей|атьёй)\s*{number})"
        rf"(?:\s*(?:КоАП\s*(?:РФ|России)?|Кодекс[а-я\s]*"
        rf"об\s+административных\s+правонарушениях))?",
        re.IGNORECASE,
    )


def _citation_is_koap(text: str, match: re.Match[str]) -> bool:
    context = text[max(0, match.start() - 300):match.end() + 400]
    return bool(re.search(
        r"КоАП\s*(?:РФ|России)?|Кодекс[а-я\s]{0,100}"
        r"об\s+административных\s+правонарушениях",
        context,
        re.IGNORECASE,
    ))


def validate_original(parsed: ParsedDecision, article: str) -> Validation:
    """Validate a complete original ruling using body and operative evidence."""
    text = re.sub(r"[ \t]+", " ", parsed.text)
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) < 1200:
        return Validation(False, "incomplete_or_too_short")
    if not re.search(
        r"(?:дел[оауе]|материал\w*)\s+об\s+административн\w*\s+правонарушен",
        compact[:3500], re.I,
    ):
        return Validation(False, "not_administrative_offence_proceeding")
    if _REVIEW_OPENING.search(compact[:5000]):
        return Validation(False, "complaint_or_protest_review_opening")

    operative_markers = list(_OPERATIVE.finditer(compact))
    if not operative_markers:
        return Validation(False, "missing_final_operative_marker")
    operative = compact[operative_markers[-1].end():]
    if len(operative) < 180:
        return Validation(False, "incomplete_operative_section")
    if _REVIEW_OUTCOME.search(operative[:1800]):
        return Validation(False, "review_outcome")

    citation = article_pattern(article)
    body_matches = [match for match in citation.finditer(compact[:operative_markers[-1].start()])
                    if _citation_is_koap(compact, match)]
    operative_matches = [match for match in citation.finditer(operative[:2500])
                         if _citation_is_koap(operative, match)]
    if not body_matches:
        return Validation(False, "target_article_absent_from_body")
    if not operative_matches:
        return Validation(False, "target_article_absent_from_operative")
    adjudication = _ADJUDICATION.search(operative[:1800])
    if not adjudication:
        return Validation(False, "missing_adjudication")
    penalty = _PENALTY.search(operative[:2500])
    if not penalty:
        return Validation(False, "missing_penalty")

    appeal_notice = re.search(
        r"постановлени[ея].{0,180}(?:может\s+быть|вправе).{0,80}обжал",
        operative,
        re.I | re.S,
    )
    judge_signature = re.search(r"судья\s+[А-ЯЁA-Z]", operative[-700:], re.I)
    if not appeal_notice and not judge_signature:
        return Validation(False, "incomplete_after_operative")

    citation_text = operative_matches[0].group(0)
    case_in_title = re.search(
        r"(?:дел[оу]\s*)?№\s*5(?:[-–/]|[а-я]-)", parsed.title, re.I,
    )
    case_in_body = re.search(
        r"дел[оауе]\s*(?:№|N)\s*5(?:[-–/]|[а-я]-)", compact[:1200], re.I,
    )
    case_evidence = (
        "first_instance_case_5_verified_from_title" if case_in_title
        else "first_instance_case_5_verified_from_body" if case_in_body
        else "first_instance_verified_from_body_and_operative"
    )
    evidence = ";".join((
        case_evidence,
        "administrative_offence_opening",
        "final_operative_marker",
        f"operative_citation={citation_text}",
        f"adjudication={adjudication.group(0)[:120]}",
        f"penalty={penalty.group(0)[:120]}",
        "completion=" + ("appeal_right_notice" if appeal_notice else "judge_signature"),
    ))
    return Validation(True, "accepted", evidence)


def write_docx(path: Path, paragraphs: Iterable[str]) -> None:
    """Write a minimal standards-compliant DOCX containing source paragraphs."""
    body = "".join(
        '<w:p><w:r><w:t xml:space="preserve">'
        + escape(re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", paragraph))
        + "</w:t></w:r></w:p>"
        for paragraph in paragraphs
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}<w:sectPr/></w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    relationships = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", relationships)
        archive.writestr("word/document.xml", document)
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _append_csv(path: Path, row: dict[str, str], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _case_number(title: str) -> str:
    match = re.search(r"по\s+делу\s+№\s*([^:]+?)(?:\s|$)", title, re.I)
    if not match:
        match = re.search(r"№\s*([^\s]+)", title)
    return match.group(1).strip(".,") if match else ""


def _safe_stem(case_number: str, identifier: str) -> str:
    case = re.sub(r"[^0-9A-Za-zА-Яа-яЁё_-]+", "_", case_number).strip("_")
    return f"{case or 'case'}_{identifier}"


class Collection:
    """Resumable on-disk collection and provenance state."""

    def __init__(self, output: Path, retry_failures: bool = False) -> None:
        self.output = output
        self.provenance_path = output / "provenance.csv"
        self.failures_path = output / "failures.csv"
        rows = _read_csv(self.provenance_path)
        self.urls = {row["source_url"] for row in rows if row.get("source_url")}
        self.hashes = {row["content_sha256"] for row in rows if row.get("content_sha256")}
        self.counts: dict[tuple[str, str], int] = {}
        for row in rows:
            key = (row.get("group", ""), row.get("article", ""))
            self.counts[key] = self.counts.get(key, 0) + 1
        self.failed_keys: set[tuple[str, str]] = set()
        if not retry_failures:
            self.failed_keys = {
                (row["source_url"], row.get("article", ""))
                for row in _read_csv(self.failures_path)
                if row.get("source_url")
            }

    def count(self, group: str, article: str | None = None) -> int:
        """Return accepted count for a group or one group/article pair."""
        if article is not None:
            return self.counts.get((group, article), 0)
        return sum(value for (row_group, _), value in self.counts.items()
                   if row_group == group)

    def known(self, url: str, article: str) -> bool:
        """Return whether a URL was already accepted or rejected."""
        return url in self.urls or (url, article) in self.failed_keys

    def reject(self, candidate: Candidate, reason: str) -> None:
        """Persist a rejected candidate so resume runs can skip it."""
        key = (candidate.url, candidate.article)
        if key in self.failed_keys:
            return
        _append_csv(self.failures_path, {
            "source_url": candidate.url,
            "attempted_at_utc": datetime.now(timezone.utc).isoformat(),
            "article": candidate.article,
            "group": candidate.group,
            "reason": reason,
            "title": candidate.title,
            "document_id": candidate.document_id,
            "search_year": str(candidate.year),
            "search_query": candidate.query,
        }, FAILURE_FIELDS)
        self.failed_keys.add(key)

    def save(
        self,
        candidate: Candidate,
        page: str,
        parsed: ParsedDecision,
        validation: Validation,
    ) -> bool:
        """Save one distinct accepted ruling and append its provenance row."""
        plain_text = parsed.text
        content_hash = hashlib.sha256(plain_text.encode("utf-8")).hexdigest()
        if content_hash in self.hashes:
            self.reject(candidate, "duplicate_content_hash")
            return False
        case_number = _case_number(parsed.title)
        stem = _safe_stem(case_number, candidate.document_id)
        article_dir = self.output / candidate.group / candidate.article
        source_dir = article_dir / "source"
        docx_path = article_dir / f"{stem}.docx"
        html_path = source_dir / f"{stem}.html"
        text_path = source_dir / f"{stem}.txt"
        source_dir.mkdir(parents=True, exist_ok=True)
        html_path.write_text(page, encoding="utf-8")
        text_path.write_text(plain_text + "\n", encoding="utf-8")
        write_docx(docx_path, parsed.paragraphs)
        row = {
            "filename": str(docx_path.relative_to(self.output)),
            "source_html": str(html_path.relative_to(self.output)),
            "source_text": str(text_path.relative_to(self.output)),
            "source_url": candidate.url,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "court": parsed.court,
            "case_number": case_number,
            "article": candidate.article,
            "content_sha256": content_hash,
            "source_sha256": hashlib.sha256(page.encode("utf-8")).hexdigest(),
            "evidence": validation.evidence,
            "validation": validation.reason,
            "group": candidate.group,
            "source": f"sudact_{candidate.corpus}",
            "title": parsed.title,
            "document_id": candidate.document_id,
            "search_year": str(candidate.year),
            "search_query": candidate.query,
        }
        _append_csv(self.provenance_path, row, PROVENANCE_FIELDS)
        self.urls.add(candidate.url)
        self.hashes.add(content_hash)
        key = (candidate.group, candidate.article)
        self.counts[key] = self.counts.get(key, 0) + 1
        return True


def process_candidate(
    candidate: Candidate,
    fetcher: PublicFetcher,
    collection: Collection,
) -> bool:
    """Fetch, validate, and conditionally save one search candidate."""
    if collection.known(candidate.url, candidate.article):
        return False
    try:
        page = fetcher.fetch_text(candidate.url)
    except Exception as exc:
        collection.reject(candidate, f"fetch_error:{type(exc).__name__}:{exc}")
        return False
    parsed = parse_decision(page)
    if parsed is None:
        collection.reject(candidate, "unparseable_or_non_ruling_page")
        return False
    validation = validate_original(parsed, candidate.article)
    if not validation.accepted:
        collection.reject(candidate, validation.reason)
        return False
    return collection.save(candidate, page, parsed, validation)


def collect_group(
    group: str,
    articles: Sequence[str],
    target: int,
    minimum_per_article: int,
    minimum_page_budget: int,
    bulk_articles: Sequence[str],
    years: Sequence[int],
    pages_per_year: int,
    corpora: Sequence[str],
    fetcher: PublicFetcher,
    collection: Collection,
) -> None:
    """Collect one group, first satisfying per-article minima then total target."""
    streams = {
        article: candidate_pages(
            fetcher, article, group, years, pages_per_year, corpora,
        )
        for article in articles
    }
    exhausted: set[str] = set()
    pages_used = {article: 0 for article in articles}
    saved_from_pages = {article: 0 for article in articles}

    def consume_page(article: str, stop_at_target: bool = True) -> bool:
        try:
            page = next(streams[article])
        except StopIteration:
            exhausted.add(article)
            return False
        pages_used[article] += 1
        before = collection.count(group, article)
        for candidate in page:
            if process_candidate(candidate, fetcher, collection):
                print(
                    f"SAVED {group}/{article} article={collection.count(group, article)} "
                    f"group={collection.count(group)} {candidate.document_id}",
                    flush=True,
                )
            if stop_at_target and collection.count(group) >= target:
                break
        saved_from_pages[article] += collection.count(group, article) - before
        return True

    # Public identifiers from the preserved historical collector are hints only;
    # every page is fetched afresh and passes the same current validator.
    if "regular" in corpora:
        for article in articles:
            for identifier in PUBLIC_SEEDS.get(article, ()):
                if collection.count(group, article) >= minimum_per_article:
                    break
                process_candidate(
                    Candidate(
                        identifier, "public URL seed", article, group,
                        years[0], "public_seed", "regular",
                    ),
                    fetcher,
                    collection,
                )

    # Guarantee broad article coverage before filling the remaining total.
    for article in articles:
        while (collection.count(group, article) < minimum_per_article
               and article not in exhausted
               and pages_used[article] < minimum_page_budget
               ):
            consume_page(article, stop_at_target=False)

    preferred = [article for article in bulk_articles if article in streams]
    fallback = [article for article in articles if article not in preferred]
    pools = (preferred, fallback)
    for pool in pools:
        if not pool or collection.count(group) >= target:
            continue
        active_pool = set(pool)
        while collection.count(group) < target and active_pool:
            active_pool.difference_update(exhausted)
            if not active_pool:
                break
            ranked = sorted(
                active_pool,
                key=lambda value: (
                    saved_from_pages[value] / max(1, pages_used[value]),
                    collection.count(group, value),
                ),
                reverse=True,
            )
            progressed = False
            for article in ranked:
                if collection.count(group) >= target:
                    break
                progressed = consume_page(article) or progressed
            if not progressed:
                break

    while collection.count(group) < target and len(exhausted) < len(articles):
        progressed = False
        active = [article for article in articles if article not in exhausted]
        for article in active:
            if collection.count(group) >= target:
                break
            progressed = consume_page(article) or progressed
        if not progressed:
            break


def parse_articles(value: str) -> tuple[str, ...]:
    """Parse and validate a comma-separated list of decimal article numbers."""
    articles = tuple(item.strip().replace(",", ".") for item in value.split(",") if item.strip())
    if not articles or any(not re.fullmatch(r"\d+\.\d+", item) for item in articles):
        raise argparse.ArgumentTypeError("articles must be comma-separated decimal numbers")
    return articles


def parse_years(value: str) -> tuple[int, ...]:
    """Parse a comma-separated year list or inclusive descending range."""
    if ":" in value:
        newest, oldest = (int(item) for item in value.split(":", 1))
        step = -1 if newest >= oldest else 1
        return tuple(range(newest, oldest + step, step))
    years = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not years:
        raise argparse.ArgumentTypeError("at least one year is required")
    return years


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without executing collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("main", "domain_shift", "all"), default="all")
    parser.add_argument("--main-articles", type=parse_articles,
                        default=MAIN_ARTICLES, help="comma-separated main article numbers")
    parser.add_argument("--domain-shift-articles", type=parse_articles,
                        default=DOMAIN_SHIFT_ARTICLES,
                        help="comma-separated domain-shift article numbers")
    parser.add_argument("--count-main", type=int, default=1000)
    parser.add_argument("--count-domain-shift", type=int, default=30)
    parser.add_argument("--minimum-per-article", type=int, default=10)
    parser.add_argument("--minimum-page-budget", type=int, default=12,
                        help="result pages allowed per article during the minima pass")
    parser.add_argument("--main-bulk-articles", type=parse_articles,
                        default=("18.8", "18.10", "18.15"))
    parser.add_argument("--pages", "--pages-per-year", dest="pages_per_year",
                        type=int, default=30)
    parser.add_argument("--years", type=parse_years,
                        default=DEFAULT_YEARS,
                        help="comma list or NEWEST:OLDEST (default: current year through 2016)")
    parser.add_argument(
        "--source",
        choices=("sudact", "sudact_regular", "sudact_magistrate"),
        default="sudact",
        help="Sudact corpus: both, district/federal only, or magistrate only",
    )
    parser.add_argument("--delay", type=float, default=1.0,
                        help="minimum seconds between public HTTP requests")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=Path("local/data/classification/dataset0929/decisions"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the configurable, resumable original-rulings collector."""
    args = build_parser().parse_args(argv)
    if args.count_main < 0 or args.count_domain_shift < 0:
        raise SystemExit("counts must be non-negative")
    if (args.minimum_per_article < 0 or args.minimum_page_budget < 1
            or args.pages_per_year < 1):
        raise SystemExit("minimum must be non-negative and page budgets must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    fetcher = PublicFetcher(args.delay, args.timeout, args.retries)
    collection = Collection(args.output, retry_failures=args.retry_failures)
    corpora = {
        "sudact": ("regular", "magistrate"),
        "sudact_regular": ("regular",),
        "sudact_magistrate": ("magistrate",),
    }[args.source]
    requests = []
    if args.group in {"main", "all"}:
        requests.append((
            "main", args.main_articles, args.count_main, args.main_bulk_articles,
        ))
    if args.group in {"domain_shift", "all"}:
        requests.append((
            "domain_shift", args.domain_shift_articles,
            args.count_domain_shift, args.domain_shift_articles,
        ))
    for group, articles, target, bulk_articles in requests:
        print(
            f"START {group}: existing={collection.count(group)} target={target} "
            f"articles={','.join(articles)}",
            flush=True,
        )
        collect_group(
            group, articles, target, args.minimum_per_article,
            args.minimum_page_budget, bulk_articles,
            args.years, args.pages_per_year, corpora, fetcher, collection,
        )
        counts = ", ".join(
            f"{article}={collection.count(group, article)}" for article in articles
        )
        print(f"DONE {group}: total={collection.count(group)}; {counts}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
