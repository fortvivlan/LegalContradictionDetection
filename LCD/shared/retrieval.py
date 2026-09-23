"""Deterministic legal-citation lookup followed by bounded FAISS retrieval."""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .common import (
    DEFAULT_EMBEDDING_REVISION,
    DEFAULT_RAG_REPOSITORY,
    DEFAULT_RAG_REVISION,
)

if TYPE_CHECKING:
    from .rag.reranking import Reranker


@dataclass(frozen=True)
class Citation:
    """Structured Russian legal citation fields, kept as strings."""

    code: str | None = None
    article: str | None = None
    part: str | None = None
    point: str | None = None


@dataclass(frozen=True)
class RetrievalRecord:
    """One premise and its retrieval audit metadata."""

    premise: str
    source: str
    method: str
    rank: int
    score: float | None
    citation: Citation
    detected_citations: tuple[Citation, ...] = ()
    unresolved_citations: tuple[Citation, ...] = ()
    initial_rank: int | None = None
    reranker_score: float | None = None


@dataclass(frozen=True)
class RetrievalOutcome:
    """Candidate pool and final records for one retrieval request."""

    candidates: tuple[RetrievalRecord, ...]
    results: tuple[RetrievalRecord, ...]
    reranked: bool
    detected_citations: tuple[Citation, ...] = ()
    unresolved_citations: tuple[Citation, ...] = ()


_NUMBER = r"[0-9]+(?:\.[0-9]+)*"
_ARTICLE_GROUP_PATTERN = re.compile(
    rf"(?ix)\bст(?:атья|атьи|атью|атье|атьей|атьёй|атьями)?\.?(?:\s*ст\.?)?"
    rf"\s*(?P<numbers>{_NUMBER}(?:\s*(?:,|;|\bи\b|[-–—])\s*(?:ст\.?)?\s*{_NUMBER})*)"
)
_PART_TOKEN = (
    rf"(?:ч\.?|часть|части|частью|частей|частями)\s*(?P<number>{_NUMBER})"
)
_POINT_TOKEN = (
    rf"(?:п\.?|пункт|пункта|пунктом|пункте|пункты|пунктами)"
    rf"\s*(?P<number>{_NUMBER})"
)
_PART_PATTERN = re.compile(rf"(?i)\b{_PART_TOKEN}")
_POINT_PATTERN = re.compile(rf"(?i)\b{_POINT_TOKEN}")
_LEFT_SUBPROVISION_PATTERN = re.compile(
    rf"(?ix)(?:(?P<point>(?:п\.?|пункт|пункта|пунктом|пункте)\s*{_NUMBER})\s*)?"
    rf"(?:(?P<part>(?:ч\.?|часть|части|частью|частей|частями)\s*{_NUMBER})\s*)?"
    r"[,;:()\s]*$"
)
_RIGHT_SUBPROVISION_PATTERN = re.compile(
    rf"(?ix)^[()\s]*(?:(?P<part>(?:ч\.?|часть|части|частью|частей|частями)\s*{_NUMBER})\s*)?"
    rf"(?:(?P<point>(?:п\.?|пункт|пункта|пунктом|пункте)\s*{_NUMBER})\s*)?"
)
_SOURCE_ARTICLE_PATTERN = re.compile(rf"(?i)\bСтатья\s+({_NUMBER})")
_SOURCE_PART_PATTERN = re.compile(rf"(?i)\bч\.\s*({_NUMBER})")
_SOURCE_POINT_PATTERN = re.compile(rf"(?i)\bп\.\s*({_NUMBER})")
_NON_CODE_INSTRUMENT_PATTERN = re.compile(
    r"(?i)\b(?:Федеральн\w*\s+закон\w*|ФЗ(?:\s*[-№])?|закон\w*\s+РФ)"
)
_MAX_CODE_DISTANCE = 180

_CODE_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "КоАП РФ",
        r"(?:КоАП(?:\s*РФ)?|КРФ\s*о\s*АП|КРФ\s*об\s*АП|КРФоАП|"
        r"Кодекс(?:а|ом|у|е)?\s+(?:Российской\s+Федерации|РФ)\s+об\s+"
        r"административных\s+правонарушениях|Кодекс(?:а|ом|у|е)?\s+РФ\s+об\s+АП)"
    ),
    (
        "УПК РФ",
        r"(?:УПК(?:\s*РФ)?|Уголовно[-\s]процессуальн\w*\s+кодекс\w*"
        r"(?:\s+(?:Российской\s+Федерации|РФ))?)"
    ),
    (
        "УИК РФ",
        r"(?:УИК(?:\s*РФ)?|Уголовно[-\s]исполнительн\w*\s+кодекс\w*"
        r"(?:\s+(?:Российской\s+Федерации|РФ))?)"
    ),
    (
        "УК РФ",
        r"(?:УК(?:\s*РФ)?|Уголовн\w*\s+кодекс\w*"
        r"(?:\s+(?:Российской\s+Федерации|РФ))?)"
    ),
    (
        "ГПК РФ",
        r"(?:ГПК(?:\s*РФ)?|Гражданск\w*\s+процессуальн\w*\s+кодекс\w*"
        r"(?:\s+(?:Российской\s+Федерации|РФ))?)"
    ),
    (
        "ГК РФ",
        r"(?:ГК(?:\s*РФ)?|Гражданск\w*\s+кодекс\w*"
        r"(?:\s+(?:Российской\s+Федерации|РФ))?)"
    ),
    (
        "КАС РФ",
        r"(?:КАС(?:\s*РФ)?|Кодекс\w*\s+административного\s+судопроизводства"
        r"(?:\s+(?:Российской\s+Федерации|РФ))?)"
    ),
    (
        "ТК",
        r"(?:ТК(?:\s*РФ)?|Трудов\w*\s+кодекс\w*"
        r"(?:\s+(?:Российской\s+Федерации|РФ))?)"
    ),
)


@dataclass(frozen=True)
class _CodeMarker:
    code: str
    start: int
    end: int


def _canonical_code(code: str) -> str:
    normalized = re.sub(r"\s+", " ", code.strip())
    if re.fullmatch(r"(?i)ГК РФ ч\.\s*\d+", normalized):
        return "ГК РФ"
    return normalized


def parse_codex_source(source: str) -> Citation | None:
    """Parse one updated ``codex.csv`` source label into structured fields."""
    article_match = _SOURCE_ARTICLE_PATTERN.search(source)
    if article_match is None:
        return None
    prefix = source[: article_match.start()].strip().rstrip(":.;").strip()
    if not prefix:
        return None
    remainder = source[article_match.end() :]
    part_match = _SOURCE_PART_PATTERN.search(remainder)
    point_match = _SOURCE_POINT_PATTERN.search(remainder)
    return Citation(
        code=prefix,
        article=article_match.group(1),
        part=part_match.group(1) if part_match else None,
        point=point_match.group(1) if point_match else None,
    )


def _literal_alias_pattern(alias: str) -> str:
    pieces = [re.escape(piece) for piece in re.split(r"\s+", alias.strip()) if piece]
    return r"\s*".join(pieces)


def _find_code_markers(
    text: str,
    code_aliases: Mapping[str, Sequence[str]] | None,
) -> list[_CodeMarker]:
    patterns: list[tuple[str, str]] = list(_CODE_PATTERNS)
    if code_aliases:
        for code, aliases in code_aliases.items():
            for alias in aliases:
                patterns.append((code, _literal_alias_pattern(alias)))
    markers: list[_CodeMarker] = []
    seen: set[tuple[str, int, int]] = set()
    for code, pattern in patterns:
        for match in re.finditer(rf"(?<!\w)(?:{pattern})(?!\w)", text, re.I):
            item = (_canonical_code(code), match.start(), match.end())
            if item not in seen:
                seen.add(item)
                markers.append(_CodeMarker(*item))
    return sorted(
        markers, key=lambda marker: (marker.start, -(marker.end - marker.start))
    )


def _token_number(token: str | None, pattern: re.Pattern[str]) -> str | None:
    if not token:
        return None
    match = pattern.search(token)
    return match.group("number") if match else None


def _expand_article_numbers(value: str) -> list[str]:
    matches = list(re.finditer(_NUMBER, value))
    if not matches:
        return []
    expanded = [matches[0].group(0)]
    for previous, current in zip(matches, matches[1:]):
        separator = value[previous.end() : current.start()]
        current_number = current.group(0)
        if not re.search(r"[-–—]", separator):
            expanded.append(current_number)
            continue
        start_parts = previous.group(0).split(".")
        end_parts = current_number.split(".")
        if (
            len(start_parts) < 2
            or start_parts[:-1] != end_parts[:-1]
            or int(end_parts[-1]) < int(start_parts[-1])
        ):
            return []
        expanded.extend(
            ".".join([*start_parts[:-1], str(number)])
            for number in range(int(start_parts[-1]) + 1, int(end_parts[-1]) + 1)
        )
    return list(dict.fromkeys(expanded))


def _associated_code(
    group_index: int,
    groups: Sequence[re.Match[str]],
    markers: Sequence[_CodeMarker],
    blockers: Sequence[re.Match[str]],
) -> str | None:
    group = groups[group_index]
    previous_end = groups[group_index - 1].end() if group_index else 0
    next_start = (
        groups[group_index + 1].start()
        if group_index + 1 < len(groups)
        else None
    )
    candidates: list[tuple[int, str, int, int]] = []
    for marker in markers:
        if marker.end <= group.start() and marker.end >= previous_end:
            distance = group.start() - marker.end
            candidates.append((distance, marker.code, marker.end, group.start()))
        elif marker.start >= group.end() and (
            next_start is None or marker.start <= next_start
        ):
            distance = marker.start - group.end()
            candidates.append((distance, marker.code, group.end(), marker.start))
    for distance, code, boundary_start, boundary_end in sorted(candidates):
        if distance > _MAX_CODE_DISTANCE:
            continue
        if any(
            blocker.start() >= boundary_start and blocker.end() <= boundary_end
            for blocker in blockers
        ):
            continue
        return code
    return None


def extract_citations(
    text: str,
    *,
    code_aliases: Mapping[str, Sequence[str]] | None = None,
) -> list[Citation]:
    """Extract ordered Russian legal citations without numeric conversion."""
    groups = list(_ARTICLE_GROUP_PATTERN.finditer(text))
    if not groups:
        return []
    markers = _find_code_markers(text, code_aliases)
    blockers = list(_NON_CODE_INSTRUMENT_PATTERN.finditer(text))
    citations: list[Citation] = []
    for group_index, group in enumerate(groups):
        articles = _expand_article_numbers(group.group("numbers"))
        if not articles:
            continue
        left_start = groups[group_index - 1].end() if group_index else 0
        right_end = (
            groups[group_index + 1].start()
            if group_index + 1 < len(groups)
            else len(text)
        )
        left = text[max(left_start, group.start() - 100) : group.start()]
        right = text[group.end() : min(right_end, group.end() + 100)]
        left_match = _LEFT_SUBPROVISION_PATTERN.search(left)
        right_match = _RIGHT_SUBPROVISION_PATTERN.search(right)
        left_part = _token_number(
            left_match.group("part") if left_match else None, _PART_PATTERN
        )
        left_point = _token_number(
            left_match.group("point") if left_match else None, _POINT_PATTERN
        )
        right_part = _token_number(
            right_match.group("part") if right_match else None, _PART_PATTERN
        )
        right_point = _token_number(
            right_match.group("point") if right_match else None, _POINT_PATTERN
        )
        code = _associated_code(group_index, groups, markers, blockers)
        for article in articles:
            citations.append(
                Citation(
                    code=code,
                    article=article,
                    part=left_part or right_part,
                    point=left_point or right_point,
                )
            )
    return list(dict.fromkeys(citations))


def extract_citation(text: str) -> Citation:
    """Return the first extracted citation, or an empty citation."""
    citations = extract_citations(text)
    return citations[0] if citations else Citation()


def ensure_rag_repository(
    rag_dir: str | Path,
    repository_url: str = DEFAULT_RAG_REPOSITORY,
    revision: str = DEFAULT_RAG_REVISION,
) -> tuple[Path, str]:
    """Fetch/check out the requested RAG revision and return its exact commit."""
    path = Path(rag_dir)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                "main",
                repository_url,
                str(path),
            ],
            check=True,
        )
    if not (path / ".git").exists():
        raise ValueError(f"RAG directory is not a Git repository: {path}")
    dirty = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(f"Cannot update RAG repository: {path} has local changes")
    if revision == "main":
        subprocess.run(
            ["git", "-C", str(path), "fetch", "--depth", "1", "origin", "main"],
            check=True,
        )
        target = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "FETCH_HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    else:
        exists = subprocess.run(
            ["git", "-C", str(path), "cat-file", "-e", f"{revision}^{{commit}}"],
            capture_output=True,
        ).returncode == 0
        if not exists:
            subprocess.run(
                ["git", "-C", str(path), "fetch", "--depth", "1", "origin", revision],
                check=True,
            )
        target = revision
    subprocess.run(
        ["git", "-C", str(path), "checkout", "--detach", target],
        check=True,
    )
    codex_path = path / "codex.csv"
    index_path = path / "faiss_index"
    if not codex_path.is_file() or not (index_path / "index.faiss").is_file() or not (
        index_path / "index.pkl"
    ).is_file():
        raise FileNotFoundError(
            f"RAG repository at {path} lacks codex.csv or faiss_index artifacts"
        )
    commit = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected_commit = subprocess.run(
        ["git", "-C", str(path), "rev-parse", f"{target}^{{commit}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != expected_commit:
        raise RuntimeError(
            f"RAG revision mismatch: expected {expected_commit}, checked out {commit}"
        )
    return path, commit


class PremiseRetriever:
    """Retrieve exact codex provisions before falling back to FAISS."""

    def __init__(
        self,
        codex_dataframe,
        vectorstore: Any,
        *,
        reranker: "Reranker | None" = None,
    ):
        required = {"text", "source"}
        if not required.issubset(codex_dataframe.columns):
            raise ValueError("codex.csv must contain text and source columns")
        self._codex = (
            codex_dataframe.loc[:, ["text", "source"]].drop_duplicates().copy()
        )
        parsed = self._codex["source"].map(
            lambda source: parse_codex_source(str(source))
        )
        self._codex["_code"] = parsed.map(
            lambda citation: citation.code if citation is not None else None
        )
        self._codex["_article"] = parsed.map(
            lambda citation: citation.article if citation is not None else None
        )
        self._codex["_part"] = parsed.map(
            lambda citation: citation.part if citation is not None else None
        )
        self._codex["_point"] = parsed.map(
            lambda citation: citation.point if citation is not None else None
        )
        aliases: dict[str, set[str]] = {}
        for source_code in self._codex["_code"].dropna().astype(str).unique():
            canonical = _canonical_code(source_code)
            aliases.setdefault(canonical, set()).update((canonical, source_code))
        self._code_aliases = {
            code: tuple(sorted(values, key=len, reverse=True))
            for code, values in aliases.items()
        }
        self._vectorstore = vectorstore
        self._reranker = reranker

    @classmethod
    def from_rag_directory(
        cls,
        rag_dir: str | Path,
        *,
        embedding_model: str = "ai-forever/sbert_large_nlu_ru",
        embedding_revision: str = DEFAULT_EMBEDDING_REVISION,
        embedding_device: str = "cpu",
    ) -> "PremiseRetriever":
        """Load trusted cloned codex and FAISS artifacts."""
        return cls.from_components(
            Path(rag_dir) / "codex.csv",
            Path(rag_dir) / "faiss_index",
            embedding_model=embedding_model,
            embedding_revision=embedding_revision,
            embedding_device=embedding_device,
        )

    @classmethod
    def from_components(
        cls,
        codex_path: str | Path,
        index_dir: str | Path,
        *,
        embedding_model: str = "ai-forever/sbert_large_nlu_ru",
        embedding_revision: str | None = DEFAULT_EMBEDDING_REVISION,
        embedding_device: str = "cpu",
        normalize_embeddings: bool = False,
        embedding_query_prefix: str = "",
        embedding_document_prefix: str = "",
        embedding_trust_remote_code: bool = False,
        embedding_precision: str = "float32",
        embedding_batch_size: int = 32,
        reranker: "Reranker | None" = None,
    ) -> "PremiseRetriever":
        """Load trusted corpus/index paths with their matching encoder."""
        import pandas as pd
        from langchain_community.vectorstores import FAISS
        from .rag.embeddings import SentenceTransformerEmbeddings

        dataframe = pd.read_csv(Path(codex_path).resolve())
        embeddings = SentenceTransformerEmbeddings(
            embedding_model,
            revision=embedding_revision,
            device=embedding_device,
            trust_remote_code=embedding_trust_remote_code,
            precision=embedding_precision,
            batch_size=embedding_batch_size,
            normalize_embeddings=normalize_embeddings,
            query_prefix=embedding_query_prefix,
            document_prefix=embedding_document_prefix,
        )
        vectorstore = FAISS.load_local(
            Path(index_dir).resolve(),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return cls(dataframe, vectorstore, reranker=reranker)

    @staticmethod
    def _code_matches(source_code: object, code: str) -> bool:
        if source_code is None:
            return False
        return _canonical_code(str(source_code)).casefold() == _canonical_code(
            code
        ).casefold()

    def exact_lookup(self, citation: Citation) -> list[RetrievalRecord]:
        """Return unambiguous exact provisions or an empty list."""
        if not citation.code or not citation.article:
            return []
        candidates = self._codex[
            self._codex["_code"].map(
                lambda source_code: self._code_matches(source_code, citation.code)
            )
            & (self._codex["_article"] == citation.article)
        ]
        if citation.part is not None:
            candidates = candidates[candidates["_part"] == citation.part]
        if citation.point is not None:
            candidates = candidates[candidates["_point"] == citation.point]
        if (
            citation.point is not None
            and citation.part is None
            and len(candidates) != 1
        ):
            return []
        if citation.part is None and citation.point is None and len(candidates) != 1:
            return []
        if candidates.empty:
            return []
        unique = candidates.drop_duplicates(subset=["source", "text"])
        return [
            RetrievalRecord(
                premise=str(row.text),
                source=str(row.source),
                method="exact",
                rank=index,
                score=None,
                citation=citation,
            )
            for index, row in enumerate(unique.itertuples(index=False), start=1)
        ]

    @staticmethod
    def _retrieval_depths(
        top_k: int, final_top_k: int | None
    ) -> tuple[int, int]:
        candidate_top_k = int(top_k)
        retained_top_k = (
            candidate_top_k if final_top_k is None else int(final_top_k)
        )
        if candidate_top_k <= 0 or retained_top_k <= 0:
            raise ValueError("top_k and final_top_k must be positive")
        if retained_top_k > candidate_top_k:
            raise ValueError("final_top_k cannot exceed top_k")
        return candidate_top_k, retained_top_k

    def _semantic_outcome(
        self,
        hypothesis: str,
        *,
        candidate_top_k: int,
        retained_top_k: int,
        detected_citations: tuple[Citation, ...] = (),
    ) -> RetrievalOutcome:
        matches = self._vectorstore.similarity_search_with_score(
            hypothesis, k=candidate_top_k
        )
        unique_matches = []
        seen_documents: set[tuple[str, str]] = set()
        for document, score in matches:
            key = (
                str(document.page_content),
                str(document.metadata.get("source", "")),
            )
            if key in seen_documents:
                continue
            seen_documents.add(key)
            unique_matches.append((document, score))
        fallback_citation = (
            detected_citations[0]
            if len(detected_citations) == 1
            else Citation()
        )
        candidates = tuple(
            RetrievalRecord(
                premise=str(document.page_content),
                source=str(document.metadata.get("source", "")),
                method="faiss",
                rank=index,
                score=float(score),
                citation=fallback_citation,
                detected_citations=detected_citations,
                unresolved_citations=detected_citations,
                initial_rank=index,
            )
            for index, (document, score) in enumerate(unique_matches, start=1)
        )
        if self._reranker is None:
            return RetrievalOutcome(
                candidates,
                candidates[:retained_top_k],
                False,
                detected_citations,
                detected_citations,
            )
        from .rag.reranking import rerank_records

        results = rerank_records(
            hypothesis,
            candidates,
            self._reranker,
            final_top_k=retained_top_k,
        )
        return RetrievalOutcome(
            candidates,
            results,
            True,
            detected_citations,
            detected_citations,
        )

    def retrieve_semantic_with_details(
        self,
        hypothesis: str,
        *,
        top_k: int = 20,
        final_top_k: int | None = None,
    ) -> RetrievalOutcome:
        """Retrieve through FAISS only, deliberately bypassing citation rules."""
        candidate_top_k, retained_top_k = self._retrieval_depths(
            top_k, final_top_k
        )
        return self._semantic_outcome(
            hypothesis,
            candidate_top_k=candidate_top_k,
            retained_top_k=retained_top_k,
        )

    def _rule_outcome(
        self, hypothesis: str
    ) -> tuple[RetrievalOutcome, tuple[Citation, ...]]:
        citations = extract_citations(hypothesis, code_aliases=self._code_aliases)
        exact: list[RetrievalRecord] = []
        unresolved: list[Citation] = []
        seen: set[tuple[str, str]] = set()
        for citation in citations:
            matches = self.exact_lookup(citation)
            if not matches:
                unresolved.append(citation)
                continue
            for record in matches:
                key = (record.source, record.premise)
                if key in seen:
                    continue
                seen.add(key)
                exact.append(record)
        detected_tuple = tuple(citations)
        unresolved_tuple = tuple(unresolved)
        results = tuple(
            replace(
                record,
                rank=rank,
                initial_rank=rank,
                detected_citations=detected_tuple,
                unresolved_citations=unresolved_tuple,
            )
            for rank, record in enumerate(exact, start=1)
        )
        return (
            RetrievalOutcome(
                results,
                results,
                False,
                detected_tuple,
                unresolved_tuple,
            ),
            detected_tuple,
        )

    def retrieve_rules_with_details(self, hypothesis: str) -> RetrievalOutcome:
        """Run deterministic citation extraction and exact lookup without FAISS."""
        outcome, _ = self._rule_outcome(hypothesis)
        return outcome

    def retrieve_with_details(
        self,
        hypothesis: str,
        *,
        top_k: int = 20,
        final_top_k: int | None = None,
    ) -> RetrievalOutcome:
        """Retrieve candidates and optionally rerank semantic fallback records."""
        candidate_top_k, retained_top_k = self._retrieval_depths(
            top_k, final_top_k
        )
        rule_outcome, detected_tuple = self._rule_outcome(hypothesis)
        if rule_outcome.results:
            return rule_outcome
        return self._semantic_outcome(
            hypothesis,
            candidate_top_k=candidate_top_k,
            retained_top_k=retained_top_k,
            detected_citations=detected_tuple,
        )

    def retrieve(
        self,
        hypothesis: str,
        *,
        top_k: int = 20,
        final_top_k: int | None = None,
    ) -> list[RetrievalRecord]:
        """Return final exact or semantic retrieval records."""
        return list(
            self.retrieve_with_details(
                hypothesis, top_k=top_k, final_top_k=final_top_k
            ).results
        )


def citation_dict(citation: Citation) -> dict[str, str | None]:
    """Return citation fields for tabular audit output."""
    return {f"citation_{key}": value for key, value in asdict(citation).items()}


def citations_json(citations: Sequence[Citation]) -> str:
    """Serialize ordered citations for tabular audit output."""
    return json.dumps([asdict(citation) for citation in citations], ensure_ascii=False)
