"""CPU-only audit of deterministic citation extraction against expert references."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from LCD.shared.common import REPOSITORY_ROOT, DEFAULT_RAG_DIR

from ..reporting import format_article_reference
from ..retrieval import Citation, PremiseRetriever, parse_codex_source
from .evaluation import (
    MISSING_COLUMNS,
    _annotations_by_hypothesis,
    _citation_key,
    _document_hypotheses,
    _merge_full_annotations,
    _resolve_workbook,
    normalize_sentence,
    read_rag_workbook,
)

SUMMARY_COLUMNS = (
    "routing_scope",
    "dataset",
    "pipeline_hypotheses",
    "annotated_hypotheses",
    "missing_annotation_hypotheses",
    "hypotheses_with_expert_articles",
    "expert_articles",
    "detected_articles",
    "matched_articles",
    "missed_expert_articles",
    "detected_not_in_expert",
    "detected_with_missing_annotation",
    "article_extraction_recall",
    "expert_full_references",
    "matched_full_references",
    "full_reference_recall",
    "resolved_expert_articles",
    "rule_retrieval_recall",
    "unresolved_detected_articles",
)

HYPOTHESIS_COLUMNS = (
    "dataset",
    "document",
    "sentence_index",
    "retrieval_route",
    "hypothesis",
    "annotation_status",
    "comparison_status",
    "expert_references",
    "detected_references",
    "resolved_references",
    "unresolved_references",
    "matched_articles",
    "missed_expert_articles",
    "detected_not_in_expert",
)

COMPARISON_COLUMNS = (
    "dataset",
    "document",
    "sentence_index",
    "retrieval_route",
    "hypothesis",
    "code",
    "article",
    "comparison_status",
    "expert_references",
    "detected_references",
    "resolved_references",
    "exact_lookup_status",
)


def _article_key(citation: Citation) -> tuple[str, str]:
    key = _citation_key(citation)
    return key[0], key[1]


def _unique_citations(citations: Iterable[Citation]) -> tuple[Citation, ...]:
    unique: dict[tuple[str, str, str, str], Citation] = {}
    for citation in citations:
        unique.setdefault(_citation_key(citation), citation)
    return tuple(unique.values())


def _group_by_article(
    citations: Iterable[Citation],
) -> dict[tuple[str, str], tuple[Citation, ...]]:
    grouped: dict[tuple[str, str], list[Citation]] = {}
    for citation in _unique_citations(citations):
        if citation.article:
            grouped.setdefault(_article_key(citation), []).append(citation)
    return {key: tuple(values) for key, values in grouped.items()}


def _format_citations(citations: Iterable[Citation]) -> str:
    references = []
    for citation in _unique_citations(citations):
        reference = format_article_reference(
            "",
            citation.code,
            citation.article,
            citation.part,
            citation.point,
        )
        if reference:
            references.append(reference)
    return "\n".join(references)


def _resolved_citations(records) -> tuple[Citation, ...]:
    citations = []
    for record in records:
        citation = parse_codex_source(record.source) or record.citation
        if citation.code and citation.article:
            citations.append(citation)
    return _unique_citations(citations)


def _comparison_status(
    *,
    annotated: bool,
    gold_full: set[tuple[str, str, str, str]],
    detected_full: set[tuple[str, str, str, str]],
    gold_articles: set[tuple[str, str]],
    detected_articles: set[tuple[str, str]],
) -> str:
    if not annotated:
        return "missing_annotation"
    if not gold_full and not detected_full:
        return "no_articles"
    missed = gold_articles - detected_articles
    extra = detected_articles - gold_articles
    if missed and extra:
        return "mixed"
    if missed:
        return "missed_expert"
    if extra:
        return "extracted_not_in_expert"
    if gold_full == detected_full:
        return "matched_full"
    return "matched_article_subprovision_diff"


def _audit_occurrence(
    dataset: str,
    occurrence: dict,
    annotation: dict | None,
    retriever: PremiseRetriever,
) -> dict:
    gold = tuple(annotation["gold_citations"]) if annotation is not None else ()
    outcome = retriever.retrieve_rules_with_details(occurrence["hypothesis"])
    detected = _unique_citations(outcome.detected_citations)
    unresolved = _unique_citations(outcome.unresolved_citations)
    resolved = _resolved_citations(outcome.results)

    gold_by_article = _group_by_article(gold)
    detected_by_article = _group_by_article(detected)
    resolved_by_article = _group_by_article(resolved)
    gold_full = {_citation_key(value) for value in gold}
    detected_full = {_citation_key(value) for value in detected}
    gold_articles = set(gold_by_article)
    detected_articles = set(detected_by_article)
    resolved_articles = set(resolved_by_article)
    annotated = annotation is not None

    item = {
        "dataset": dataset,
        **occurrence,
        "retrieval_route": "rules" if outcome.results else "faiss",
        "annotated": annotated,
        "gold": gold,
        "detected": detected,
        "resolved": resolved,
        "unresolved": unresolved,
        "gold_full": gold_full,
        "detected_full": detected_full,
        "gold_articles": gold_articles,
        "detected_articles": detected_articles,
        "resolved_articles": resolved_articles,
        "gold_by_article": gold_by_article,
        "detected_by_article": detected_by_article,
        "resolved_by_article": resolved_by_article,
    }
    item["comparison_status"] = _comparison_status(
        annotated=annotated,
        gold_full=gold_full,
        detected_full=detected_full,
        gold_articles=gold_articles,
        detected_articles=detected_articles,
    )
    return item


def _hypothesis_row(item: dict) -> dict:
    return {
        "dataset": item["dataset"],
        "document": item["document"],
        "sentence_index": item["sentence_index"],
        "retrieval_route": item["retrieval_route"],
        "hypothesis": item["hypothesis"],
        "annotation_status": "annotated" if item["annotated"] else "missing",
        "comparison_status": item["comparison_status"],
        "expert_references": _format_citations(item["gold"]),
        "detected_references": _format_citations(item["detected"]),
        "resolved_references": _format_citations(item["resolved"]),
        "unresolved_references": _format_citations(item["unresolved"]),
        "matched_articles": len(item["gold_articles"] & item["detected_articles"]),
        "missed_expert_articles": len(
            item["gold_articles"] - item["detected_articles"]
        ),
        "detected_not_in_expert": (
            len(item["detected_articles"] - item["gold_articles"])
            if item["annotated"]
            else None
        ),
    }


def _comparison_rows(item: dict) -> list[dict]:
    rows = []
    all_articles = item["gold_articles"] | item["detected_articles"]
    for article_key in sorted(all_articles):
        gold = item["gold_by_article"].get(article_key, ())
        detected = item["detected_by_article"].get(article_key, ())
        resolved = item["resolved_by_article"].get(article_key, ())
        if not item["annotated"]:
            status = "unannotated_extraction"
        elif gold and not detected:
            status = "missed_expert"
        elif detected and not gold:
            status = "extracted_not_in_expert"
        elif {_citation_key(value) for value in gold} == {
            _citation_key(value) for value in detected
        }:
            status = "matched_full"
        else:
            status = "matched_article_subprovision_diff"

        representative = (gold or detected)[0]
        lookup_status = (
            "not_extracted"
            if not detected
            else "resolved"
            if resolved
            else "unresolved"
        )
        rows.append(
            {
                "dataset": item["dataset"],
                "document": item["document"],
                "sentence_index": item["sentence_index"],
                "retrieval_route": item["retrieval_route"],
                "hypothesis": item["hypothesis"],
                "code": representative.code,
                "article": representative.article,
                "comparison_status": status,
                "expert_references": _format_citations(gold),
                "detected_references": _format_citations(detected),
                "resolved_references": _format_citations(resolved),
                "exact_lookup_status": lookup_status,
            }
        )
    return rows


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _summary_row(dataset: str, items: Sequence[dict], routing_scope: str) -> dict:
    annotated = [item for item in items if item["annotated"]]
    expert_articles = sum(len(item["gold_articles"]) for item in annotated)
    matched_articles = sum(
        len(item["gold_articles"] & item["detected_articles"])
        for item in annotated
    )
    expert_full = sum(len(item["gold_full"]) for item in annotated)
    matched_full = sum(
        len(item["gold_full"] & item["detected_full"]) for item in annotated
    )
    resolved_expert = sum(
        len(item["gold_articles"] & item["resolved_articles"])
        for item in annotated
    )
    return {
        "routing_scope": routing_scope,
        "dataset": dataset,
        "pipeline_hypotheses": len(items),
        "annotated_hypotheses": len(annotated),
        "missing_annotation_hypotheses": len(items) - len(annotated),
        "hypotheses_with_expert_articles": sum(
            bool(item["gold_articles"]) for item in annotated
        ),
        "expert_articles": expert_articles,
        "detected_articles": sum(len(item["detected_articles"]) for item in items),
        "matched_articles": matched_articles,
        "missed_expert_articles": expert_articles - matched_articles,
        "detected_not_in_expert": sum(
            len(item["detected_articles"] - item["gold_articles"])
            for item in annotated
        ),
        "detected_with_missing_annotation": sum(
            len(item["detected_articles"]) for item in items if not item["annotated"]
        ),
        "article_extraction_recall": _safe_ratio(matched_articles, expert_articles),
        "expert_full_references": expert_full,
        "matched_full_references": matched_full,
        "full_reference_recall": _safe_ratio(matched_full, expert_full),
        "resolved_expert_articles": resolved_expert,
        "rule_retrieval_recall": _safe_ratio(resolved_expert, expert_articles),
        "unresolved_detected_articles": sum(
            len(item["detected_articles"] - item["resolved_articles"])
            for item in items
        ),
    }


def _timestamped_output(results_dir: str | Path, routing_scope: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(results_dir) / f"citation_audit_{routing_scope}_{stamp}.xlsx"


def _normalize_routing_scope(value: str) -> str:
    scope = str(value).strip().lower()
    if scope not in {"all", "rules", "faiss"}:
        raise ValueError("routing_scope must be all, rules, or faiss")
    return scope


def run_citation_audit(
    *,
    codex_path: str | Path = DEFAULT_RAG_DIR / "codex.csv",
    rag_test_dir: str | Path = REPOSITORY_ROOT / "local/data/benchmarks/rag_tests",
    dialogue_workbook: str | Path | None = None,
    full_workbook: str | Path | None = None,
    full_additional_workbook: str | Path | None = None,
    test_docx_dir: str | Path = REPOSITORY_ROOT / "local/data/benchmarks/test_docx",
    results_dir: str | Path = REPOSITORY_ROOT / "local/experiments/lawbook/series_01_embedding_reranking/results/citation_audit",
    output_path: str | Path | None = None,
    routing_scope: str = "all",
    selected_datasets: Sequence[str] = ("Full",),
) -> Path:
    """Compare production citation extraction with expert RAG annotations.

    The workflow reads only filtered operative-section hypotheses and performs
    deterministic extraction plus exact ``codex.csv`` lookup. It does not load
    an embedding model, FAISS index, reranker, or GPU dependency. Routing scope
    can retain all hypotheses or only the production rules/FAISS subset.
    """
    import pandas as pd

    scope = _normalize_routing_scope(routing_scope)
    target = (
        Path(output_path)
        if output_path
        else _timestamped_output(results_dir, scope)
    )
    if target.exists():
        raise FileExistsError(f"Citation audit output already exists: {target}")
    codex = Path(codex_path)
    if not codex.is_file():
        raise FileNotFoundError(f"codex.csv is missing: {codex}")
    retriever = PremiseRetriever(pd.read_csv(codex), vectorstore=None)

    workbook_root = Path(rag_test_dir)
    selected = tuple(name.upper() for name in selected_datasets)
    if not selected or len(set(selected)) != len(selected) or set(selected) - {"FULL", "DIALOGUE"}:
        raise ValueError("selected_datasets must contain unique Full/Dialogue names")
    annotations = {}
    checked_workbooks = {}
    if "DIALOGUE" in selected:
        dialogue_path = _resolve_workbook(dialogue_workbook, workbook_root, "RAG_DIALOGUE_test.xlsx")
        annotations["DIALOGUE"] = _annotations_by_hypothesis(read_rag_workbook(dialogue_path))
        checked_workbooks["DIALOGUE"] = (dialogue_path,)
    if "FULL" in selected:
        full_path = _resolve_workbook(full_workbook, workbook_root, "RAG_FULL_test.xlsx")
        additional_path = _resolve_workbook(full_additional_workbook, workbook_root, "RAG_FULL_additional_test.xlsx")
        annotations["FULL"] = _merge_full_annotations(read_rag_workbook(full_path), read_rag_workbook(additional_path))
        checked_workbooks["FULL"] = (full_path, additional_path)

    document_root = Path(test_docx_dir)
    items = []
    missing = []
    for dataset in selected:
        directory_name = dataset.title()
        dataset_missing = []
        for occurrence in _document_hypotheses(document_root / directory_name):
            annotation = annotations[dataset].get(occurrence["normalized_hypothesis"])
            item = _audit_occurrence(dataset, occurrence, annotation, retriever)
            if scope != "all" and item["retrieval_route"] != scope:
                continue
            items.append(item)
            if annotation is None:
                dataset_missing.append(
                    {
                        "dataset": dataset,
                        "document": occurrence["document"],
                        "sentence_index": occurrence["sentence_index"],
                        "hypothesis": occurrence["hypothesis"],
                        "checked_workbooks": " | ".join(
                            path.name for path in checked_workbooks[dataset]
                        ),
                    }
                )
        if dataset_missing:
            unique = len(
                {
                    normalize_sentence(row["hypothesis"])
                    for row in dataset_missing
                }
            )
            warnings.warn(
                f"{dataset}: {len(dataset_missing)} DOCX hypothesis occurrences "
                f"({unique} unique) are missing from rag_tests and excluded "
                "from Recall",
                UserWarning,
                stacklevel=2,
            )
            missing.extend(dataset_missing)

    summary_rows = [
        _summary_row(
            dataset,
            [item for item in items if item["dataset"] == dataset],
            scope,
        )
        for dataset in selected
    ]
    summary_rows.append(_summary_row("ALL", items, scope))
    hypothesis_rows = [_hypothesis_row(item) for item in items]
    comparison_rows = [row for item in items for row in _comparison_rows(item)]

    target.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(target, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS).to_excel(
            writer, sheet_name="summary", index=False
        )
        pd.DataFrame(hypothesis_rows, columns=HYPOTHESIS_COLUMNS).to_excel(
            writer, sheet_name="hypotheses", index=False
        )
        pd.DataFrame(comparison_rows, columns=COMPARISON_COLUMNS).to_excel(
            writer, sheet_name="citation_comparison", index=False
        )
        pd.DataFrame(missing, columns=MISSING_COLUMNS).to_excel(
            writer, sheet_name="missing_hypotheses", index=False
        )
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
    return target.resolve()
