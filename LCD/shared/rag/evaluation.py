"""Compact DOCX-driven Recall evaluation for legal premise retrieval."""

from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from LCD.shared.common import REPOSITORY_ROOT, DEFAULT_RAG_DIR

from ..documents import (
    extract_operative_section,
    read_docx_text,
    split_russian_sentences,
    textcheck,
)
from ..retrieval import Citation, PremiseRetriever, extract_citations, parse_codex_source
from .artifacts import RagBundle, load_rag_bundle
from .reranking import Reranker, score_and_sort_records

DEFAULT_RETRIEVAL_DEPTHS: tuple[tuple[int, int], ...] = ((20, 10), (40, 20))
SUMMARY_COLUMNS = (
    "variant",
    "candidate_top_k",
    "final_top_k",
    "dialogue_faiss_recall",
    "dialogue_rules_recall",
    "dialogue_total_recall",
    "full_faiss_recall",
    "full_rules_recall",
    "full_total_recall",
)
EVALUATION_VARIANTS = (
    "baseline_embeddings__no_reranker",
    "baseline_embeddings__pretrained_reranker",
    "tuned_embeddings__no_reranker",
    "tuned_embeddings__pretrained_reranker",
    "baseline_embeddings__finetuned_reranker",
    "tuned_embeddings__finetuned_reranker",
)
MISSING_COLUMNS = (
    "dataset",
    "document",
    "sentence_index",
    "hypothesis",
    "checked_workbooks",
)


def normalize_sentence(value: object) -> str:
    """Normalize a hypothesis for conservative XLSX/DOCX alignment."""
    return " ".join(str(value or "").replace("\xa0", " ").split()).casefold()


def _citation_key(citation: Citation) -> tuple[str, str, str, str]:
    return tuple(
        " ".join(str(value or "").split()).casefold()
        for value in (citation.code, citation.article, citation.part, citation.point)
    )  # type: ignore[return-value]


def _parse_gold(value: str) -> Citation:
    citations = extract_citations(value)
    if len(citations) != 1 or not citations[0].article:
        raise ValueError(f"Could not parse one gold citation: {value!r}")
    return citations[0]


def read_rag_workbook(path: str | Path):
    """Read a headerless three-column RAG annotation workbook."""
    import pandas as pd

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"RAG annotation workbook is missing: {source}")
    frame = pd.read_excel(
        source, header=None, dtype=str, keep_default_na=False, engine="openpyxl"
    )
    if frame.shape[1] != 3:
        raise ValueError(f"{source} must have exactly three columns")
    frame.columns = ["hypothesis", "gold_reference", "gold_text"]
    frame["normalized_hypothesis"] = frame["hypothesis"].map(normalize_sentence)
    if frame["normalized_hypothesis"].eq("").any():
        raise ValueError(f"{source} contains a blank hypothesis")
    rows = []
    for normalized, group in frame.groupby("normalized_hypothesis", sort=False):
        references = [value for value in group["gold_reference"] if value.strip()]
        citations: dict[tuple[str, str, str, str], Citation] = {}
        for reference in references:
            citation = _parse_gold(reference)
            citations[_citation_key(citation)] = citation
        rows.append(
            {
                "normalized_hypothesis": normalized,
                "hypothesis": group.iloc[0]["hypothesis"],
                "gold_citations": tuple(citations.values()),
                "gold_references": tuple(dict.fromkeys(references)),
                "gold_texts": tuple(
                    dict.fromkeys(
                        value for value in group["gold_text"] if value.strip()
                    )
                ),
            }
        )
    return rows


def _document_hypotheses(directory: Path) -> list[dict]:
    """Extract every hypothesis occurrence exactly as full inference does."""
    documents = sorted(directory.glob("*.docx"))
    if not documents:
        raise FileNotFoundError(f"No DOCX test documents found in {directory}")
    rows = []
    for document in documents:
        section = extract_operative_section(read_docx_text(document))
        if section is None:
            warnings.warn(
                f"{document.name}: ПОСТАНОВИЛ section was not found; document ignored",
                UserWarning,
                stacklevel=2,
            )
            continue
        for sentence_index, hypothesis in enumerate(split_russian_sentences(section)):
            if not textcheck(hypothesis):
                rows.append(
                    {
                        "document": document.name,
                        "sentence_index": sentence_index,
                        "hypothesis": hypothesis,
                        "normalized_hypothesis": normalize_sentence(hypothesis),
                    }
                )
    return rows


def _annotations_by_hypothesis(rows) -> dict[str, dict]:
    return {row["normalized_hypothesis"]: row for row in rows}


def _merge_full_annotations(primary_rows, additional_rows) -> dict[str, dict]:
    """Use additional annotations only as fallback and reject conflicts."""
    primary = _annotations_by_hypothesis(primary_rows)
    additional = _annotations_by_hypothesis(additional_rows)
    for hypothesis in primary.keys() & additional.keys():
        primary_gold = {
            _citation_key(value)
            for value in primary[hypothesis]["gold_citations"]
        }
        additional_gold = {
            _citation_key(value) for value in additional[hypothesis]["gold_citations"]
        }
        if primary_gold != additional_gold:
            raise ValueError(
                "Conflicting Full RAG annotations for one normalized hypothesis "
                "between primary and additional workbooks"
            )
    return {**additional, **primary}


def _align_dataset(
    dataset: str,
    document_rows,
    annotations: dict[str, dict],
    checked_workbooks: Sequence[Path],
):
    aligned = []
    missing = []
    checked = " | ".join(path.name for path in checked_workbooks)
    for occurrence in document_rows:
        annotation = annotations.get(occurrence["normalized_hypothesis"])
        if annotation is None:
            missing.append(
                {
                    "dataset": dataset,
                    "document": occurrence["document"],
                    "sentence_index": occurrence["sentence_index"],
                    "hypothesis": occurrence["hypothesis"],
                    "checked_workbooks": checked,
                }
            )
            continue
        aligned.append(
            {
                **occurrence,
                "gold_citations": annotation["gold_citations"],
            }
        )
    if missing:
        unique_missing = len(
            {normalize_sentence(row["hypothesis"]) for row in missing}
        )
        warnings.warn(
            f"{dataset}: {len(missing)} DOCX hypothesis occurrences "
            f"({unique_missing} unique) are missing from rag_tests and excluded "
            "from Recall",
            UserWarning,
            stacklevel=2,
        )
    return aligned, missing


def _validate_retrieval_depths(
    values: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    depths = tuple((int(candidate), int(final)) for candidate, final in values)
    if not depths:
        raise ValueError("retrieval_depths cannot be empty")
    if len(set(depths)) != len(depths):
        raise ValueError("retrieval_depths cannot contain duplicates")
    for candidate, final in depths:
        if candidate <= 0 or final <= 0:
            raise ValueError("retrieval depths must be positive")
        if final > candidate:
            raise ValueError("final_top_k cannot exceed candidate_top_k")
    return depths


def _retrieved_citation(source: str) -> Citation | None:
    parsed = parse_codex_source(source)
    if parsed is not None:
        return parsed
    citations = extract_citations(source)
    return citations[0] if len(citations) == 1 else None


def _retrieved_keys(records) -> set[tuple[str, str, str, str]]:
    citations = (_retrieved_citation(record.source) for record in records)
    return {_citation_key(value) for value in citations if value is not None}


def _records_for_depth(records, candidate_top_k: int, final_top_k: int):
    within_pool = [
        record
        for record in records
        if int(record.initial_rank or record.rank) <= candidate_top_k
    ]
    return tuple(within_pool[:final_top_k])


def _make_retriever(bundle: RagBundle, device: str) -> PremiseRetriever:
    return PremiseRetriever.from_components(
        bundle.codex_path,
        bundle.index_dir,
        embedding_model=bundle.embedding_model,
        embedding_revision=bundle.embedding_revision,
        embedding_device=device,
        normalize_embeddings=bundle.normalize_embeddings,
        embedding_query_prefix=bundle.embedding_query_prefix,
        embedding_document_prefix=bundle.embedding_document_prefix,
        embedding_trust_remote_code=bundle.embedding_trust_remote_code,
        embedding_precision=bundle.embedding_precision,
        embedding_batch_size=bundle.embedding_batch_size,
    )


def _resolve_workbook(value: str | Path | None, root: Path, default: str) -> Path:
    path = root / default if value is None else Path(value)
    return path if path.is_absolute() else path.resolve()


def _select_variant_definitions(
    requested: Sequence[str] | None,
    *,
    pretrained_reranker: Reranker | None,
    finetuned_reranker: Reranker | None,
) -> list[tuple[str, int, str | None]]:
    definitions = {
        "baseline_embeddings__no_reranker": (0, None),
        "baseline_embeddings__pretrained_reranker": (0, "pretrained_reranker"),
        "tuned_embeddings__no_reranker": (1, None),
        "tuned_embeddings__pretrained_reranker": (1, "pretrained_reranker"),
        "baseline_embeddings__finetuned_reranker": (0, "finetuned_reranker"),
        "tuned_embeddings__finetuned_reranker": (1, "finetuned_reranker"),
    }
    if requested is None:
        if pretrained_reranker is None:
            raise ValueError("pretrained_reranker is required for full RAG comparison")
        names = list(EVALUATION_VARIANTS[:4])
        if finetuned_reranker is not None:
            names.extend(EVALUATION_VARIANTS[4:])
    else:
        names = [str(value) for value in requested]
        if not names:
            raise ValueError("evaluation_variants cannot be empty")
        if len(set(names)) != len(names):
            raise ValueError("evaluation_variants cannot contain duplicates")
        unknown = [name for name in names if name not in definitions]
        if unknown:
            raise ValueError(f"Unknown RAG evaluation variants: {unknown}")
    if any("pretrained_reranker" in name for name in names):
        if pretrained_reranker is None:
            raise ValueError("A selected variant requires pretrained_reranker")
    if any("finetuned_reranker" in name for name in names):
        if finetuned_reranker is None:
            raise ValueError("A selected variant requires finetuned_reranker")
    return [(name, *definitions[name]) for name in names]


def run_rag_evaluation(
    rag_sources: Sequence[str | Path | RagBundle],
    *,
    rag_test_dir: str | Path = REPOSITORY_ROOT / "local/data/benchmarks/rag_tests",
    dialogue_workbook: str | Path | None = None,
    full_workbook: str | Path | None = None,
    full_additional_workbook: str | Path | None = None,
    test_docx_dir: str | Path = REPOSITORY_ROOT / "local/data/benchmarks/test_docx",
    results_dir: str | Path = REPOSITORY_ROOT / "local/experiments/lawbook/series_01_embedding_reranking/results/evaluation",
    embedding_device: str = "cpu",
    retriever_factory: Callable[[RagBundle, str], PremiseRetriever] | None = None,
    retrieval_depths: Sequence[tuple[int, int]] = DEFAULT_RETRIEVAL_DEPTHS,
    pretrained_reranker: Reranker | None = None,
    finetuned_reranker: Reranker | None = None,
    evaluation_variants: Sequence[str] | None = None,
    selected_datasets: Sequence[str] = ("Full",),
):
    """Write method-conditional and total Recall for each retrieval variant."""
    import pandas as pd

    depths = _validate_retrieval_depths(retrieval_depths)
    bundles = [
        source if isinstance(source, RagBundle) else load_rag_bundle(source)
        for source in rag_sources
    ]
    if len(bundles) != 2:
        raise ValueError("rag_sources must contain baseline and tuned RAG versions")

    workbook_root = Path(rag_test_dir)
    selected = tuple(name.casefold() for name in selected_datasets)
    if not selected or len(set(selected)) != len(selected) or set(selected) - {"dialogue", "full"}:
        raise ValueError("selected_datasets must contain unique Full/Dialogue names")
    document_root = Path(test_docx_dir)
    datasets = {}
    missing_rows = []
    if "dialogue" in selected:
        dialogue_path = _resolve_workbook(dialogue_workbook, workbook_root, "RAG_DIALOGUE_test.xlsx")
        rows, missing = _align_dataset(
            "DIALOGUE", _document_hypotheses(document_root / "Dialogue"),
            _annotations_by_hypothesis(read_rag_workbook(dialogue_path)), (dialogue_path,),
        )
        datasets["dialogue"] = rows
        missing_rows.extend(missing)
    if "full" in selected:
        full_path = _resolve_workbook(full_workbook, workbook_root, "RAG_FULL_test.xlsx")
        full_additional_path = _resolve_workbook(
            full_additional_workbook, workbook_root, "RAG_FULL_additional_test.xlsx"
        )
        full_annotations = _merge_full_annotations(
            read_rag_workbook(full_path), read_rag_workbook(full_additional_path)
        )
        rows, missing = _align_dataset(
            "FULL", _document_hypotheses(document_root / "Full"), full_annotations,
            (full_path, full_additional_path),
        )
        datasets["full"] = rows
        missing_rows.extend(missing)

    rerankers = {
        "pretrained_reranker": pretrained_reranker,
        "finetuned_reranker": finetuned_reranker,
    }
    variant_definitions = _select_variant_definitions(
        evaluation_variants,
        pretrained_reranker=pretrained_reranker,
        finetuned_reranker=finetuned_reranker,
    )

    counters: dict[tuple[str, int, int, str, str], list[int]] = defaultdict(
        lambda: [0, 0]
    )
    make_retriever = retriever_factory or _make_retriever
    maximum_candidates = max(candidate for candidate, _ in depths)
    for bundle_index, bundle in enumerate(bundles):
        bundle_variants = [
            definition
            for definition in variant_definitions
            if definition[1] == bundle_index
        ]
        if not bundle_variants:
            continue
        retriever = make_retriever(bundle, embedding_device)
        for dataset_name, rows in datasets.items():
            for row in rows:
                gold_keys = {
                    _citation_key(value) for value in row["gold_citations"]
                }
                if not gold_keys:
                    continue
                rules = retriever.retrieve_rules_with_details(
                    row["hypothesis"]
                ).results
                semantic = ()
                ranked_by_reranker = {}
                if not rules:
                    semantic = retriever.retrieve_semantic_with_details(
                        row["hypothesis"],
                        top_k=maximum_candidates,
                        final_top_k=maximum_candidates,
                    ).candidates
                    ranked_by_reranker = {
                        key: score_and_sort_records(
                            row["hypothesis"], semantic, model
                        )
                        for key, model in rerankers.items()
                        if model is not None
                    }
                for variant, _, reranker_key in bundle_variants:
                    for candidate_top_k, final_top_k in depths:
                        if rules:
                            system_records = {"rules": rules, "total": rules}
                        else:
                            semantic_ranking = (
                                semantic
                                if reranker_key is None
                                else ranked_by_reranker[reranker_key]
                            )
                            faiss_records = _records_for_depth(
                                semantic_ranking, candidate_top_k, final_top_k
                            )
                            system_records = {
                                "faiss": faiss_records,
                                "total": faiss_records,
                            }
                        for system, records in system_records.items():
                            retrieved = _retrieved_keys(records)
                            counter = counters[
                                (
                                    variant,
                                    candidate_top_k,
                                    final_top_k,
                                    dataset_name,
                                    system,
                                )
                            ]
                            counter[0] += len(gold_keys & retrieved)
                            counter[1] += len(gold_keys)

    summary_rows = []
    for variant, _, _ in variant_definitions:
        for candidate_top_k, final_top_k in depths:
            result = {
                "variant": variant,
                "candidate_top_k": candidate_top_k,
                "final_top_k": final_top_k,
            }
            for dataset_name in selected:
                for system in ("faiss", "rules", "total"):
                    hits, gold = counters[
                        (
                            variant,
                            candidate_top_k,
                            final_top_k,
                            dataset_name,
                            system,
                        )
                    ]
                    result[f"{dataset_name}_{system}_recall"] = (
                        hits / gold if gold else None
                    )
            summary_rows.append(result)
    score_columns = ("variant", "candidate_top_k", "final_top_k") + tuple(
        f"{dataset}_{method}_recall" for dataset in selected
        for method in ("faiss", "rules", "total")
    )
    scores = pd.DataFrame(summary_rows, columns=score_columns)
    missing = pd.DataFrame(missing_rows, columns=MISSING_COLUMNS)
    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / "rag_recall.csv", index=False, encoding="utf-8")
    with pd.ExcelWriter(output / "rag_recall.xlsx", engine="openpyxl") as writer:
        scores.to_excel(writer, sheet_name="recall", index=False)
        missing.to_excel(writer, sheet_name="missing_hypotheses", index=False)
    return scores
