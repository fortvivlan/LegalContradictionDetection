from dataclasses import dataclass
from pathlib import Path
import subprocess

import pandas as pd
import pytest

from LCD.shared.retrieval import (
    Citation,
    PremiseRetriever,
    ensure_rag_repository,
    extract_citation,
    extract_citations,
    parse_codex_source,
)
from LCD.shared.rag.reranking import score_and_sort_records


@dataclass
class FakeDocument:
    page_content: str
    metadata: dict[str, str]


class FakeVectorStore:
    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    def similarity_search_with_score(self, query: str, k: int):
        self.calls.append((query, k))
        return [
            (FakeDocument(f"semantic {index}", {"source": f"s{index}"}), index / 10)
            for index in range(k)
        ]


class FakeReranker:
    model_id = "fake/reranker"
    revision = "revision"

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def score(self, query, documents):
        self.calls.append((query, list(documents)))
        return self.scores[: len(documents)]


def _retriever() -> tuple[PremiseRetriever, FakeVectorStore]:
    codex = pd.DataFrame(
        {
            "text": [
                "5.35/1",
                "5.35.1/1",
                "18.8/3",
                "18.8/3.1",
                "32.9/1/1",
                "32.9/1/2",
                "29.9",
                "29.10",
                "29.11",
                "instruction",
            ],
            "source": [
                "КоАП РФ: Статья 5.35. ч. 1 п. 1.",
                "КоАП РФ: Статья 5.35.1. ч. 1 п. 1.",
                "КоАП РФ: Статья 18.8. ч. 3.",
                "КоАП РФ: Статья 18.8. ч. 3.1.",
                "КоАП РФ: Статья 32.9. ч. 1 п. 1.",
                "КоАП РФ: Статья 32.9. ч. 1 п. 2.",
                "КоАП РФ: Статья 29.9.",
                "КоАП РФ: Статья 29.10.",
                "КоАП РФ: Статья 29.11.",
                "Инструкция ОС: Статья 7. ч. 2.",
            ],
        }
    )
    vectorstore = FakeVectorStore()
    return PremiseRetriever(codex, vectorstore), vectorstore


def test_extract_citation_preserves_dotted_numbers_and_long_code_name() -> None:
    citation = extract_citation(
        "по ч. 3.1 ст. 18.8 Кодекса Российской Федерации "
        "об административных правонарушениях"
    )
    assert citation.article == "18.8"
    assert citation.part == "3.1"
    assert citation.code == "КоАП РФ"


def test_extract_citations_supports_both_orders_and_full_words() -> None:
    assert extract_citations("п. 2 ч. 1 ст. 32.9 КоАП РФ") == [
        Citation("КоАП РФ", "32.9", "1", "2")
    ]
    assert extract_citations("ст. 32.9 ч. 1 п. 2 КоАП РФ") == [
        Citation("КоАП РФ", "32.9", "1", "2")
    ]
    assert extract_citations("ст. 27.19.1 ч 1 КоАП РФ") == [
        Citation("КоАП РФ", "27.19.1", "1", None)
    ]
    assert extract_citations(
        "Частью 3 статьи 18.8 Кодекса Российской Федерации "
        "об административных правонарушениях"
    ) == [Citation("КоАП РФ", "18.8", "3", None)]


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("ст. 1225 Гражданского кодекса Российской Федерации", "ГК РФ"),
        ("ст. 1 Гражданского процессуального кодекса РФ", "ГПК РФ"),
        ("ст. 1 Кодекса административного судопроизводства РФ", "КАС РФ"),
        ("ст. 1 Трудового кодекса Российской Федерации", "ТК"),
        ("ст. 1 Уголовно-исполнительного кодекса РФ", "УИК РФ"),
        ("ст. 1 Уголовного кодекса РФ", "УК РФ"),
        ("ст. 1 Уголовно-процессуального кодекса РФ", "УПК РФ"),
    ],
)
def test_extract_citations_recognizes_full_names_for_corpus_codes(
    text: str, code: str
) -> None:
    assert extract_citations(text)[0].code == code


def test_extract_citations_keeps_codes_local_to_each_article() -> None:
    citations = extract_citations(
        "ч. 2 ст. 5 Федерального закона № 115-ФЗ, то есть "
        "ч. 3.1 ст. 18.8 КоАП РФ"
    )
    assert citations == [
        Citation(None, "5", "2", None),
        Citation("КоАП РФ", "18.8", "3.1", None),
    ]


def test_extract_citations_expands_lists_and_same_prefix_ranges() -> None:
    citations = extract_citations(
        "руководствуясь ст.ст. 3.5, 29.9-29.11 КоАП РФ"
    )
    assert [citation.article for citation in citations] == [
        "3.5",
        "29.9",
        "29.10",
        "29.11",
    ]
    assert {citation.code for citation in citations} == {"КоАП РФ"}


def test_parse_updated_codex_source_hierarchy() -> None:
    assert parse_codex_source("КоАП РФ: Статья 32.9. ч. 1 п. 2.") == Citation(
        "КоАП РФ", "32.9", "1", "2"
    )
    assert parse_codex_source("ТК: Статья 58. п. 1.") == Citation(
        "ТК", "58", None, "1"
    )
    assert parse_codex_source("ГК РФ ч.4: Статья 1225. ч. 1 п. 1.") == Citation(
        "ГК РФ ч.4", "1225", "1", "1"
    )


def test_exact_lookup_does_not_confuse_dotted_article_prefixes() -> None:
    retriever, vectorstore = _retriever()

    records = retriever.retrieve("п. 1 ч. 1 ст. 5.35 КоАП РФ")

    assert [record.source for record in records] == [
        "КоАП РФ: Статья 5.35. ч. 1 п. 1."
    ]
    assert records[0].method == "exact"
    assert vectorstore.calls == []


def test_exact_lookup_distinguishes_point_3_from_3_1() -> None:
    retriever, _ = _retriever()
    records = retriever.retrieve("ч. 3.1 ст. 18.8 КоАП РФ")
    assert len(records) == 1
    assert "ч. 3.1." in records[0].source


def test_part_lookup_returns_all_points_and_point_lookup_returns_one() -> None:
    retriever, vectorstore = _retriever()

    part_records = retriever.retrieve("ст. 32.9 ч. 1 КоАП РФ")
    point_records = retriever.retrieve("п. 2 ч. 1 ст. 32.9 КоАП РФ")

    assert [record.source for record in part_records] == [
        "КоАП РФ: Статья 32.9. ч. 1 п. 1.",
        "КоАП РФ: Статья 32.9. ч. 1 п. 2.",
    ]
    assert [record.premise for record in point_records] == ["32.9/1/2"]
    assert vectorstore.calls == []


def test_multiple_and_partially_resolved_citations_do_not_mix_in_faiss() -> None:
    retriever, vectorstore = _retriever()

    records = retriever.retrieve(
        "ст. 29.9 КоАП РФ и ч. 1 ст. 999 КоАП РФ"
    )

    assert [record.premise for record in records] == ["29.9"]
    assert records[0].unresolved_citations == (
        Citation("КоАП РФ", "999", "1", None),
    )
    assert vectorstore.calls == []


def test_literal_code_aliases_are_derived_from_codex() -> None:
    retriever, vectorstore = _retriever()

    records = retriever.retrieve("ст. 7 ч. 2 Инструкция ОС")

    assert [record.premise for record in records] == ["instruction"]
    assert vectorstore.calls == []


def test_ambiguous_citation_uses_configurable_candidate_depth() -> None:
    retriever, vectorstore = _retriever()

    records = retriever.retrieve("ст. 18.8 КоАП РФ", top_k=200)

    assert len(records) == 200
    assert all(record.method == "faiss" for record in records)
    assert vectorstore.calls == [("ст. 18.8 КоАП РФ", 200)]


def test_semantic_candidates_are_reranked_then_truncated_stably() -> None:
    retriever, vectorstore = _retriever()
    reranker = FakeReranker([0.1, 0.9, 0.9, 0.2])
    retriever._reranker = reranker

    outcome = retriever.retrieve_with_details(
        "без точной ссылки", top_k=4, final_top_k=2
    )

    assert vectorstore.calls == [("без точной ссылки", 4)]
    assert [record.initial_rank for record in outcome.candidates] == [1, 2, 3, 4]
    assert [record.initial_rank for record in outcome.results] == [2, 3]
    assert [record.rank for record in outcome.results] == [1, 2]
    assert [record.reranker_score for record in outcome.results] == [0.9, 0.9]
    assert reranker.calls[0][0] == "без точной ссылки"


def test_semantic_retrieval_deduplicates_identical_index_chunks() -> None:
    retriever, vectorstore = _retriever()
    duplicate = FakeDocument("same premise", {"source": "same source"})
    vectorstore.similarity_search_with_score = lambda query, k: [
        (duplicate, 0.1),
        (duplicate, 0.2),
        (FakeDocument("unique premise", {"source": "unique source"}), 0.3),
    ]

    outcome = retriever.retrieve_semantic_with_details(
        "без точной ссылки", top_k=3, final_top_k=3
    )

    assert [record.premise for record in outcome.results] == [
        "same premise",
        "unique premise",
    ]
    assert [record.rank for record in outcome.results] == [1, 2]


def test_full_reranked_pool_retains_scores_for_dropped_candidates() -> None:
    retriever, _ = _retriever()
    candidates = retriever.retrieve_with_details(
        "без точной ссылки", top_k=3, final_top_k=3
    ).candidates

    ranked = score_and_sort_records(
        "без точной ссылки", candidates, FakeReranker([0.2, 0.9, 0.1])
    )

    assert [record.initial_rank for record in ranked] == [2, 1, 3]
    assert [record.rank for record in ranked] == [1, 2, 3]
    assert [record.reranker_score for record in ranked] == [0.9, 0.2, 0.1]


def test_exact_results_bypass_reranker_and_final_truncation() -> None:
    retriever, _ = _retriever()
    reranker = FakeReranker([1.0])
    retriever._reranker = reranker

    outcome = retriever.retrieve_with_details(
        "ч. 1 ст. 32.9 КоАП РФ", top_k=5, final_top_k=1
    )

    assert len(outcome.results) == 2
    assert not outcome.reranked
    assert reranker.calls == []


def test_rule_only_retrieval_never_falls_back_to_faiss() -> None:
    retriever, vectorstore = _retriever()

    resolved = retriever.retrieve_rules_with_details(
        "п. 2 ч. 1 ст. 32.9 КоАП РФ"
    )
    unresolved = retriever.retrieve_rules_with_details(
        "ч. 1 ст. 999 КоАП РФ"
    )

    assert [record.premise for record in resolved.results] == ["32.9/1/2"]
    assert unresolved.results == ()
    assert unresolved.detected_citations == (
        Citation("КоАП РФ", "999", "1", None),
    )
    assert unresolved.unresolved_citations == unresolved.detected_citations
    assert vectorstore.calls == []


def test_semantic_only_retrieval_bypasses_an_available_exact_citation() -> None:
    retriever, vectorstore = _retriever()

    outcome = retriever.retrieve_semantic_with_details(
        "ч. 1 ст. 32.9 КоАП РФ", top_k=3, final_top_k=2
    )

    assert vectorstore.calls == [("ч. 1 ст. 32.9 КоАП РФ", 3)]
    assert len(outcome.candidates) == 3
    assert len(outcome.results) == 2
    assert {record.method for record in outcome.results} == {"faiss"}
    assert outcome.results[0].detected_citations == ()


@pytest.mark.parametrize(
    ("candidate_top_k", "final_top_k", "message"),
    [(0, 1, "positive"), (5, 0, "positive"), (5, 6, "cannot exceed")],
)
def test_retrieval_depth_validation(candidate_top_k, final_top_k, message) -> None:
    retriever, _ = _retriever()
    with pytest.raises(ValueError, match=message):
        retriever.retrieve(
            "без ссылки", top_k=candidate_top_k, final_top_k=final_top_k
        )


def _write_rag_artifacts(repository: Path, marker: str) -> None:
    (repository / "faiss_index").mkdir(exist_ok=True)
    (repository / "faiss_index" / "index.faiss").write_bytes(b"index")
    (repository / "faiss_index" / "index.pkl").write_bytes(b"metadata")
    (repository / "codex.csv").write_text(
        f"text,source\n{marker},s\n", encoding="utf-8"
    )


def _commit(repository: Path, message: str) -> str:
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-q", "-m", message],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_rag_main_is_refetched_on_every_run(tmp_path: Path) -> None:
    origin = tmp_path / "origin"
    origin.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(origin)], check=True)
    subprocess.run(
        ["git", "-C", str(origin), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(origin), "config", "user.name", "Test"], check=True
    )
    _write_rag_artifacts(origin, "first")
    first = _commit(origin, "first")
    checkout = tmp_path / "checkout"

    _, first_resolved = ensure_rag_repository(
        checkout, repository_url=str(origin), revision="main"
    )
    _write_rag_artifacts(origin, "second")
    second = _commit(origin, "second")
    _, second_resolved = ensure_rag_repository(
        checkout, repository_url=str(origin), revision="main"
    )

    assert first_resolved == first
    assert second_resolved == second
    assert (checkout / "codex.csv").read_text(encoding="utf-8") == (
        "text,source\nsecond,s\n"
    )


def test_rag_repository_is_checked_out_at_requested_commit(tmp_path: Path) -> None:
    repository = tmp_path / "rag"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Test"],
        check=True,
    )
    (repository / "faiss_index").mkdir()
    (repository / "faiss_index" / "index.faiss").write_bytes(b"index")
    (repository / "faiss_index" / "index.pkl").write_bytes(b"metadata")
    (repository / "codex.csv").write_text("text,source\na,s\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-q", "-m", "artifacts"],
        check=True,
    )
    pinned = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (repository / "README.md").write_text("later change", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-q", "-m", "later"],
        check=True,
    )

    path, resolved = ensure_rag_repository(repository, revision=pinned)

    assert path == repository
    assert resolved == pinned


def test_dirty_rag_repository_is_rejected(tmp_path: Path) -> None:
    repository = tmp_path / "rag"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Test"],
        check=True,
    )
    _write_rag_artifacts(repository, "committed")
    pinned = _commit(repository, "artifacts")
    (repository / "codex.csv").write_text(
        "text,source\nmodified,s\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="local changes"):
        ensure_rag_repository(repository, revision=pinned)
