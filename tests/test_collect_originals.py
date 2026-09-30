from __future__ import annotations

from zipfile import ZipFile

from LCD.experiments.data_creation.collect_originals import (
    ParsedDecision,
    article_pattern,
    parse_decision,
    parse_search_results,
    validate_original,
    write_docx,
)


def ruling_text(article: str = "18.8") -> tuple[str, ...]:
    filler = "Исследованы письменные доказательства по делу. " * 35
    return (
        "ПОСТАНОВЛЕНИЕ",
        "Судья районного суда, рассмотрев дело об административном правонарушении, "
        f"предусмотренном частью 1 статьи {article} Кодекса Российской Федерации "
        "об административных правонарушениях, в отношении гражданина Иванова.",
        "УСТАНОВИЛ:",
        filler,
        f"Суд квалифицирует действия по части 1 статьи {article} КоАП РФ.",
        "Руководствуясь статьей 29.10 КоАП РФ, судья ПОСТАНОВИЛ:",
        f"Признать Иванова И.И. виновным в совершении административного "
        f"правонарушения, предусмотренного частью 1 статьи {article} КоАП РФ, "
        "и назначить административное наказание в виде штрафа в размере 2000 рублей.",
        "Постановление может быть обжаловано в краевой суд в течение десяти суток.",
        "Судья А.А. Иванова",
    )


def test_parse_and_validate_complete_original_ruling():
    body = "<br>".join(ruling_text())
    page = (
        '<h1>Постановление № 5-12/2024 по делу № 5-12/2024</h1>'
        '<div class="b-justice"><a>Районный суд</a> - Административные правонарушения</div>'
        f'<hr class="hr-h1">{body}<!--AdFox START-->'
    )
    parsed = parse_decision(page)
    assert parsed is not None
    assert parsed.court.startswith("Районный суд")
    result = validate_original(parsed, "18.8")
    assert result.accepted
    assert "operative_citation=" in result.evidence


def test_accepts_decision_title_when_body_is_original_ruling():
    parsed = ParsedDecision(
        "Решение от 4 сентября 2024 г.", "Мировой судья", ruling_text("6.9"),
    )
    result = validate_original(parsed, "6.9")
    assert result.accepted
    assert result.evidence.startswith("first_instance_verified_from_body_and_operative")


def test_rejects_complaint_review_despite_appeal_right_notice():
    paragraphs = list(ruling_text())
    paragraphs[1] = (
        "Судья районного суда, рассмотрев жалобу Иванова на постановление мирового "
        "судьи по делу об административном правонарушении, предусмотренном частью "
        "1 статьи 18.8 КоАП РФ."
    )
    paragraphs[-3] = (
        "Постановление мирового судьи оставить без изменения, жалобу без удовлетворения."
    )
    result = validate_original(
        ParsedDecision("Постановление № 5-1/2024", "Суд", tuple(paragraphs)), "18.8",
    )
    assert not result.accepted
    assert result.reason == "complaint_or_protest_review_opening"


def test_requires_target_article_in_final_operative():
    paragraphs = list(ruling_text("18.20"))
    paragraphs[4] += " Ранее лицо привлекалось по статье 18.20 КоАП РФ."
    paragraphs[-3] = (
        "Признать Иванова И.И. виновным в совершении административного "
        "правонарушения, предусмотренного частью 1 статьи 18.8 КоАП РФ, и назначить "
        "административное наказание в виде штрафа в размере 2000 рублей."
    )
    result = validate_original(
        ParsedDecision("Постановление № 5-1/2024", "Суд", tuple(paragraphs)), "18.20",
    )
    assert not result.accepted
    assert result.reason == "target_article_absent_from_operative"


def test_article_match_has_numeric_boundary():
    pattern = article_pattern("18.8")
    assert pattern.search("статья 18.8 КоАП РФ")
    assert not pattern.search("статья 18.18 КоАП РФ")


def test_search_parser_keeps_only_original_case_titles():
    content = """
    <h4><a href="/regular/doc/AAA111/?regular-txt=18.8&amp;page=1#snippet">Постановление № 5-12/2024 по делу № 5-12/2024</a></h4>
    <h4><a href="/magistrate/doc/DDD444/?magistrate-txt=18.8#snippet">Постановление № 5-22/2024 по делу № 5-22/2024</a></h4>
    <h4><a href="/regular/doc/BBB222/">Решение № 12-34/2024</a></h4>
    <h4><a href="/regular/doc/CCC333/">Постановление № 12-9/2024</a></h4>
    """
    rows = parse_search_results(content, "18.8", "main", 2024)
    assert [row.document_id for row in rows] == ["AAA111", "DDD444"]
    assert [row.corpus for row in rows] == ["regular", "magistrate"]


def test_docx_writer_creates_readable_ooxml(tmp_path):
    path = tmp_path / "ruling.docx"
    write_docx(path, ("ПОСТАНОВИЛ:", "Назначить штраф."))
    with ZipFile(path) as archive:
        document = archive.read("word/document.xml").decode("utf-8")
    assert "ПОСТАНОВИЛ:" in document
    assert "Назначить штраф." in document
