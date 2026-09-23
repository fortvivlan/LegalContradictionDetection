from LCD.shared.documents import (
    extract_operative_section,
    split_russian_sentences,
    textcheck,
)


def test_extract_operative_section_uses_final_spaced_marker() -> None:
    text = "Вводная часть. ПОСТАНОВИЛ: старое. П О С Т А Н О В И Л : Назначить штраф."
    assert extract_operative_section(text) == "Назначить штраф."


def test_extract_operative_section_returns_none_when_missing_or_empty() -> None:
    assert extract_operative_section("РЕШИЛ: отказать") is None
    assert extract_operative_section("ПОСТАНОВИЛ:") is None


def test_split_russian_sentences() -> None:
    assert split_russian_sentences("Назначить штраф. Жалоба может быть подана!") == [
        "Назначить штраф.",
        "Жалоба может быть подана!",
    ]


def test_split_russian_sentences_preserves_otd_abbreviation() -> None:
    text = "Банк получателя: Отд. № 1 Банка России, БИК 044525000. Назначить штраф."

    sentences = split_russian_sentences(text)

    assert sentences == [
        "Банк получателя: Отд. № 1 Банка России, БИК 044525000.",
        "Назначить штраф.",
    ]
    assert [sentence for sentence in sentences if not textcheck(sentence)] == [
        "Назначить штраф."
    ]


def test_textcheck_filters_signatures_and_payment_details_case_insensitively() -> None:
    assert textcheck("Судья Петрова")
    assert textcheck("Оплатить штраф по указанным РЕКВИЗИТАМ.")
    assert textcheck("Банковские ре...изиты скрыты.")
    assert textcheck("Квитанцию необходимо представить в суд.")
    assert not textcheck("Назначить административный штраф.")


def test_textcheck_filters_only_standalone_case_sensitive_bank_details() -> None:
    assert textcheck("БИК: 044525000")
    assert textcheck("ИНН 7700000000")
    assert textcheck("УИН 18810177240010001111")
    assert textcheck("КБК 18811601181019000140")
    assert textcheck("ОКТМО: 45382000")
    assert textcheck("л/с 40100770005")
    assert textcheck("р/с: 40101810045250010041")
    assert textcheck("Банк (БИК) получателя")
    assert not textcheck("бик 044525000")
    assert not textcheck("инн 7700000000")
    assert not textcheck("уин 18810177240010001111")
    assert not textcheck("кбк 18811601181019000140")
    assert not textcheck("октмо 45382000")
    assert not textcheck("Л/С 40100770005")
    assert not textcheck("Р/С 40101810045250010041")
    assert not textcheck("БИКОВСКИЙ")
    assert not textcheck("МОЙИНН")
    assert not textcheck("АУИНА")
    assert not textcheck("КБКА")
    assert not textcheck("ОКТМОВСКИЙ")
    assert not textcheck("пол/с")
    assert not textcheck("р/счет")


def test_textcheck_filters_numeric_only_sentences() -> None:
    assert textcheck("123456")
    assert textcheck("123456.")
    assert textcheck("123 456")
    assert not textcheck("Статья 12")
    assert not textcheck("12.3")
