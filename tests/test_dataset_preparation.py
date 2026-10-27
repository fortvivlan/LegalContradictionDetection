from pathlib import Path

import pytest
from openpyxl import Workbook

from LCD.experiments.data_creation.series_01_nli_rag.preparation import (
    extract_hypotheses,
    quota_targets,
    read_expert_rules,
    sentence_records,
)


def test_sentence_records_respect_paragraphs_and_source_offsets() -> None:
    text = "  Первый вывод. Второй!\nБанк: Отд. № 1 Банка России.\n"
    records = sentence_records(text)

    assert [record["text"] for record in records] == [
        "Первый вывод.",
        "Второй!",
        "Банк: Отд. № 1 Банка России.",
    ]
    assert [text[record["start"] : record["end"]] for record in records] == [
        record["text"] for record in records
    ]
    assert [record["index"] for record in records] == [0, 1, 2]


def test_extract_hypotheses_filters_lines_without_dropping_substantive_identifiers() -> None:
    text = (
        "Вводная часть. ПОСТАНОВИЛ:\n"
        "ООО «Тест» (ИНН 123) признать виновным и назначить административный штраф.\n"
        "ИНН 1234567890\n"
        "Получатель платежа: УФК по области.\n"
        "Банковские реквизиты для уплаты штрафа:\n"
        "Номер счета получателя платежа: 123.\n"
        "Штраф должен быть уплачен в течение 60 дней.\n"
        "Постановление может быть обжаловано в течение 10 суток.\n"
        "<данные изъяты>\n"
        "Судья И.И. Иванов\n"
    )

    kept, audit = extract_hypotheses(text)

    assert [record["text"] for record in kept] == [
        "ООО «Тест» (ИНН 123) признать виновным и назначить административный штраф.",
        "Штраф должен быть уплачен в течение 60 дней.",
        "Постановление может быть обжаловано в течение 10 суток.",
    ]
    assert {record["reason"] for record in audit if not record["kept"]} == {
        "bank_or_payment_details",
        "masked_only",
        "judge_signature",
    }
    assert all(text[record["start"] : record["end"]] == record["text"] for record in audit)


def test_extract_hypotheses_uses_final_marker_and_fails_closed() -> None:
    kept, _ = extract_hypotheses("ПОСТАНОВИЛ: Старое. П О С Т А Н О В И Л: Назначить штраф.")
    assert [record["text"] for record in kept] == ["Назначить штраф."]
    with pytest.raises(ValueError, match="no ПОСТАНОВИЛ"):
        extract_hypotheses("РЕШИЛ: назначить штраф")
    with pytest.raises(ValueError, match="is empty"):
        extract_hypotheses("ПОСТАНОВИЛ:")


def test_extract_hypotheses_filters_payment_titles_and_block_continuations() -> None:
    text = (
        "ПОСТАНОВИЛ:\n"
        "Назначить административный штраф в размере 5000 рублей.\n"
        "Наименование получателя платежа: УФК по Московской области\n"
        "(ОМВД России по городскому округу Серебряные Пруды Московской области)\n"
        "Наименование платежа: штраф по ст. 18.10 КоАП РФ.\n"
        "УМВД России по городскому округу Электросталь,\n"
        "Штраф подлежит уплате по следующим реквизитам: УФК по области.\n"
        "Штраф должен быть уплачен не позднее следующего дня по реквизитам УФК.\n"
        "Постановление может быть обжаловано в течение 10 суток.\n"
    )

    kept, audit = extract_hypotheses(text)

    assert [record["text"] for record in kept] == [
        "Назначить административный штраф в размере 5000 рублей.",
        "Штраф должен быть уплачен не позднее следующего дня по реквизитам УФК.",
        "Постановление может быть обжаловано в течение 10 суток.",
    ]
    removed = {record["text"]: record["reason"] for record in audit if not record["kept"]}
    assert removed["(ОМВД России по городскому округу Серебряные Пруды Московской области)"] == (
        "bank_block_continuation"
    )
    assert removed["Наименование платежа: штраф по ст. 18.10 КоАП РФ."] == (
        "bank_or_payment_details"
    )
    assert removed["УМВД России по городскому округу Электросталь,"] == (
        "bank_block_continuation"
    )
    assert removed["Штраф подлежит уплате по следующим реквизитам: УФК по области."] == (
        "bank_or_payment_details"
    )


def _expert_workbook(path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(
        [
            "Сигнал / ошибка",
            "Тип сигнала / ошибки",
            "Добавляем в текущую выборку (октябрь 2026)",
            "Процент документов от общего количества синтетических документов текущей основной и отдельной выборок",
            "Комментарий",
            "Инструкция для внесения синтетической ошибки",
            "Промпт для поиска сигналов / ошибок",
            "Пример 1",
            "Пример 2",
        ]
    )
    sheet.append(["A", "Противоречие", "Да", "7% от общей", "C", "I", "P", "E1", None])
    sheet.append(["B", "Контроль", "Да\nОтдельной выборкой", "25%", None, None, None, None, None])
    sheet.append(["C", "Ошибка", " Да ", "3,5%", None, "Edit", None, None, "E2"])
    workbook.save(path)


def test_read_expert_rules_keeps_only_exact_main_yes(tmp_path: Path) -> None:
    path = tmp_path / "rules.xlsx"
    _expert_workbook(path)

    rules = read_expert_rules(path)

    assert [rule["rule_id"] for rule in rules] == ["xlsx_row_2", "xlsx_row_4"]
    assert rules[0]["examples"] == ["E1"]
    assert rules[1]["weight_percent"] == 3.5
    assert rules[1]["examples"] == ["E2"]


def test_quota_targets_use_relative_largest_remainders_and_minimum() -> None:
    rules = [
        {"rule_id": "a", "weight_percent": 7},
        {"rule_id": "b", "weight_percent": 3},
    ]
    assert quota_targets(rules, 7) == {"a": 5, "b": 2}
    assert quota_targets(rules, 5, minimum=1) == {"a": 3, "b": 2}
    with pytest.raises(ValueError, match="smaller"):
        quota_targets(rules, 1, minimum=1)


def test_signature_continuations_and_payment_heading_are_removed() -> None:
    text = (
        "ПОСТАНОВИЛ:\nНазначить административный штраф 5000 рублей.\n"
        "Судья Чеховского Городского суда: О.Е.Мядюта\n"
        "Городского суда: Л.В. Почукаева\n"
        "Уплата административного штрафа подлежит по следующим реквизитам:\n"
        "Уплатить штраф в течение 60 дней по следующим реквизитам."
    )
    kept, audit = extract_hypotheses(text)
    assert [r["text"] for r in kept] == [
        "Назначить административный штраф 5000 рублей.",
        "Уплатить штраф в течение 60 дней по следующим реквизитам.",
    ]
    assert sum(r["reason"] == "judge_signature" for r in audit) == 2
