from pathlib import Path

from LCD.shared.inference import (
    ModelPrediction,
    aggregate_pair_labels,
    format_model_premise,
    run_document_inference,
)
from LCD.shared.retrieval import Citation, RetrievalRecord


class FakeRetriever:
    def retrieve(self, hypothesis: str, *, top_k: int):
        assert top_k <= 20
        detected = (Citation("КоАП РФ", "32.9", "1", None),)
        return [
            RetrievalRecord(
                "p1", "source 1", "exact", 1, None, detected[0], detected
            ),
            RetrievalRecord(
                "p2", "source 2", "faiss", 2, None, detected[0], detected
            ),
        ]


class TrackingRetriever(FakeRetriever):
    def __init__(self) -> None:
        self.hypotheses = []

    def retrieve(self, hypothesis: str, *, top_k: int):
        self.hypotheses.append(hypothesis)
        return super().retrieve(hypothesis, top_k=top_k)


class FakePredictor:
    def __init__(self) -> None:
        self.premises = []

    def predict_pairs(self, premises, hypothesis):
        self.premises = list(premises)
        return [
            ModelPrediction("not mentioned", "not mentioned"),
            ModelPrediction("contradiction", "contradiction"),
        ]


def test_aggregate_pair_labels() -> None:
    assert aggregate_pair_labels(["entailment", "contradiction"], "ternary") == "contradiction"
    assert aggregate_pair_labels([None], "ternary") == "invalid"
    assert aggregate_pair_labels(["not mentioned", "entailment"], "ternary") == "entailment"


def test_format_model_premise_adds_source_once() -> None:
    assert format_model_premise("Provision body.", "КоАП РФ: Статья 20.20.") == (
        "КоАП РФ: Статья 20.20. Provision body."
    )
    assert format_model_premise(
        "КоАП РФ: Статья 20.20. Provision body.", "КоАП РФ: Статья 20.20."
    ) == "КоАП РФ: Статья 20.20. Provision body."
    assert format_model_premise(" Provision body. ", "") == "Provision body."


def test_document_inference_preserves_contradiction_premise(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "LCD.shared.inference.read_docx_text", lambda path: "ПОСТАНОВИЛ: sentence"
    )
    monkeypatch.setattr(
        "LCD.shared.inference.split_russian_sentences", lambda text: ["sentence"]
    )

    document_path = tmp_path / "decision.docx"
    document_path.write_bytes(b"stable document")
    predictor = FakePredictor()
    tables = run_document_inference(
        [document_path],
        predictor=predictor,
        retriever=FakeRetriever(),
        model_id="bert",
        task="ternary",
    )

    assert tables.aggregates.iloc[0]["prediction"] == "contradiction"
    contradiction = tables.pairs[tables.pairs["prediction"] == "contradiction"].iloc[0]
    assert predictor.premises == ["source 1 p1", "source 2 p2"]
    assert contradiction["premise"] == "p2"
    assert contradiction["model_premise"] == "source 2 p2"
    assert contradiction["source"] == "source 2"
    assert contradiction["retrieval_method"] == "faiss"
    assert len(contradiction["document_sha256"]) == 64
    assert '"article": "32.9"' in contradiction["detected_citations"]
    assert tables.aggregates.iloc[0]["unresolved_citations"] == "[]"


def test_document_inference_can_keep_bert_premises_body_only(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "LCD.shared.inference.read_docx_text", lambda path: "ПОСТАНОВИЛ: sentence"
    )
    monkeypatch.setattr(
        "LCD.shared.inference.split_russian_sentences", lambda text: ["sentence"]
    )
    document_path = tmp_path / "decision.docx"
    document_path.write_bytes(b"stable document")
    predictor = FakePredictor()

    tables = run_document_inference(
        [document_path],
        predictor=predictor,
        retriever=FakeRetriever(),
        model_id="bert",
        task="ternary",
        include_source_prefix=False,
    )

    assert predictor.premises == ["p1", "p2"]
    assert tables.pairs["model_premise"].tolist() == ["p1", "p2"]


def test_document_inference_filters_irrelevant_sentences_before_retrieval(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "LCD.shared.inference.read_docx_text", lambda path: "ПОСТАНОВИЛ: text"
    )
    monkeypatch.setattr(
        "LCD.shared.inference.split_russian_sentences",
        lambda text: [
            "Судья Петрова.",
            "Оплатить по реквизитам.",
            "Банковские ре...изиты.",
            "Приложить квитанцию.",
            "Банк получателя, БИК 044525000.",
            "Получатель, ИНН 7700000000.",
            "Номер начисления, УИН 18810177240010001111.",
            "Код дохода, КБК 18811601181019000140.",
            "Код территории, ОКТМО 45382000.",
            "Лицевой счет, л/с 40100770005.",
            "Расчетный счет, р/с 40101810045250010041.",
            "40101810045250010041.",
            "Назначить административный штраф.",
        ],
    )
    document_path = tmp_path / "decision.docx"
    document_path.write_bytes(b"stable document")
    retriever = TrackingRetriever()

    tables = run_document_inference(
        [document_path],
        predictor=FakePredictor(),
        retriever=retriever,
        model_id="model",
        task="ternary",
    )

    assert retriever.hypotheses == ["Назначить административный штраф."]
    assert set(tables.pairs["hypothesis"]) == {"Назначить административный штраф."}
    assert set(tables.pairs["sentence_index"]) == {12}
    assert set(tables.pairs["hypothesis_id"]) == {"decision.docx:00012"}


def test_missing_operative_section_is_reported_and_skipped(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "LCD.shared.inference.read_docx_text", lambda path: "No marker"
    )
    document_path = tmp_path / "decision.docx"
    document_path.write_bytes(b"stable document")
    tables = run_document_inference(
        [document_path],
        predictor=FakePredictor(),
        retriever=FakeRetriever(),
        model_id="model",
        task="ternary",
    )
    assert tables.pairs.empty
    assert "not found" in tables.errors.iloc[0]["error"]
