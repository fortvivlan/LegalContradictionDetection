"""Small, offline checks for the isolated Full-document classifier workflow."""

import csv
import sys
from pathlib import Path

import pytest
from openpyxl import Workbook

from LCD.experiments.classifiers.series_01_qwen3_full_document import data, training
from LCD.shared.prompting import build_generation_prompt, build_training_texts


class CharTokenizer:
    chat_template = None

    def __call__(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": [ord(character) for character in text]}


def workbook(path: Path, *, label="entailment") -> None:
    book = Workbook()
    sheet = book.active
    sheet.append(["hypothesis", "premise", "article_number", "model_prediction", "expert_label"])
    sheet.append(["Гипотеза", "Предпосылка", "КоАП Статья 1", "contradiction", label])
    book.save(path)


def test_matching_and_row_provenance(tmp_path, monkeypatch):
    xlsx = tmp_path / "xlsx"
    docx = tmp_path / "docx"
    xlsx.mkdir()
    docx.mkdir()
    source = xlsx / "Тест_Алимов_ternary_model_predictions.xlsx"
    document = docx / "Тест_Алимов.docx"
    workbook(source)
    document.touch()
    monkeypatch.setattr(data, "read_docx_text", lambda path: "Судебное решение")
    csv_path, manifest_path = data.export_full(xlsx, docx, tmp_path / "out")
    with csv_path.open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["workbook"] == source.name
    assert row["document"] == document.name
    assert row["workbook_row"] == "2"
    assert row["article_number"] == "КоАП Статья 1"
    assert row["expert_label"] == "entailment"
    assert row["document_context"] == "Судебное решение"
    assert '"row_count": 1' in manifest_path.read_text(encoding="utf-8")


def test_missing_match_and_invalid_expert_label(tmp_path, monkeypatch):
    xlsx, docx = tmp_path / "xlsx", tmp_path / "docx"
    xlsx.mkdir()
    docx.mkdir()
    source = xlsx / "Алимов.xlsx"
    workbook(source, label="bad label")
    with pytest.raises(ValueError, match="nonempty"):
        data.match_sources(xlsx, docx)
    document = docx / "Алимов.docx"
    document.touch()
    monkeypatch.setattr(data, "read_docx_text", lambda path: "Решение")
    with pytest.raises(ValueError, match="invalid expert_label"):
        data.read_reviewed_rows(source, document)


def test_context_prompt_and_response_only_context_trim():
    tokenizer = CharTokenizer()
    legacy = build_generation_prompt(tokenizer, "P", "H", "ternary")
    contextual = build_generation_prompt(tokenizer, "P", "H", "ternary",
                                         document_context="ABC")
    assert "Контекст: ABC\nПредпосылка: P\nГипотеза: H" in contextual
    assert "Контекст:" not in legacy
    row = {"hypothesis": "H", "premise": "P", "article_number": "C",
           "expert_label": "entailment", "document_context": "ABCDEFGHIJ"}
    complete, _ = training.tokenize_example(tokenizer, row, 10000)
    trimmed, retained = training.tokenize_example(tokenizer, row,
                                                    len(complete["input_ids"]) - 4)
    text = "".join(map(chr, trimmed["input_ids"]))
    supervised = "".join(chr(value) for value in trimmed["labels"] if value != -100)
    assert retained < 10
    assert f"Контекст: {'ABCDEFGHIJ'[-retained:]}\n" in text
    assert "Предпосылка: C\nP\nГипотеза: H" in text
    assert supervised.strip() == "entailment"
    assert len(trimmed["input_ids"]) <= len(complete["input_ids"]) - 4
    assert all(value == -100 for value in trimmed["labels"][:-len(supervised)])
    prompt, full = build_training_texts(tokenizer, "P", "H", "entailment", "ternary")
    assert "Контекст:" not in prompt + full


def test_noncontext_overflow_is_rejected():
    row = {"hypothesis": "H", "premise": "P", "article_number": "C",
           "expert_label": "entailment", "document_context": "ABC"}
    with pytest.raises(ValueError, match="Non-context fields"):
        training.tokenize_example(CharTokenizer(), row, 10)


def test_disposable_probe_has_exact_length_and_dummy_response():
    tokenizer = CharTokenizer()
    row = {"hypothesis": "H", "premise": "P", "article_number": "C",
           "expert_label": "contradiction", "document_context": "source text"}
    base, _ = training.tokenize_example(tokenizer, dict(row, expert_label="not mentioned"),
                                        10000, context="")
    target = len(base["input_ids"]) + 30
    example = training.exact_probe_example(tokenizer, row, target)
    assert len(example["input_ids"]) == target
    response = "".join(chr(value) for value in example["labels"] if value != -100)
    assert response.strip() == "not mentioned"
    assert "source text" in "".join(map(chr, example["input_ids"]))


def test_revision_placeholder_is_rejected_before_loading_model(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["training", "probe", "--revision", "REVISION"])
    with pytest.raises(SystemExit) as error:
        training.main()
    assert error.value.code == 2
    assert "REVISION is a placeholder" in capsys.readouterr().err


@pytest.mark.parametrize("response_tokens", [1, 3])
def test_selective_loss_matches_full_qwen_loss_and_gradients(response_tokens):
    """Compare with the old model(labels=...) path on a tiny random Qwen."""
    torch = pytest.importorskip("torch")
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(7)
    config = Qwen3Config(vocab_size=37, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8,
                         max_position_embeddings=64, attention_dropout=0.0,
                         use_cache=False)
    full_model = Qwen3ForCausalLM(config).eval()
    selective_model = Qwen3ForCausalLM(config).eval()
    selective_model.load_state_dict(full_model.state_dict())
    ids = torch.tensor([[1, 4, 9, 12, 6, 3, 8]])
    labels = torch.full_like(ids, -100)
    labels[:, -response_tokens:] = ids[:, -response_tokens:]
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
             "labels": labels}

    full_loss = full_model(**batch).loss
    selective_loss = training.selective_response_loss(selective_model, batch)
    torch.testing.assert_close(selective_loss, full_loss, atol=1e-6, rtol=1e-6)
    full_loss.backward()
    selective_loss.backward()
    for (name, full_parameter), (other_name, selective_parameter) in zip(
            full_model.named_parameters(), selective_model.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(selective_parameter.grad, full_parameter.grad,
                                   atol=1e-6, rtol=1e-5)


def test_selective_loss_rejects_non_suffix_labels():
    torch = pytest.importorskip("torch")
    labels = torch.tensor([[-100, 2, -100, 4]])
    batch = {"input_ids": torch.tensor([[1, 2, 3, 4]]), "labels": labels}
    with pytest.raises(ValueError, match="contiguous suffix"):
        training.selective_response_loss(None, batch)
