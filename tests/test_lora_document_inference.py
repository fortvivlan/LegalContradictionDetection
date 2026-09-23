from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from LCD.shared import lora


def test_prediction_datasets_include_root_and_nested_docx(tmp_path: Path) -> None:
    (tmp_path / "root.docx").write_bytes(b"docx")
    (tmp_path / "Full").mkdir()
    (tmp_path / "Full" / "nested.docx").write_bytes(b"docx")

    datasets = lora._prediction_datasets(tmp_path, selected_datasets=("Full", "root"))
    default_datasets = lora._prediction_datasets(tmp_path)

    assert [name for name, _ in default_datasets] == ["Full"]

    assert [(name, [path.name for path in paths]) for name, paths in datasets] == [
        ("Full", ["nested.docx"]),
        ("root", ["root.docx"]),
    ]


def test_run_lora_document_inference_writes_persistent_workbooks(
    monkeypatch, tmp_path: Path
) -> None:
    test_docx = tmp_path / "test_docx"
    (test_docx / "Dialogue").mkdir(parents=True)
    (test_docx / "Full").mkdir()
    (test_docx / "Dialogue" / "same.docx").write_bytes(b"dialogue")
    (test_docx / "Full" / "same.docx").write_bytes(b"full")
    manifest = {
        "hyperparameters": {},
        "resolved_revision": "a" * 40,
    }
    monkeypatch.setattr(lora, "load_saved_artifact_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(lora, "configure_reproducibility", lambda *args, **kwargs: None)
    monkeypatch.setattr(lora, "get_huggingface_token", lambda: None)
    monkeypatch.setattr(
        lora,
        "ensure_rag_repository",
        lambda *args, **kwargs: (tmp_path / "rag", "b" * 40),
    )
    monkeypatch.setattr(
        lora,
        "_load_saved_adapter",
        lambda *args, **kwargs: (object(), object(), "float16"),
    )
    monkeypatch.setattr(lora, "CausalPredictor", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        lora.PremiseRetriever,
        "from_rag_directory",
        lambda *args, **kwargs: object(),
    )

    def fake_inference(documents, **kwargs):
        document = documents[0]
        return SimpleNamespace(
            pairs=pd.DataFrame(
                [
                    {
                        "document": document.name,
                        "task": "ternary",
                        "sentence_index": 0,
                        "hypothesis": "Sentence.",
                        "premise": f"Premise from {document.parent.name}.",
                        "source": "КоАП РФ: Статья 1.",
                        "retrieval_rank": 1,
                        "prediction": "not mentioned",
                    }
                ]
            )
        )

    monkeypatch.setattr(lora, "run_document_inference", fake_inference)

    output = lora.run_lora_document_inference(
        "ministral",
        "ternary",
        artifact_root=tmp_path / "artifacts",
        rag_dir=tmp_path / "rag",
        test_docx_dir=test_docx,
        selected_datasets=("Dialogue", "Full"),
        results_dir=tmp_path / "results",
    )

    assert (output / "Dialogue" / "same_ternary_model_predictions.xlsx").is_file()
    assert (output / "Full" / "same_ternary_model_predictions.xlsx").is_file()
