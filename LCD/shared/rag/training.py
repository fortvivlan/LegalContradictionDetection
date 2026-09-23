"""Complete legal sentence-embedding fine-tuning workflow."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from LCD.shared.common import REPOSITORY_ROOT, DEFAULT_RAG_DIR
from typing import Any, Mapping, Sequence

from ..local_support import get_huggingface_token
from ..common import (
    DEFAULT_BERT_REVISION,
    configure_reproducibility,
    file_sha256,
    merge_parameters,
    reproducibility_metadata,
    resolve_huggingface_revision,
)
from ..retrieval import ensure_rag_repository
from .artifacts import load_rag_bundle, write_rag_manifest
from .data import convert_embedding_dataset, dataframe_sha256
from .evaluation import (
    DEFAULT_RETRIEVAL_DEPTHS,
    _validate_retrieval_depths,
    run_rag_evaluation,
)
from .indexing import build_faiss_index
from .reranker_training import train_reranker
from .reranking import CrossEncoderReranker

DEFAULT_RERANKER_MODEL = "Alibaba-NLP/gte-multilingual-reranker-base"

DEFAULT_RAG_TRAINING_PARAMETERS: dict[str, Any] = {
    "max_seq_length": 512,
    "margin": 0.5,
    "epochs": 4,
    "learning_rate": 2e-5,
    "batch_size": 8,
    "eval_batch_size": 16,
    "gradient_accumulation_steps": 4,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "precision": "auto",
    "gradient_checkpointing": True,
    "num_workers": 2,
    "seed": 42,
    "deterministic": True,
    "embedding_device": "cuda",
    "index_device": "cpu",
    "index_batch_size": 32,
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "reranker_max_length": 1024,
    "reranker_batch_size": 8,
    "reranker_eval_batch_size": 16,
    "reranker_gradient_accumulation_steps": 4,
    "reranker_epochs": 3,
    "reranker_learning_rate": 2e-5,
    "reranker_warmup_ratio": 0.1,
    "reranker_weight_decay": 0.01,
    "reranker_precision": "auto",
    "reranker_gradient_checkpointing": True,
    "reranker_device": "cuda",
}


def _completed_run_configuration(
    artifacts: Path, reranker_mode: str
) -> tuple[Path, dict[str, Any]] | None:
    """Return completed experiment metadata when requested models are present."""
    manifest = artifacts / "rag_manifest.json"
    config_path = artifacts / "run_config.json"
    if not manifest.is_file() or not config_path.is_file():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    if reranker_mode == "finetuned":
        reranker = manifest_value.get("reranker") or {}
        if reranker.get("mode") != "finetuned" or not reranker.get("local"):
            return None
    return manifest, config


def _make_evaluation_rerankers(
    *,
    reranker_model_id: str,
    reranker_revision: str | None,
    reranker_trust_remote_code: bool | None,
    finetuned_model: str | None,
    parameters: Mapping[str, Any],
):
    token = get_huggingface_token()
    local = Path(reranker_model_id).expanduser().exists()
    resolved_revision = (
        None
        if local
        else resolve_huggingface_revision(
            reranker_model_id, reranker_revision, token=token
        )
    )
    trust_remote_code = (
        reranker_model_id == DEFAULT_RERANKER_MODEL
        if reranker_trust_remote_code is None
        else bool(reranker_trust_remote_code)
    )
    common = {
        "token": token,
        "trust_remote_code": trust_remote_code,
        "device": str(parameters["reranker_device"]),
        "precision": str(parameters["reranker_precision"]),
        "batch_size": int(parameters["reranker_eval_batch_size"]),
        "max_length": int(parameters["reranker_max_length"]),
    }
    pretrained = CrossEncoderReranker(
        reranker_model_id,
        revision=resolved_revision,
        **common,
    )
    finetuned = (
        CrossEncoderReranker(finetuned_model, revision=None, **common)
        if finetuned_model is not None
        else None
    )
    return pretrained, finetuned, resolved_revision, trust_remote_code, local, token


def _train_encoder(train_frame, validation_frame, *, model_id: str, revision: str, output_dir: Path, parameters):
    import torch
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        losses,
    )
    from sentence_transformers.evaluation import BinaryClassificationEvaluator

    model = SentenceTransformer(model_id, revision=revision, token=get_huggingface_token())
    model.max_seq_length = int(parameters["max_seq_length"])
    train_dataset = Dataset.from_dict(
        {
            "sentence_A": train_frame["premise"].tolist(),
            "sentence_B": train_frame["hypothesis"].tolist(),
            "label": train_frame["label"].astype(float).tolist(),
        }
    )
    evaluator = BinaryClassificationEvaluator(
        validation_frame["premise"].tolist(),
        validation_frame["hypothesis"].tolist(),
        validation_frame["label"].tolist(),
        name="validation",
        show_progress_bar=True,
    )
    precision = str(parameters["precision"]).lower()
    if precision not in {"auto", "float16", "bfloat16", "float32"}:
        raise ValueError("precision must be auto, float16, bfloat16, or float32")
    bf16 = precision == "bfloat16" or (
        precision == "auto" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    )
    fp16 = precision == "float16" or (
        precision == "auto" and torch.cuda.is_available() and not bf16
    )
    trainer_dir = output_dir / "trainer"
    arguments = SentenceTransformerTrainingArguments(
        output_dir=str(trainer_dir),
        num_train_epochs=float(parameters["epochs"]),
        per_device_train_batch_size=int(parameters["batch_size"]),
        per_device_eval_batch_size=int(parameters["eval_batch_size"]),
        gradient_accumulation_steps=int(parameters["gradient_accumulation_steps"]),
        learning_rate=float(parameters["learning_rate"]),
        warmup_ratio=float(parameters["warmup_ratio"]),
        weight_decay=float(parameters["weight_decay"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_validation_cosine_ap",
        greater_is_better=True,
        save_total_limit=2,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=bool(parameters["gradient_checkpointing"]),
        dataloader_num_workers=int(parameters["num_workers"]),
        seed=int(parameters["seed"]),
        data_seed=int(parameters["seed"]),
        full_determinism=bool(parameters["deterministic"]),
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        loss=losses.ContrastiveLoss(model, margin=float(parameters["margin"])),
        evaluator=evaluator,
    )
    result = trainer.train()
    model_dir = output_dir / "embedding_model"
    model.save_pretrained(str(model_dir), safe_serialization=True)
    metrics = evaluator(model, output_path=str(output_dir))
    import numpy as np
    import pandas as pd
    from sklearn.metrics import precision_recall_curve

    premise_vectors = model.encode(
        validation_frame["premise"].tolist(),
        batch_size=int(parameters["eval_batch_size"]),
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    hypothesis_vectors = model.encode(
        validation_frame["hypothesis"].tolist(),
        batch_size=int(parameters["eval_batch_size"]),
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    similarities = np.sum(premise_vectors * hypothesis_vectors, axis=1)
    precision_values, recall_values, thresholds = precision_recall_curve(
        validation_frame["label"].tolist(), similarities
    )
    f1_values = 2 * precision_values * recall_values / np.maximum(
        precision_values + recall_values, 1e-12
    )
    threshold_index = int(np.argmax(f1_values[:-1])) if len(thresholds) else 0
    threshold = float(thresholds[threshold_index]) if len(thresholds) else 0.5
    metrics["validation_selected_threshold"] = threshold
    metrics["validation_selected_f1"] = float(f1_values[threshold_index])
    validation_predictions = validation_frame.copy()
    validation_predictions["cosine_similarity"] = similarities
    validation_predictions["prediction"] = (similarities >= threshold).astype(int)
    validation_predictions["correct"] = (
        validation_predictions["prediction"] == validation_predictions["label"]
    )
    validation_predictions.to_csv(
        output_dir / "validation_predictions.csv", index=False, encoding="utf-8"
    )
    history = trainer.state.log_history
    return model_dir, metrics, history, result.metrics


def run_rag_experiment(
    *,
    experiment_id: str = "sbert_legal_v1",
    train_path: str | Path = REPOSITORY_ROOT / "local/data/classification/train.xlsx",
    val_path: str | Path = REPOSITORY_ROOT / "local/data/classification/val.xlsx",
    rag_dir: str | Path = DEFAULT_RAG_DIR,
    rag_revision: str = "main",
    rag_test_dir: str | Path = REPOSITORY_ROOT / "local/data/benchmarks/rag_tests",
    dialogue_workbook: str | Path | None = None,
    full_workbook: str | Path | None = None,
    full_additional_workbook: str | Path | None = None,
    test_docx_dir: str | Path = REPOSITORY_ROOT / "local/data/benchmarks/test_docx",
    model_id: str = "ai-forever/sbert_large_nlu_ru",
    revision: str | None = DEFAULT_BERT_REVISION,
    retrieval_depths: Sequence[tuple[int, int]] = DEFAULT_RETRIEVAL_DEPTHS,
    reranker_mode: str = "pretrained",
    reranker_model_id: str = DEFAULT_RERANKER_MODEL,
    reranker_revision: str | None = None,
    reranker_trust_remote_code: bool | None = None,
    hyperparameters: Mapping[str, Any] | None = None,
    artifact_root: str | Path = REPOSITORY_ROOT / "local/experiments/lawbook/series_01_embedding_reranking/artifacts",
    results_root: str | Path = REPOSITORY_ROOT / "local/experiments/lawbook/series_01_embedding_reranking/results",
):
    """Train missing retrieval models or reevaluate a completed experiment."""
    parameters = merge_parameters(DEFAULT_RAG_TRAINING_PARAMETERS, hyperparameters)
    depths = _validate_retrieval_depths(retrieval_depths)
    normalized_reranker_mode = reranker_mode.strip().lower()
    if normalized_reranker_mode not in {"pretrained", "finetuned"}:
        raise ValueError("reranker_mode must be pretrained or finetuned")
    configure_reproducibility(
        int(parameters["seed"]), deterministic=bool(parameters["deterministic"])
    )
    artifacts = Path(artifact_root) / experiment_id
    results = Path(results_root) / experiment_id
    artifacts.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    completed = _completed_run_configuration(artifacts, normalized_reranker_mode)
    if completed is not None:
        manifest, stored_config = completed
        stored_model = str(stored_config.get("model_id", ""))
        if stored_model != model_id:
            raise ValueError(
                f"Completed experiment uses model {stored_model!r}, not {model_id!r}"
            )
        if normalized_reranker_mode == "finetuned":
            stored_base = str(
                (stored_config.get("reranker") or {}).get("base_model_id", "")
            )
            if stored_base and stored_base != reranker_model_id:
                raise ValueError(
                    "Completed experiment uses reranker base model "
                    f"{stored_base!r}, not {reranker_model_id!r}"
                )
        stored_rag_commit = str(stored_config.get("rag_commit", ""))
        if not stored_rag_commit:
            raise ValueError("Completed experiment lacks its pinned rag_commit")
        rag_path, _ = ensure_rag_repository(rag_dir, revision=stored_rag_commit)
        bundle = load_rag_bundle(manifest)
        finetuned_model = (
            bundle.reranker.model
            if normalized_reranker_mode == "finetuned"
            and bundle.reranker is not None
            else None
        )
        pretrained_reranker, finetuned_reranker, *_ = _make_evaluation_rerankers(
            reranker_model_id=reranker_model_id,
            reranker_revision=reranker_revision,
            reranker_trust_remote_code=reranker_trust_remote_code,
            finetuned_model=finetuned_model,
            parameters=parameters,
        )
        print(
            f"Complete artifacts found in {artifacts}; skipping training and "
            "index rebuilding, and recalculating Recall only."
        )
        return run_rag_evaluation(
            [rag_path, manifest],
            rag_test_dir=rag_test_dir,
            dialogue_workbook=dialogue_workbook,
            full_workbook=full_workbook,
            full_additional_workbook=full_additional_workbook,
            test_docx_dir=test_docx_dir,
            results_dir=results,
            embedding_device=str(parameters["embedding_device"]),
            retrieval_depths=depths,
            pretrained_reranker=pretrained_reranker,
            finetuned_reranker=finetuned_reranker,
        )
    train = convert_embedding_dataset(train_path, "train")
    validation = convert_embedding_dataset(val_path, "validation")
    train.to_csv(results / "train_embedding.csv", index=False, encoding="utf-8")
    validation.to_csv(results / "val_embedding.csv", index=False, encoding="utf-8")
    rag_path, rag_commit = ensure_rag_repository(rag_dir, revision=rag_revision)
    token = get_huggingface_token()
    resolved_revision = resolve_huggingface_revision(model_id, revision, token=token)
    model_dir, validation_metrics, history, training_metrics = _train_encoder(
        train,
        validation,
        model_id=model_id,
        revision=resolved_revision,
        output_dir=artifacts,
        parameters=parameters,
    )
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ModuleNotFoundError:
        pass
    index_dir = artifacts / "faiss_index"
    index_metadata = build_faiss_index(
        rag_path / "codex.csv",
        model_dir,
        index_dir,
        device=str(parameters["index_device"]),
        batch_size=int(parameters["index_batch_size"]),
        chunk_size=int(parameters["chunk_size"]),
        chunk_overlap=int(parameters["chunk_overlap"]),
    )
    reranker_token = get_huggingface_token()
    reranker_local = Path(reranker_model_id).expanduser().exists()
    resolved_reranker_revision = (
        None
        if reranker_local
        else resolve_huggingface_revision(
            reranker_model_id, reranker_revision, token=reranker_token
        )
    )
    trust_reranker_code = (
        reranker_model_id == DEFAULT_RERANKER_MODEL
        if reranker_trust_remote_code is None
        else bool(reranker_trust_remote_code)
    )
    reranker_validation_metrics = None
    reranker_training_metrics = None
    reranker_history = None
    finetuned_reranker_model = None
    if normalized_reranker_mode == "finetuned":
        reranker_model, reranker_validation_metrics, reranker_history, reranker_training_metrics = train_reranker(
            train,
            validation,
            model_id=reranker_model_id,
            revision=resolved_reranker_revision,
            token=reranker_token,
            trust_remote_code=trust_reranker_code,
            output_dir=artifacts,
            parameters=parameters,
        )
        finetuned_reranker_model = str(reranker_model)
        reranker_manifest = {
            "mode": "finetuned",
            "model": str(Path(reranker_model).resolve().relative_to(artifacts.resolve())),
            "local": True,
            "revision": None,
            "base_model_id": reranker_model_id,
            "base_model_revision": resolved_reranker_revision,
            "trust_remote_code": trust_reranker_code,
            "max_length": parameters["reranker_max_length"],
        }
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ModuleNotFoundError:
            pass
    else:
        manifest_reranker_model = reranker_model_id
        if reranker_local:
            resolved_local_reranker = Path(reranker_model_id).expanduser().resolve()
            try:
                manifest_reranker_model = str(
                    resolved_local_reranker.relative_to(artifacts.resolve())
                )
            except ValueError:
                manifest_reranker_model = str(resolved_local_reranker)
        reranker_manifest = {
            "mode": "pretrained",
            "model": manifest_reranker_model,
            "local": reranker_local,
            "revision": resolved_reranker_revision,
            "trust_remote_code": trust_reranker_code,
            "max_length": parameters["reranker_max_length"],
        }
    pretrained_reranker = CrossEncoderReranker(
        reranker_model_id,
        revision=resolved_reranker_revision,
        token=reranker_token,
        trust_remote_code=trust_reranker_code,
        device=str(parameters["reranker_device"]),
        precision=str(parameters["reranker_precision"]),
        batch_size=int(parameters["reranker_eval_batch_size"]),
        max_length=int(parameters["reranker_max_length"]),
    )
    finetuned_reranker = (
        CrossEncoderReranker(
            finetuned_reranker_model,
            revision=None,
            token=reranker_token,
            trust_remote_code=trust_reranker_code,
            device=str(parameters["reranker_device"]),
            precision=str(parameters["reranker_precision"]),
            batch_size=int(parameters["reranker_eval_batch_size"]),
            max_length=int(parameters["reranker_max_length"]),
        )
        if finetuned_reranker_model is not None
        else None
    )
    run_metadata = {
        "model_id": model_id,
        "resolved_revision": resolved_revision,
        "rag_commit": rag_commit,
        "train_path": str(Path(train_path).resolve()),
        "validation_path": str(Path(val_path).resolve()),
        "train_file_sha256": file_sha256(train_path),
        "validation_file_sha256": file_sha256(val_path),
        "converted_train_sha256": dataframe_sha256(train),
        "converted_validation_sha256": dataframe_sha256(validation),
        "parameters": parameters,
        "validation_metrics": validation_metrics,
        "training_metrics": training_metrics,
        "training_history": history,
        "index": index_metadata,
        "retrieval_depths": [
            {"candidate_top_k": candidate, "final_top_k": final}
            for candidate, final in depths
        ],
        "evaluated_rerankers": [
            {
                "mode": "pretrained",
                "model": reranker_model_id,
                "revision": resolved_reranker_revision,
            },
            *(
                [
                    {
                        "mode": "finetuned",
                        "model": finetuned_reranker_model,
                        "base_model": reranker_model_id,
                        "base_revision": resolved_reranker_revision,
                    }
                ]
                if finetuned_reranker_model is not None
                else []
            ),
        ],
        "reranker": reranker_manifest,
        "reranker_validation_metrics": reranker_validation_metrics,
        "reranker_training_metrics": reranker_training_metrics,
        "reranker_training_history": reranker_history,
        "reproducibility": reproducibility_metadata(
            int(parameters["seed"]), deterministic=bool(parameters["deterministic"])
        ),
    }
    (artifacts / "run_config.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    manifest = write_rag_manifest(
        artifacts,
        experiment_id=experiment_id,
        codex_path=rag_path / "codex.csv",
        embedding_model_dir=model_dir,
        index_dir=index_dir,
        metadata={
            "model_id": model_id,
            "resolved_revision": resolved_revision,
            "rag_commit": rag_commit,
            "chunk_size": parameters["chunk_size"],
            "chunk_overlap": parameters["chunk_overlap"],
            "seed": parameters["seed"],
            "retrieval_depths": [list(value) for value in depths],
        },
        reranker=reranker_manifest,
    )
    return run_rag_evaluation(
        [rag_path, manifest],
        rag_test_dir=rag_test_dir,
        dialogue_workbook=dialogue_workbook,
        full_workbook=full_workbook,
        full_additional_workbook=full_additional_workbook,
        test_docx_dir=test_docx_dir,
        results_dir=results,
        embedding_device=str(parameters["embedding_device"]),
        retrieval_depths=depths,
        pretrained_reranker=pretrained_reranker,
        finetuned_reranker=finetuned_reranker,
    )
