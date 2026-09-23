"""BERT baseline training, validation, and full RAG document inference."""

from __future__ import annotations

import gc
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .local_support import (
    deliver_file,
    get_huggingface_token,
    prepare_artifact_root,
    selected_docx_files,
)
from .autotest_scoring import (
    add_test_dataset,
    discover_autotest_datasets,
    score_autotest_predictions,
)
from .common import (
    DEFAULT_AUTOTEST_DIR,
    DEFAULT_BERT_REVISION,
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_EMBEDDING_REVISION,
    DEFAULT_RAG_DIR,
    DEFAULT_RAG_REVISION,
    DEFAULT_RESULTS_DIR,
    DEFAULT_TEST_DOCX_DIR,
    LABEL2ID_BY_TASK,
    Task,
    announce_stage,
    configure_reproducibility,
    default_dataset_path,
    evaluate_predictions,
    file_sha256,
    load_saved_artifact_manifest,
    load_dataset,
    merge_parameters,
    reproducibility_metadata,
    resolve_huggingface_revision,
    saved_artifact_revision,
    seed_data_loader_worker,
)
from .inference import ModelPrediction, run_document_inference
from .reporting import (
    display_scores,
    write_document_review_package,
    write_results_workbook,
)
from .retrieval import PremiseRetriever, ensure_rag_repository

DEFAULT_BERT_MODEL = "ai-forever/sbert_large_nlu_ru"
DEFAULT_BERT_PARAMETERS: dict[str, Any] = {
    "max_length": 512,
    "batch_size": 16,
    "inference_batch_size": 32,
    "epochs": 6,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "num_workers": 2,
    "mixed_precision": True,
    "precision": "auto",
    "device": "auto",
    "gradient_checkpointing": False,
    "retrieval_top_k": 20,
    "embedding_device": "cpu",
    "embedding_revision": DEFAULT_EMBEDDING_REVISION,
    "seed": 42,
    "deterministic": True,
}


class _BertPairDataset:
    """Pickle-safe pair dataset for spawn-based DataLoader workers."""

    def __init__(self, dataframe, label2id: Mapping[str, int]):
        rows = dataframe.reset_index(drop=True)
        self.premises = rows["premise"].tolist()
        self.hypotheses = rows["hypothesis"].tolist()
        self.labels = [label2id[label] for label in rows["tag"]]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[str, str, int]:
        return self.premises[index], self.hypotheses[index], self.labels[index]


class _BertBatchCollator:
    """Tokenize pair batches in a pickle-safe DataLoader callable."""

    def __init__(self, tokenizer, *, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows):
        import torch

        premises, hypotheses, labels = zip(*rows)
        encoded = self.tokenizer(
            list(premises),
            list(hypotheses),
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded


class BertPredictor:
    """Batched sequence-classifier predictor."""

    def __init__(self, model, tokenizer, *, batch_size: int, max_length: int):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def predict_examples(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
        *,
        progress_description: str | None = None,
    ) -> list[ModelPrediction]:
        if len(premises) != len(hypotheses):
            raise ValueError("Premises and hypotheses must have equal length")
        import torch

        predictions: list[ModelPrediction] = []
        batch_starts = range(0, len(premises), self.batch_size)
        if progress_description is not None:
            from tqdm.auto import tqdm

            batch_starts = tqdm(
                batch_starts,
                desc=progress_description,
                unit="batch",
            )
        for start in batch_starts:
            encoded = self.tokenizer(
                list(premises[start : start + self.batch_size]),
                list(hypotheses[start : start + self.batch_size]),
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.model.device)
            with torch.no_grad():
                ids = self.model(**encoded).logits.argmax(dim=-1).cpu().tolist()
            predictions.extend(
                ModelPrediction(self.model.config.id2label[int(label_id)])
                for label_id in ids
            )
        return predictions

    def predict_pairs(
        self, premises: Sequence[str], hypothesis: str
    ) -> list[ModelPrediction]:
        return self.predict_examples(premises, [hypothesis] * len(premises))


def _bert_device(parameters: Mapping[str, Any]):
    import torch

    requested = str(parameters["device"]).lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested but is not available")
    return device


def _load_bert_artifact(
    artifact_dir: Path,
    *,
    task: Task,
    parameters: Mapping[str, Any],
):
    """Load a complete locally saved BERT classifier without network access."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = _bert_device(parameters)
    tokenizer = AutoTokenizer.from_pretrained(artifact_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        artifact_dir, local_files_only=True
    ).to(device)
    saved_labels = set(model.config.id2label.values())
    expected_labels = set(LABEL2ID_BY_TASK[task])
    if saved_labels != expected_labels:
        raise ValueError(
            f"Saved BERT labels {sorted(saved_labels)} do not match {task} labels "
            f"{sorted(expected_labels)}"
        )
    model.eval()
    compute_dtype = str(next(model.parameters()).dtype).removeprefix("torch.")
    return model, tokenizer, compute_dtype


def _train_bert(
    train_dataframe,
    val_dataframe,
    *,
    task: Task,
    model_id: str,
    revision: str,
    token: str | None,
    parameters: Mapping[str, Any],
):
    import torch
    from sklearn.metrics import f1_score
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    device = _bert_device(parameters)
    label2id = LABEL2ID_BY_TASK[task]
    id2label = {index: label for label, index in label2id.items()}
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, token=token
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        revision=revision,
        token=token,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    ).to(device)
    if bool(parameters["gradient_checkpointing"]):
        model.gradient_checkpointing_enable()

    generator = torch.Generator().manual_seed(int(parameters["seed"]))
    collator = _BertBatchCollator(
        tokenizer, max_length=int(parameters["max_length"])
    )
    train_loader = DataLoader(
        _BertPairDataset(train_dataframe, label2id),
        batch_size=int(parameters["batch_size"]),
        shuffle=True,
        collate_fn=collator,
        num_workers=int(parameters["num_workers"]),
        pin_memory=device.type == "cuda",
        generator=generator,
        worker_init_fn=seed_data_loader_worker,
    )
    val_loader = DataLoader(
        _BertPairDataset(val_dataframe, label2id),
        batch_size=int(parameters["inference_batch_size"]),
        shuffle=False,
        collate_fn=collator,
        num_workers=int(parameters["num_workers"]),
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_data_loader_worker,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    total_steps = len(train_loader) * int(parameters["epochs"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * float(parameters["warmup_ratio"])),
        num_training_steps=total_steps,
    )
    use_precision = bool(parameters["mixed_precision"]) and device.type == "cuda"
    requested_precision = str(parameters["precision"]).lower()
    if requested_precision not in {"auto", "bfloat16", "float16"}:
        raise ValueError("precision must be 'auto', 'bfloat16', or 'float16'")
    use_bf16 = requested_precision == "bfloat16" or (
        requested_precision == "auto"
        and device.type == "cuda"
        and bool(torch.cuda.is_bf16_supported())
    )
    if use_bf16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("bfloat16 was requested but is unsupported by this GPU")
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_precision and not use_bf16
    )
    history: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="lcd_bert_best_") as temp:
        best_path = Path(temp) / "best_state.pt"
        best_f1 = -1.0
        for epoch in range(1, int(parameters["epochs"]) + 1):
            model.train()
            total_loss = 0.0
            for batch in tqdm(train_loader, desc=f"BERT {task} epoch {epoch}"):
                batch = {
                    key: value.to(device, non_blocking=device.type == "cuda")
                    for key, value in batch.items()
                }
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, dtype=autocast_dtype, enabled=use_precision
                ):
                    output = model(**batch)
                    loss = output.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(parameters["max_grad_norm"])
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                total_loss += float(loss.item())

            model.eval()
            gold_ids: list[int] = []
            predicted_ids: list[int] = []
            with torch.no_grad():
                for batch in val_loader:
                    gold_ids.extend(batch["labels"].tolist())
                    gpu_batch = {
                        key: value.to(device, non_blocking=device.type == "cuda")
                        for key, value in batch.items()
                        if key != "labels"
                    }
                    predicted_ids.extend(
                        model(**gpu_batch).logits.argmax(dim=-1).cpu().tolist()
                    )
            macro_f1 = f1_score(gold_ids, predicted_ids, average="macro")
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": total_loss / max(1, len(train_loader)),
                    "validation_macro_f1": macro_f1,
                }
            )
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                torch.save(model.state_dict(), best_path)
        try:
            state = torch.load(best_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(best_path, map_location=device)
        model.load_state_dict(state)
    model.eval()
    compute_dtype = "float32" if not use_precision else ("bfloat16" if use_bf16 else "float16")
    return model, tokenizer, history, compute_dtype


def _run_bert(
    task: Task,
    *,
    train_path: str | Path | None,
    val_path: str | Path | None,
    rag_dir: str | Path,
    rag_revision: str,
    artifact_root: str | Path,
    model_id: str,
    revision: str | None,
    use_existing_model: bool,
    hyperparameters: Mapping[str, Any] | None,
    document_paths: Sequence[str | Path] | None,
    autotest_dir: str | Path,
    test_docx_dir: str | Path,
    score_autotest: bool,
    multiple_test: bool,
    results_dir: str | Path,
):
    workflow = f"bert/{task}"
    announce_stage(workflow, "setup", "Preparing datasets and artifact paths.")
    train_path = Path(train_path or default_dataset_path("train", task))
    val_path = Path(val_path or default_dataset_path("val", task))
    train_hash = file_sha256(train_path)
    val_hash = file_sha256(val_path)
    project_drive = prepare_artifact_root(artifact_root)
    model_target = project_drive / "models" / "bert" / task
    artifact_manifest: dict[str, Any] | None = None
    if use_existing_model:
        announce_stage(
            workflow,
            "reuse",
            f"Checking the previously trained model at {model_target}.",
        )
        artifact_manifest = load_saved_artifact_manifest(
            model_target,
            required_files=("run_config.json", "config.json", "tokenizer_config.json"),
            weight_files=("model.safetensors", "pytorch_model.bin"),
            expected={
                "model_id": model_id,
                "task": task,
                "train_sha256": train_hash,
                "validation_sha256": val_hash,
            },
        )
        stored_parameters = artifact_manifest.get("hyperparameters")
        if not isinstance(stored_parameters, dict):
            raise ValueError(f"Saved BERT manifest has no hyperparameters: {model_target}")
        parameters = merge_parameters(DEFAULT_BERT_PARAMETERS, stored_parameters)
        parameters = merge_parameters(parameters, hyperparameters)
    else:
        parameters = merge_parameters(DEFAULT_BERT_PARAMETERS, hyperparameters)
    configure_reproducibility(
        int(parameters["seed"]),
        deterministic=bool(parameters["deterministic"]),
    )
    reproducibility = reproducibility_metadata(
        int(parameters["seed"]),
        deterministic=bool(parameters["deterministic"]),
    )
    val_dataframe = load_dataset(val_path, task)
    token = get_huggingface_token()
    if artifact_manifest is not None:
        effective_revision = saved_artifact_revision(artifact_manifest, model_target)
        if revision is not None:
            requested_commit = resolve_huggingface_revision(
                model_id, revision, token=token
            )
            if requested_commit != effective_revision:
                raise ValueError(
                    "Requested BERT revision does not match the saved model: "
                    f"{requested_commit} != {effective_revision}"
                )
    else:
        requested_revision = revision or (
            DEFAULT_BERT_REVISION if model_id == DEFAULT_BERT_MODEL else None
        )
        effective_revision = resolve_huggingface_revision(
            model_id, requested_revision, token=token
        )
    announce_stage(workflow, "rag", "Preparing the pinned RAG repository.")
    rag_path, rag_commit = ensure_rag_repository(
        rag_dir, revision=rag_revision
    )
    announce_stage(workflow, "rag", f"RAG repository ready at {rag_commit[:12]}.")
    model = tokenizer = None
    try:
        if artifact_manifest is not None:
            announce_stage(
                workflow,
                "load",
                "Loading the trained BERT model from the artifact directory; training is skipped.",
            )
            model, tokenizer, compute_dtype = _load_bert_artifact(
                model_target, task=task, parameters=parameters
            )
            history = artifact_manifest.get("training_history", [])
            announce_stage(workflow, "load", "Saved BERT model loaded.")
        else:
            train_dataframe = load_dataset(train_path, task)
            announce_stage(
                workflow,
                "training",
                f"Starting BERT training on {len(train_dataframe)} examples.",
            )
            model, tokenizer, history, compute_dtype = _train_bert(
                train_dataframe,
                val_dataframe,
                task=task,
                model_id=model_id,
                revision=effective_revision,
                token=token,
                parameters=parameters,
            )
            model_target.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(model_target, safe_serialization=True)
            tokenizer.save_pretrained(model_target)
            (model_target / "run_config.json").write_text(
                json.dumps(
                    {
                        "model_id": model_id,
                        "requested_revision": revision or "registry_default",
                        "resolved_revision": effective_revision,
                        "task": task,
                        "hyperparameters": parameters,
                        "training_history": history,
                        "train_sha256": train_hash,
                        "validation_sha256": val_hash,
                        "rag_requested_revision": rag_revision,
                        "rag_revision": rag_commit,
                        "reproducibility": reproducibility,
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
            announce_stage(
                workflow,
                "training",
                f"Training finished; final model saved to {model_target}.",
            )
        announce_stage(
            workflow,
            "validation",
            f"Starting validation on {len(val_dataframe)} examples.",
        )
        predictor = BertPredictor(
            model,
            tokenizer,
            batch_size=int(parameters["inference_batch_size"]),
            max_length=int(parameters["max_length"]),
        )
        validation = predictor.predict_examples(
            val_dataframe["premise"].tolist(),
            val_dataframe["hypothesis"].tolist(),
            progress_description=f"Validating BERT {task}",
        )
        evaluation = evaluate_predictions(
            val_dataframe,
            [item.label for item in validation],
            [item.raw_output for item in validation],
            model_id=model_id,
            task=task,
        )
        evaluation.predictions["model_premise"] = val_dataframe["premise"].tolist()
        announce_stage(workflow, "validation", "Validation finished.")
        announce_stage(workflow, "testing", "Loading the RAG retrieval index.")
        retriever = PremiseRetriever.from_rag_directory(
            rag_path,
            embedding_revision=str(parameters["embedding_revision"]),
            embedding_device=str(parameters["embedding_device"]),
        )
        announce_stage(workflow, "testing", "RAG retrieval index loaded.")
        announce_stage(
            workflow,
            "testing",
            "Document testing is ready. Preparing benchmark or explicit DOCX files.",
        )
        import pandas as pd

        if document_paths is None:
            datasets = discover_autotest_datasets(
                autotest_dir, test_docx_dir, multiple_test=multiple_test
            )
            dataset_inputs = [
                (
                    dataset.name,
                    dataset.autotest_dir,
                    dataset.docx_dir,
                    dataset.documents,
                )
                for dataset in datasets
            ]
        else:
            dataset_inputs = [
                ("default", Path(autotest_dir), Path(test_docx_dir), document_paths)
            ]

        inference_runs = []
        for dataset_name, review_dir, document_dir, paths in dataset_inputs:
            with selected_docx_files(paths) as documents:
                if documents:
                    announce_stage(
                        workflow,
                        "testing",
                        f"Starting {dataset_name} full RAG inference for "
                        f"{len(documents)} document(s).",
                    )
                else:
                    announce_stage(
                        workflow,
                        "testing",
                        f"No DOCX files selected for {dataset_name}; testing skipped.",
                    )
                tables = run_document_inference(
                    documents,
                    predictor=predictor,
                    retriever=retriever,
                    model_id=model_id,
                    task=task,
                    top_k=int(parameters["retrieval_top_k"]),
                    include_source_prefix=False,
                )
            inference_runs.append(
                (
                    dataset_name,
                    review_dir,
                    document_dir,
                    tuple(documents),
                    add_test_dataset(tables.aggregates, dataset_name),
                    add_test_dataset(tables.pairs, dataset_name),
                    add_test_dataset(tables.errors, dataset_name),
                )
            )
            if documents:
                announce_stage(
                    workflow,
                    "testing",
                    f"{dataset_name} full document inference finished.",
                )

        document_aggregates = pd.concat(
            [run[4] for run in inference_runs], ignore_index=True
        )
        document_pairs = pd.concat([run[5] for run in inference_runs], ignore_index=True)
        document_errors = pd.concat([run[6] for run in inference_runs], ignore_index=True)
        autotest_tables = []
        if score_autotest:
            announce_stage(
                workflow, "scoring", "Scoring fresh document pairs against autotest."
            )
            for dataset_name, review_dir, document_dir, documents, _, pairs, _ in inference_runs:
                if not documents:
                    continue
                autotest_tables.append(
                    score_autotest_predictions(
                        pairs,
                        documents,
                        model_id=model_id,
                        task=task,
                        autotest_dir=review_dir,
                        docx_dir=document_dir,
                        test_dataset=dataset_name,
                    )
                )
        combined_scores = pd.concat(
            [evaluation.scores]
            + [tables.scores for tables in autotest_tables],
            ignore_index=True,
        )
        combined_per_class = pd.concat(
            [evaluation.per_class]
            + [tables.per_class for tables in autotest_tables],
            ignore_index=True,
        )
        combined_confusion = pd.concat(
            [evaluation.confusion_matrix]
            + [tables.confusion_matrix for tables in autotest_tables],
            ignore_index=True,
        )
        announce_stage(workflow, "results", "Writing score and review artifacts.")
        workbook = write_results_workbook(
            f"bert_{task}",
            {
                "scores": combined_scores,
                "per_class": combined_per_class,
                "confusion_matrix": combined_confusion,
                "validation_predictions": evaluation.predictions,
                "training_history": pd.DataFrame(history),
                "document_aggregates": document_aggregates,
                "document_pairs": document_pairs,
                "errors": document_errors,
                "rag_summary": pd.concat(
                    [tables.rag_summary for tables in autotest_tables], ignore_index=True
                ) if autotest_tables else None,
                "autotest_alignment": pd.concat(
                    [tables.alignment for tables in autotest_tables], ignore_index=True
                ) if autotest_tables else None,
                "inferred_gold": pd.concat(
                    [tables.inferred_gold for tables in autotest_tables], ignore_index=True
                ) if autotest_tables else None,
                "autotest_excluded": pd.concat(
                    [tables.excluded for tables in autotest_tables], ignore_index=True
                ) if autotest_tables else None,
                "file_matching": pd.concat(
                    [tables.file_matching for tables in autotest_tables], ignore_index=True
                ) if autotest_tables else None,
            },
            {
                "workflow": "bert",
                "model_id": model_id,
                "requested_revision": revision or (
                    "saved_artifact" if artifact_manifest is not None else "registry_default"
                ),
                "resolved_revision": effective_revision,
                "task": task,
                "parameters": parameters,
                "train_sha256": train_hash,
                "validation_sha256": val_hash,
                "rag_requested_revision": rag_revision,
                "rag_commit": rag_commit,
                "compute_dtype": compute_dtype,
                "summary_enabled": False,
                "autotest_enabled": score_autotest,
                "multiple_test": multiple_test,
                "test_datasets": [run[0] for run in inference_runs],
                "autotest_dir": Path(autotest_dir),
                "test_docx_dir": Path(test_docx_dir),
                "drive_model_path": model_target,
                "used_existing_model": artifact_manifest is not None,
                "training_skipped": artifact_manifest is not None,
                "artifact_hyperparameters": (
                    artifact_manifest.get("hyperparameters")
                    if artifact_manifest is not None
                    else None
                ),
                "reproducibility": reproducibility,
                "bitwise_reproducibility_scope": "same code, lockfile, GPU model, driver, and CUDA stack",
            },
            output_dir=results_dir,
        )
        review_package = write_document_review_package(
            f"bert_{task}",
            document_pairs,
            output_dir=results_dir,
        )
        display_scores(combined_scores)
        deliver_file(workbook)
        if review_package is not None:
            deliver_file(review_package)
        announce_stage(workflow, "complete", "Workflow finished.")
        return combined_scores
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ModuleNotFoundError:
            pass


def run_bert_ternary(
    *,
    train_path: str | Path | None = None,
    val_path: str | Path | None = None,
    rag_dir: str | Path = DEFAULT_RAG_DIR,
    rag_revision: str = DEFAULT_RAG_REVISION,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    model_id: str = DEFAULT_BERT_MODEL,
    revision: str | None = None,
    use_existing_model: bool = False,
    hyperparameters: Mapping[str, Any] | None = None,
    document_paths: Sequence[str | Path] | None = None,
    autotest_dir: str | Path = DEFAULT_AUTOTEST_DIR,
    test_docx_dir: str | Path = DEFAULT_TEST_DOCX_DIR,
    score_autotest: bool = True,
    multiple_test: bool = False,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
):
    """Train or reuse, validate, and document-test the ternary BERT baseline.

    ``use_existing_model=True`` requires a compatible final model in the
    configured artifact root and skips training completely.
    ``multiple_test=True`` evaluates paired child folders below the autotest
    and DOCX roots as separate datasets.
    """
    return _run_bert(
        "ternary",
        train_path=train_path,
        val_path=val_path,
        rag_dir=rag_dir,
        rag_revision=rag_revision,
        artifact_root=artifact_root,
        model_id=model_id,
        revision=revision,
        use_existing_model=use_existing_model,
        hyperparameters=hyperparameters,
        document_paths=document_paths,
        autotest_dir=autotest_dir,
        test_docx_dir=test_docx_dir,
        score_autotest=score_autotest,
        multiple_test=multiple_test,
        results_dir=results_dir,
    )
