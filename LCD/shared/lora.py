"""QLoRA training, validation, and full RAG document-inference workflow."""

from __future__ import annotations

import gc
import json
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
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_EMBEDDING_REVISION,
    DEFAULT_RAG_DIR,
    DEFAULT_RAG_REVISION,
    DEFAULT_RESULTS_DIR,
    DEFAULT_TEST_DOCX_DIR,
    Task,
    announce_stage,
    configure_reproducibility,
    default_dataset_path,
    evaluate_predictions,
    file_sha256,
    load_saved_artifact_manifest,
    load_dataset,
    merge_parameters,
    prompt_sha256,
    reproducibility_metadata,
    resolve_huggingface_revision,
    resolve_model,
    saved_artifact_revision,
    slugify_model_id,
    validate_task,
)
from .inference import (
    SOURCE_PREFIXED_PREMISE_FORMAT,
    format_model_premise,
    run_document_inference,
)
from .llm_common import CausalPredictor, load_causal_model
from .prompting import (
    build_ministral_training_texts,
    build_training_texts,
    prompt_for_task,
)
from .reporting import (
    display_scores,
    write_document_review_package,
    write_document_review_workbooks,
    write_results_workbook,
)
from .retrieval import PremiseRetriever, ensure_rag_repository

DEFAULT_LORA_HYPERPARAMETERS: dict[str, Any] = {
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": "all-linear",
    "max_seq_length": 1024,
    "batch_size": 2,
    "gradient_accumulation_steps": 8,
    "epochs": 3,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "gradient_checkpointing": True,
    "optimizer": "auto",
    "logging_steps": 20,
    "eval_strategy": "no",
    "eval_batch_size": 8,
    "save_strategy": "no",
    "save_total_limit": None,
    "load_best_model_at_end": False,
    "num_workers": 0,
    "quantization": True,
    "device_map": "auto",
    "precision": "auto",
    "inference_batch_size": 1,
    "document_batch_size": 8,
    "max_input_length": 4096,
    "max_new_tokens": 16,
    "retrieval_top_k": 20,
    "embedding_device": "cpu",
    "embedding_revision": DEFAULT_EMBEDDING_REVISION,
    "seed": 42,
    "deterministic": True,
}

STANDARD_PROMPT_PROCESSING = "standard_chat_template_v1"
MINISTRAL_PROMPT_PROCESSING = "ministral_duplicated_system_prompt_v1"


def _prompt_processing_strategy(model_alias: str) -> str:
    return (
        MINISTRAL_PROMPT_PROCESSING
        if model_alias == "ministral"
        else STANDARD_PROMPT_PROCESSING
    )


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    length = 0
    for left_id, right_id in zip(left, right):
        if left_id != right_id:
            break
        length += 1
    return length


def _tokenize_training_rows(
    dataframe,
    tokenizer,
    task: Task,
    max_length: int,
    *,
    model_alias: str,
):
    """Tokenize SFT rows while preserving both prompt context and label tokens."""
    from tqdm.auto import tqdm

    prompt_processing = _prompt_processing_strategy(model_alias)
    build_texts = (
        build_ministral_training_texts
        if prompt_processing == MINISTRAL_PROMPT_PROCESSING
        else build_training_texts
    )
    rows: list[dict[str, list[int]]] = []
    for row in tqdm(
        dataframe.itertuples(index=False),
        total=len(dataframe),
        desc=f"Formatting {task} training data",
    ):
        model_premise = format_model_premise(row.premise, row.source)
        prompt, full = build_texts(
            tokenizer, model_premise, row.hypothesis, row.tag, task
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
        common_length = _common_prefix_length(prompt_ids, full_ids)
        response_ids = full_ids[common_length:]
        if not response_ids:
            response_ids = tokenizer(
                f" {row.tag}{tokenizer.eos_token or ''}", add_special_tokens=False
            )["input_ids"]
        response_ids = response_ids[:max_length]
        prompt_budget = max_length - len(response_ids)
        if len(prompt_ids) > prompt_budget:
            prefix_length = prompt_budget // 2
            suffix_length = prompt_budget - prefix_length
            prompt_ids = (
                prompt_ids[:prefix_length]
                + (prompt_ids[-suffix_length:] if suffix_length else [])
            )
        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + list(response_ids)
        if not any(label != -100 for label in labels):
            raise ValueError("Training example has no assistant-label tokens")
        rows.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            }
        )
    return rows


class _TokenizedRowsDataset:
    """Minimal picklable dataset wrapper for pre-tokenized LoRA examples."""

    def __init__(self, rows: Sequence[Mapping[str, Sequence[int]]]):
        self._rows = list(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> Mapping[str, Sequence[int]]:
        return self._rows[index]


def _train_adapter(
    dataframe,
    *,
    validation_dataframe=None,
    spec,
    task: Task,
    revision: str,
    token: str | None,
    parameters: Mapping[str, Any],
    trainer_output_dir: str | Path | None = None,
    resume_from_checkpoint: str | Path | bool | None = None,
):
    import torch
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    loaded = load_causal_model(
        spec,
        revision=revision,
        token=token,
        quantization=bool(parameters["quantization"]),
        device_map=parameters["device_map"],
        precision=str(parameters["precision"]),
    )
    tokenizer = loaded.tokenizer
    tokenizer.padding_side = "right"
    model = loaded.model
    model.config.use_cache = False
    if bool(parameters["quantization"]):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(parameters["gradient_checkpointing"]),
        )
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(parameters["lora_rank"]),
            lora_alpha=int(parameters["lora_alpha"]),
            target_modules=parameters["target_modules"],
            lora_dropout=float(parameters["lora_dropout"]),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    tokenized_rows = _tokenize_training_rows(
        dataframe,
        tokenizer,
        task,
        int(parameters["max_seq_length"]),
        model_alias=spec.alias,
    )
    eval_strategy = str(parameters["eval_strategy"])
    tokenized_validation_rows = None
    if eval_strategy != "no":
        if validation_dataframe is None:
            raise ValueError(
                "validation_dataframe is required when eval_strategy is enabled"
            )
        tokenized_validation_rows = _tokenize_training_rows(
            validation_dataframe,
            tokenizer,
            task,
            int(parameters["max_seq_length"]),
            model_alias=spec.alias,
        )

    use_bf16 = loaded.compute_dtype == "bfloat16"
    optimizer = str(parameters["optimizer"])
    if optimizer == "auto":
        optimizer = (
            "paged_adamw_8bit"
            if bool(parameters["quantization"])
            else "adamw_torch"
        )
    output_dir = Path(trainer_output_dir or DEFAULT_ARTIFACT_ROOT / "trainer")
    save_total_limit = parameters["save_total_limit"]
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(parameters["epochs"]),
        per_device_train_batch_size=int(parameters["batch_size"]),
        per_device_eval_batch_size=int(parameters["eval_batch_size"]),
        gradient_accumulation_steps=int(parameters["gradient_accumulation_steps"]),
        gradient_checkpointing=bool(parameters["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=optimizer,
        learning_rate=float(parameters["learning_rate"]),
        lr_scheduler_type=str(parameters["lr_scheduler_type"]),
        warmup_ratio=float(parameters["warmup_ratio"]),
        weight_decay=float(parameters["weight_decay"]),
        max_grad_norm=float(parameters["max_grad_norm"]),
        logging_steps=int(parameters["logging_steps"]),
        save_strategy=str(parameters["save_strategy"]),
        save_total_limit=(
            None if save_total_limit is None else int(save_total_limit)
        ),
        eval_strategy=eval_strategy,
        load_best_model_at_end=bool(parameters["load_best_model_at_end"]),
        bf16=use_bf16,
        fp16=not use_bf16,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=int(parameters["num_workers"]),
        seed=int(parameters["seed"]),
        data_seed=int(parameters["seed"]),
        full_determinism=bool(parameters["deterministic"]),
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=_TokenizedRowsDataset(tokenized_rows),
        eval_dataset=(
            _TokenizedRowsDataset(tokenized_validation_rows)
            if tokenized_validation_rows is not None
            else None
        ),
        data_collator=collator,
    )
    train_output = trainer.train(
        resume_from_checkpoint=(
            str(resume_from_checkpoint)
            if isinstance(resume_from_checkpoint, Path)
            else resume_from_checkpoint
        )
    )
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()
    tokenizer.padding_side = "left"
    history = list(trainer.state.log_history)
    training_metrics = dict(train_output.metrics)
    del trainer, train_output
    gc.collect()
    torch.cuda.empty_cache()
    return model, tokenizer, loaded.compute_dtype, history, training_metrics


def _load_saved_adapter(
    adapter_dir: Path,
    *,
    spec,
    revision: str,
    token: str | None,
    parameters: Mapping[str, Any],
):
    """Load a saved LoRA adapter over its immutable base-model revision."""
    from peft import PeftModel
    from transformers import AutoTokenizer

    loaded = load_causal_model(
        spec,
        revision=revision,
        token=token,
        quantization=bool(parameters["quantization"]),
        device_map=parameters["device_map"],
        precision=str(parameters["precision"]),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir,
        local_files_only=True,
        trust_remote_code=spec.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = PeftModel.from_pretrained(
        loaded.model,
        adapter_dir,
        is_trainable=False,
        local_files_only=True,
    )
    model.config.use_cache = True
    model.eval()
    return model, tokenizer, loaded.compute_dtype


def _prediction_datasets(
    test_docx_dir: str | Path,
    *,
    selected_datasets: Sequence[str] = ("Full",),
) -> list[tuple[str, tuple[Path, ...]]]:
    """Group selected DOCX benchmark folders by relative directory."""
    root = Path(test_docx_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Test DOCX directory does not exist: {root}")
    grouped: dict[str, list[Path]] = {}
    for document in sorted(root.rglob("*.docx"), key=lambda path: str(path).casefold()):
        relative_parent = document.parent.relative_to(root)
        dataset = relative_parent.as_posix() if relative_parent.parts else "root"
        grouped.setdefault(dataset, []).append(document)
    if not selected_datasets or len(set(selected_datasets)) != len(selected_datasets):
        raise ValueError("selected_datasets must contain unique dataset names")
    missing = set(selected_datasets) - grouped.keys()
    if missing:
        raise FileNotFoundError("No DOCX files found for: " + ", ".join(sorted(missing)))
    return [(name, tuple(grouped[name])) for name in selected_datasets]


def run_lora_document_inference(
    model_name: str,
    task: str,
    hyperparameters: Mapping[str, Any] | None = None,
    *,
    rag_dir: str | Path = DEFAULT_RAG_DIR,
    rag_revision: str = DEFAULT_RAG_REVISION,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    revision: str | None = None,
    test_docx_dir: str | Path = DEFAULT_TEST_DOCX_DIR,
    selected_datasets: Sequence[str] = ("Full",),
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> Path:
    """Load an existing LoRA adapter and write XLSX predictions for test DOCX files.

    This inference-only workflow does not load training/validation datasets, train,
    validate, or score against ``autotest``. Selected ``.docx`` benchmark folders
    below ``test_docx_dir`` are processed with the current parsing, filtering, retrieval,
    and prompting code. The returned directory contains one persistent
    ``model_predictions.xlsx`` workbook per document.
    """
    import pandas as pd

    validated_task = validate_task(task)
    spec = resolve_model(model_name)
    workflow = f"lora-inference/{spec.alias}/{validated_task}"
    announce_stage(workflow, "setup", "Loading the saved adapter configuration.")
    artifact_root = prepare_artifact_root(artifact_root)
    adapter_target = (
        artifact_root
        / "models"
        / "lora"
        / slugify_model_id(spec.model_id)
        / validated_task
    )
    manifest = load_saved_artifact_manifest(
        adapter_target,
        required_files=(
            "run_config.json",
            "adapter_config.json",
            "tokenizer_config.json",
        ),
        weight_files=("adapter_model.safetensors", "adapter_model.bin"),
        expected={
            "model_id": spec.model_id,
            "task": validated_task,
            "prompt_sha256": prompt_sha256(prompt_for_task(validated_task)),
            "prompt_processing": _prompt_processing_strategy(spec.alias),
            "premise_format": SOURCE_PREFIXED_PREMISE_FORMAT,
        },
    )
    stored_parameters = manifest.get("hyperparameters")
    if not isinstance(stored_parameters, dict):
        raise ValueError(
            f"Saved LoRA manifest has no hyperparameters: {adapter_target}"
        )
    parameters = merge_parameters(DEFAULT_LORA_HYPERPARAMETERS, stored_parameters)
    parameters = merge_parameters(parameters, hyperparameters)
    configure_reproducibility(
        int(parameters["seed"]),
        deterministic=bool(parameters["deterministic"]),
    )
    effective_revision = saved_artifact_revision(manifest, adapter_target)
    token = get_huggingface_token()
    if revision is not None:
        requested_revision = resolve_huggingface_revision(
            spec.model_id, revision, token=token
        )
        if requested_revision != effective_revision:
            raise ValueError(
                "Requested base-model revision does not match the saved adapter: "
                f"{requested_revision} != {effective_revision}"
            )

    datasets = _prediction_datasets(test_docx_dir, selected_datasets=selected_datasets)
    announce_stage(workflow, "rag", "Preparing the pinned RAG repository.")
    rag_path, rag_commit = ensure_rag_repository(rag_dir, revision=rag_revision)
    announce_stage(workflow, "rag", f"RAG repository ready at {rag_commit[:12]}.")
    model = tokenizer = None
    try:
        announce_stage(workflow, "load", "Loading the saved LoRA adapter.")
        model, tokenizer, _ = _load_saved_adapter(
            adapter_target,
            spec=spec,
            revision=effective_revision,
            token=token,
            parameters=parameters,
        )
        predictor = CausalPredictor(
            model,
            tokenizer,
            validated_task,
            batch_size=int(parameters["document_batch_size"]),
            max_input_length=int(parameters["max_input_length"]),
            max_new_tokens=int(parameters["max_new_tokens"]),
        )
        retriever = PremiseRetriever.from_rag_directory(
            rag_path,
            embedding_revision=str(parameters["embedding_revision"]),
            embedding_device=str(parameters["embedding_device"]),
        )
        pair_tables = []
        for dataset_name, documents in datasets:
            announce_stage(
                workflow,
                "testing",
                f"Predicting {len(documents)} document(s) from {dataset_name}.",
            )
            tables = run_document_inference(
                documents,
                predictor=predictor,
                retriever=retriever,
                model_id=spec.model_id,
                task=validated_task,
                top_k=int(parameters["retrieval_top_k"]),
                include_source_prefix=True,
            )
            pair_tables.append(add_test_dataset(tables.pairs, dataset_name))
        document_pairs = pd.concat(pair_tables, ignore_index=True)
        output = write_document_review_workbooks(
            f"lora_{slugify_model_id(spec.model_id)}_{validated_task}",
            document_pairs,
            output_dir=results_dir,
        )
        if output is None:
            raise RuntimeError("Document inference produced no prediction pairs")
        announce_stage(workflow, "complete", f"Prediction workbooks saved to {output}.")
        return output
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


def run(
    model_name: str,
    task: str,
    hyperparameters: Mapping[str, Any] | None = None,
    *,
    train_path: str | Path | None = None,
    val_path: str | Path | None = None,
    rag_dir: str | Path = DEFAULT_RAG_DIR,
    rag_revision: str = DEFAULT_RAG_REVISION,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    revision: str | None = None,
    use_existing_model: bool = False,
    allow_validation_mismatch: bool = False,
    document_paths: Sequence[str | Path] | None = None,
    autotest_dir: str | Path = DEFAULT_AUTOTEST_DIR,
    test_docx_dir: str | Path = DEFAULT_TEST_DOCX_DIR,
    score_autotest: bool = True,
    multiple_test: bool = False,
    selected_datasets: Sequence[str] = ("Full",),
    benchmark_scopes: Sequence[str] = ("autotest_model", "autotest_total"),
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    trainer_output_dir: str | Path | None = None,
    resume_from_checkpoint: str | Path | bool | None = None,
    training_only: bool = False,
):
    """Train or reuse, validate, and document-test one task LoRA adapter.

    ``hyperparameters`` may override any key in
    :data:`DEFAULT_LORA_HYPERPARAMETERS`; unknown keys are rejected.
    With ``use_existing_model=True``, the compatible final adapter in artifact
    storage is required and training is skipped. With no explicit
    ``document_paths``, the matched repository benchmark DOCX files are used.
    Set ``score_autotest=False`` to disable benchmark scoring, or
    ``multiple_test=True`` to evaluate paired child folders separately. The
    function returns the combined validation and autotest score table.
    ``trainer_output_dir`` optionally isolates Trainer checkpoints and logs.
    ``resume_from_checkpoint`` resumes a compatible interrupted Trainer run.
    ``allow_validation_mismatch`` permits scoring an existing adapter on a new
    validation file; it does not permit a different training set or settings.
    Set ``training_only=True`` to save the adapter and skip validation, RAG
    loading, document inference, and result-workbook generation.
    """
    validated_task = validate_task(task)
    if allow_validation_mismatch and not use_existing_model:
        raise ValueError("allow_validation_mismatch requires use_existing_model")
    spec = resolve_model(model_name)
    prompt_processing = _prompt_processing_strategy(spec.alias)
    workflow = f"lora/{spec.alias}/{validated_task}"
    announce_stage(workflow, "setup", "Preparing datasets and artifact paths.")
    train_path = Path(train_path or default_dataset_path("train", validated_task))
    val_path = Path(val_path or default_dataset_path("val", validated_task))
    train_hash = file_sha256(train_path)
    val_hash = file_sha256(val_path)
    current_prompt_hash = prompt_sha256(prompt_for_task(validated_task))
    project_drive = prepare_artifact_root(artifact_root)
    adapter_target = (
        project_drive
        / "models"
        / "lora"
        / slugify_model_id(spec.model_id)
        / validated_task
    )
    artifact_manifest: dict[str, Any] | None = None
    if use_existing_model:
        announce_stage(
            workflow,
            "reuse",
            f"Checking the previously trained adapter at {adapter_target}.",
        )
        expected_manifest = {
            "model_id": spec.model_id,
            "task": validated_task,
            "train_sha256": train_hash,
            "prompt_sha256": current_prompt_hash,
            "prompt_processing": prompt_processing,
            "premise_format": SOURCE_PREFIXED_PREMISE_FORMAT,
        }
        if not allow_validation_mismatch:
            expected_manifest["validation_sha256"] = val_hash
        artifact_manifest = load_saved_artifact_manifest(
            adapter_target,
            required_files=(
                "run_config.json",
                "adapter_config.json",
                "tokenizer_config.json",
            ),
            weight_files=("adapter_model.safetensors", "adapter_model.bin"),
            expected=expected_manifest,
        )
        stored_parameters = artifact_manifest.get("hyperparameters")
        if not isinstance(stored_parameters, dict):
            raise ValueError(
                f"Saved LoRA manifest has no hyperparameters: {adapter_target}"
            )
        if allow_validation_mismatch and not artifact_manifest.get("validation_sha256"):
            raise ValueError("Saved LoRA manifest has no validation hash")
        parameters = merge_parameters(DEFAULT_LORA_HYPERPARAMETERS, stored_parameters)
        requested_parameters = merge_parameters(parameters, hyperparameters)
        if allow_validation_mismatch and (
            requested_parameters != parameters
            or (
                artifact_manifest.get("validation_sha256") != val_hash
                and parameters["load_best_model_at_end"]
            )
        ):
            raise ValueError(
                "Validation-data reuse requires unchanged adapter settings and "
                "a checkpoint that was not selected by validation"
            )
        parameters = requested_parameters
    else:
        parameters = merge_parameters(DEFAULT_LORA_HYPERPARAMETERS, hyperparameters)
    configure_reproducibility(
        int(parameters["seed"]),
        deterministic=bool(parameters["deterministic"]),
    )
    reproducibility = reproducibility_metadata(
        int(parameters["seed"]),
        deterministic=bool(parameters["deterministic"]),
    )
    val_dataframe = load_dataset(val_path, validated_task)
    token = get_huggingface_token()
    if artifact_manifest is not None:
        effective_revision = saved_artifact_revision(
            artifact_manifest, adapter_target
        )
        if revision is not None:
            requested_commit = resolve_huggingface_revision(
                spec.model_id, revision, token=token
            )
            if requested_commit != effective_revision:
                raise ValueError(
                    "Requested base-model revision does not match the saved adapter: "
                    f"{requested_commit} != {effective_revision}"
                )
    else:
        effective_revision = resolve_huggingface_revision(
            spec.model_id, revision or spec.revision, token=token
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
                "Loading the saved LoRA adapter; training is skipped.",
            )
            model, tokenizer, compute_dtype = _load_saved_adapter(
                adapter_target,
                spec=spec,
                revision=effective_revision,
                token=token,
                parameters=parameters,
            )
            history = artifact_manifest.get("training_history", [])
            training_metrics = artifact_manifest.get("training_metrics", {})
            announce_stage(workflow, "load", "Saved LoRA adapter loaded.")
        else:
            train_dataframe = load_dataset(train_path, validated_task)
            announce_stage(
                workflow,
                "training",
                f"Starting LoRA fine-tuning on {len(train_dataframe)} examples.",
            )
            model, tokenizer, compute_dtype, history, training_metrics = _train_adapter(
                train_dataframe,
                validation_dataframe=val_dataframe,
                spec=spec,
                task=validated_task,
                revision=effective_revision,
                token=token,
                parameters=parameters,
                trainer_output_dir=trainer_output_dir,
                resume_from_checkpoint=resume_from_checkpoint,
            )
            adapter_target.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(adapter_target, safe_serialization=True)
            tokenizer.save_pretrained(adapter_target)
            (adapter_target / "run_config.json").write_text(
                json.dumps(
                    {
                        "model_id": spec.model_id,
                        "requested_revision": revision or "registry_default",
                        "resolved_revision": effective_revision,
                        "task": validated_task,
                        "hyperparameters": parameters,
                        "training_history": history,
                        "training_metrics": training_metrics,
                        "train_sha256": train_hash,
                        "validation_sha256": val_hash,
                        "prompt_sha256": current_prompt_hash,
                        "prompt_processing": prompt_processing,
                        "premise_format": SOURCE_PREFIXED_PREMISE_FORMAT,
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
                f"Fine-tuning finished; final adapter saved to {adapter_target}.",
            )
        if training_only:
            announce_stage(
                workflow,
                "complete",
                f"Training-only workflow finished at {adapter_target}.",
            )
            return adapter_target
        announce_stage(
            workflow,
            "validation",
            f"Starting validation on {len(val_dataframe)} examples.",
        )
        predictor = CausalPredictor(
            model,
            tokenizer,
            validated_task,
            batch_size=int(parameters["inference_batch_size"]),
            max_input_length=int(parameters["max_input_length"]),
            max_new_tokens=int(parameters["max_new_tokens"]),
        )
        validation_premises = [
            format_model_premise(row.premise, row.source)
            for row in val_dataframe.itertuples(index=False)
        ]
        generated = predictor.predict_examples(
            validation_premises,
            val_dataframe["hypothesis"].tolist(),
            progress_description=f"Validating LoRA {validated_task}",
        )
        evaluation = evaluate_predictions(
            val_dataframe,
            [item.label for item in generated],
            [item.raw_output for item in generated],
            model_id=spec.model_id,
            task=validated_task,
        )
        evaluation.predictions["model_premise"] = validation_premises
        announce_stage(workflow, "validation", "Validation finished.")
        announce_stage(workflow, "testing", "Loading the RAG retrieval index.")
        retriever = PremiseRetriever.from_rag_directory(
            rag_path,
            embedding_revision=str(parameters["embedding_revision"]),
            embedding_device=str(parameters["embedding_device"]),
        )
        announce_stage(workflow, "testing", "RAG retrieval index loaded.")
        document_predictor = CausalPredictor(
            model,
            tokenizer,
            validated_task,
            batch_size=int(parameters["document_batch_size"]),
            max_input_length=int(parameters["max_input_length"]),
            max_new_tokens=int(parameters["max_new_tokens"]),
        )
        announce_stage(
            workflow,
            "testing",
            "Document testing is ready. Select or provide DOCX files, or skip it.",
        )
        import pandas as pd

        if document_paths is None:
            datasets = discover_autotest_datasets(
                autotest_dir, test_docx_dir, multiple_test=multiple_test,
                selected_datasets=selected_datasets,
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
                    predictor=document_predictor,
                    retriever=retriever,
                    model_id=spec.model_id,
                    task=validated_task,
                    top_k=int(parameters["retrieval_top_k"]),
                    include_source_prefix=True,
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
                        model_id=spec.model_id,
                        task=validated_task,
                        autotest_dir=review_dir,
                        docx_dir=document_dir,
                        test_dataset=dataset_name,
                        benchmark_scopes=benchmark_scopes,
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
            f"lora_{slugify_model_id(spec.model_id)}_{validated_task}",
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
                "workflow": "lora",
                "model_id": spec.model_id,
                "requested_revision": revision or (
                    "saved_artifact" if artifact_manifest is not None else "registry_default"
                ),
                "resolved_revision": effective_revision,
                "task": validated_task,
                "parameters": parameters,
                "training_metrics": training_metrics,
                "train_sha256": train_hash,
                "validation_sha256": val_hash,
                "prompt_sha256": current_prompt_hash,
                "prompt_processing": prompt_processing,
                "premise_format": SOURCE_PREFIXED_PREMISE_FORMAT,
                "rag_requested_revision": rag_revision,
                "rag_commit": rag_commit,
                "compute_dtype": compute_dtype,
                "summary_enabled": False,
                "autotest_enabled": score_autotest,
                "multiple_test": multiple_test,
                "test_datasets": [run[0] for run in inference_runs],
                "autotest_dir": Path(autotest_dir),
                "test_docx_dir": Path(test_docx_dir),
                "drive_adapter_path": adapter_target,
                "artifact_path": adapter_target,
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
            f"lora_{slugify_model_id(spec.model_id)}_{validated_task}",
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
