"""Optional legal-domain cross-encoder fine-tuning."""

from __future__ import annotations

from pathlib import Path
from LCD.shared.common import REPOSITORY_ROOT, DEFAULT_RAG_DIR


def train_reranker(
    train_frame,
    validation_frame,
    *,
    model_id: str,
    revision: str | None,
    token: str | None,
    trust_remote_code: bool,
    output_dir: str | Path,
    parameters,
):
    """Fine-tune a one-logit reranker and return its artifact metadata."""
    import torch
    from datasets import Dataset
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder import (
        CrossEncoderTrainer,
        CrossEncoderTrainingArguments,
        losses,
    )
    from sentence_transformers.cross_encoder.evaluation import (
        CrossEncoderClassificationEvaluator,
    )

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    model = CrossEncoder(
        model_id,
        num_labels=1,
        revision=revision,
        token=token,
        trust_remote_code=trust_remote_code,
        max_length=int(parameters["reranker_max_length"]),
    )
    train_dataset = Dataset.from_dict(
        {
            "query": train_frame["hypothesis"].tolist(),
            "document": train_frame["premise"].tolist(),
            "label": train_frame["label"].astype(float).tolist(),
        }
    )
    evaluator = CrossEncoderClassificationEvaluator(
        sentence_pairs=[
            [hypothesis, premise]
            for hypothesis, premise in zip(
                validation_frame["hypothesis"], validation_frame["premise"]
            )
        ],
        labels=validation_frame["label"].astype(int).tolist(),
        name="validation",
        batch_size=int(parameters["reranker_eval_batch_size"]),
        show_progress_bar=True,
    )
    precision = str(parameters["reranker_precision"]).strip().lower()
    if precision not in {"auto", "float16", "bfloat16", "float32"}:
        raise ValueError(
            "reranker_precision must be auto, float16, bfloat16, or float32"
        )
    bf16 = precision == "bfloat16" or (
        precision == "auto"
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )
    fp16 = precision == "float16" or (
        precision == "auto" and torch.cuda.is_available() and not bf16
    )
    arguments = CrossEncoderTrainingArguments(
        output_dir=str(target / "trainer"),
        num_train_epochs=float(parameters["reranker_epochs"]),
        per_device_train_batch_size=int(parameters["reranker_batch_size"]),
        per_device_eval_batch_size=int(parameters["reranker_eval_batch_size"]),
        gradient_accumulation_steps=int(
            parameters["reranker_gradient_accumulation_steps"]
        ),
        learning_rate=float(parameters["reranker_learning_rate"]),
        warmup_ratio=float(parameters["reranker_warmup_ratio"]),
        weight_decay=float(parameters["reranker_weight_decay"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_validation_average_precision",
        greater_is_better=True,
        save_total_limit=2,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=bool(parameters["reranker_gradient_checkpointing"]),
        dataloader_num_workers=int(parameters["num_workers"]),
        seed=int(parameters["seed"]),
        data_seed=int(parameters["seed"]),
        full_determinism=bool(parameters["deterministic"]),
        report_to="none",
    )
    trainer = CrossEncoderTrainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        loss=losses.BinaryCrossEntropyLoss(model),
        evaluator=evaluator,
    )
    result = trainer.train()
    model_dir = target / "reranker_model"
    model.save_pretrained(str(model_dir), safe_serialization=True)
    metrics = evaluator(model, output_path=str(target))
    scores = model.predict(
        [
            (hypothesis, premise)
            for hypothesis, premise in zip(
                validation_frame["hypothesis"], validation_frame["premise"]
            )
        ],
        batch_size=int(parameters["reranker_eval_batch_size"]),
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    predictions = validation_frame.copy()
    predictions["reranker_score"] = scores
    predictions.to_csv(
        target / "reranker_validation_predictions.csv",
        index=False,
        encoding="utf-8",
    )
    return model_dir, metrics, list(trainer.state.log_history), dict(result.metrics)
