"""Evaluate a saved coordinate-search LoRA on a new validation CSV.

This inference-only entrypoint does not check the training-time validation hash,
because a changed validation set is the purpose of this workflow. It preserves
the adapter and the original search reports.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from LCD.shared.common import (
    configure_reproducibility,
    default_dataset_path,
    evaluate_predictions,
    file_sha256,
    load_dataset,
    prompt_sha256,
    resolve_model,
)
from LCD.shared.inference import format_model_premise
from LCD.shared.llm_common import CausalPredictor
from LCD.shared.lora import _load_saved_adapter
from LCD.shared.local_support import get_huggingface_token
from LCD.shared.prompting import prompt_for_task
from LCD.shared.reporting import write_results_workbook


def validate_saved_adapter(
    adapter_dir: str | Path,
    *,
    val_path: str | Path = default_dataset_path("val", "ternary"),
    output_dir: str | Path,
    device_map: str = "auto",
) -> Path:
    """Score a saved adapter on ``val_path`` and write an auditable workbook.

    The adapter's saved base-model revision, generation settings, and seed are
    reused. The workbook contains summary, per-class, confusion, predictions,
    and provenance sheets. No training or document benchmark is run.
    """
    adapter_dir = Path(adapter_dir).resolve()
    val_path = Path(val_path).resolve()
    output_dir = Path(output_dir).resolve()
    manifest = json.loads((adapter_dir / "run_config.json").read_text(encoding="utf-8"))
    if not (adapter_dir / "adapter_config.json").is_file() or not any(
        (adapter_dir / name).is_file()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise FileNotFoundError(f"Incomplete adapter at {adapter_dir}")
    task = manifest["task"]
    spec = resolve_model(manifest["model_id"])
    parameters = dict(manifest["hyperparameters"])
    parameters["device_map"] = device_map
    current_prompt_hash = prompt_sha256(prompt_for_task(task))
    if current_prompt_hash != manifest["prompt_sha256"]:
        raise ValueError("Current task prompt differs from the adapter's training prompt")
    dataframe = load_dataset(val_path, task)
    configure_reproducibility(
        int(parameters["seed"]), deterministic=bool(parameters["deterministic"])
    )
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    model, tokenizer, _ = _load_saved_adapter(
        adapter_dir,
        spec=spec,
        revision=manifest["resolved_revision"],
        token=get_huggingface_token(),
        parameters=parameters,
    )
    premises = [
        format_model_premise(row.premise, row.source)
        for row in dataframe.itertuples(index=False)
    ]
    predictor = CausalPredictor(
        model,
        tokenizer,
        task,
        batch_size=int(parameters["inference_batch_size"]),
        max_input_length=int(parameters["max_input_length"]),
        max_new_tokens=int(parameters["max_new_tokens"]),
    )
    generated = predictor.predict_examples(
        premises,
        dataframe["hypothesis"].tolist(),
        progress_description="Validating saved LoRA",
    )
    evaluation = evaluate_predictions(
        dataframe,
        [item.label for item in generated],
        [item.raw_output for item in generated],
        model_id=spec.model_id,
        task=task,
    )
    evaluation.predictions["model_premise"] = premises
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation.predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    evaluation.scores.to_csv(output_dir / "scores.csv", index=False)
    metadata = {
        "adapter_dir": str(adapter_dir),
        "base_model_revision": manifest["resolved_revision"],
        "adapter_training_validation_sha256": manifest["validation_sha256"],
        "validation_csv": str(val_path),
        "validation_sha256": file_sha256(val_path),
        "prompt_sha256": current_prompt_hash,
        "seed": parameters["seed"],
        "device_map": device_map,
        "inference_batch_size": parameters["inference_batch_size"],
        "max_input_length": parameters["max_input_length"],
        "max_new_tokens": parameters["max_new_tokens"],
    }
    workbook = write_results_workbook(
        "saved_lora_validation",
        {
            "scores": evaluation.scores,
            "per_class": evaluation.per_class,
            "confusion": evaluation.confusion_matrix,
            "validation_predictions": evaluation.predictions,
        },
        metadata,
        output_dir=output_dir,
    )
    print(evaluation.scores.to_string(index=False), flush=True)
    print(f"Workbook: {workbook}", flush=True)
    return workbook


def main() -> None:
    """Parse local CLI arguments for saved-adapter validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--val-path", type=Path, default=default_dataset_path("val", "ternary"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()
    validate_saved_adapter(
        args.adapter_dir,
        val_path=args.val_path,
        output_dir=args.output_dir,
        device_map=args.device_map,
    )


if __name__ == "__main__":
    main()
