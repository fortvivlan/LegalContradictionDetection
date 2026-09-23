"""Evaluate a recovered LoRA adapter on saved current and legacy pair sets."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Sequence

from LCD.shared.autotest_scoring import (
    RAW_LEGACY_REVIEW_COLUMNS,
    _evaluate,
    discover_autotest_cases,
    normalize_subject_key,
    score_autotest_predictions,
)
from LCD.shared.common import file_sha256, prompt_sha256
from LCD.shared.prompting import (
    apply_chat_template,
    build_generation_prompt,
    build_messages,
    prompt_for_task,
)
from LCD.shared.reporting import write_results_workbook


def extract_legacy_label(text: str) -> str | None:
    """Parse a ternary label with the precedence used in the legacy notebook."""
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text)
    normalized = cleaned.lower().strip()
    for label in ("not mentioned", "contradiction", "entailment"):
        if label in normalized:
            return label
    return None


def load_legacy_autotest_pairs(legacy_autotest_dir: str | Path):
    """Load exact premise/hypothesis strings from raw 2025 autotest exports."""
    import pandas as pd

    root = Path(legacy_autotest_dir)
    rows: list[dict[str, object]] = []
    for workbook in sorted(
        path for path in root.glob("*.xlsx") if not path.name.startswith("~$")
    ):
        frame = pd.read_excel(
            workbook, sheet_name=0, dtype=str, keep_default_na=False, engine="openpyxl"
        )
        missing = sorted(set(RAW_LEGACY_REVIEW_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(
                f"{workbook} is not a raw legacy autotest export; missing: "
                + ", ".join(missing)
            )
        for row_index, row in frame.iterrows():
            label = str(row["answer"]).strip().casefold()
            if label not in {"contradiction", "entailment", "not mentioned"}:
                raise ValueError(
                    f"Invalid legacy label in {workbook.name}, row {row_index + 2}: "
                    f"{label!r}"
                )
            rows.append(
                {
                    "pair_set": "legacy_exact",
                    "subject_key": normalize_subject_key(workbook),
                    "expert_workbook": workbook.name,
                    "excel_row": int(row_index) + 2,
                    "hypothesis": str(row["sentence"]),
                    "premise": str(row["premise"]),
                    "article_number": str(row["article"]),
                    "gold_label": label,
                }
            )
    if not rows:
        raise FileNotFoundError(f"No legacy XLSX files found in {root}")
    return pd.DataFrame(rows)


def _load_adapter(
    adapter_dir: Path,
    *,
    base_model_id: str,
    token: str | None,
    quantization: bool,
    compute_dtype: str,
):
    # PyTorch 2.13 enables eager Triton-native kernels that JIT a C shim. The
    # regular CUDA kernels are sufficient for inference and work in minimal
    # Minimal local environments without a system C compiler.
    os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
    import torch
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        PreTrainedTokenizerFast,
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir, local_files_only=True)
    except ValueError as error:
        if "TokenizersBackend" not in str(error):
            raise
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(adapter_dir / "tokenizer.json"),
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="</s>",
        )
        tokenizer.chat_template = (adapter_dir / "chat_template.jinja").read_text(
            encoding="utf-8"
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.float16 if compute_dtype == "float16" else torch.bfloat16
    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "token": token,
        "dtype": dtype,
    }
    if quantization:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)
    model = PeftModel.from_pretrained(
        base_model, adapter_dir, is_trainable=False, local_files_only=True
    )
    model.config.use_cache = True
    model.eval()
    return model, tokenizer


def _predict_pairs(
    model: Any,
    tokenizer: Any,
    premises: Sequence[str],
    hypotheses: Sequence[str],
    *,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    prompt_mode: str,
) -> tuple[list[str], list[str]]:
    import torch
    from tqdm.auto import tqdm

    predictions: list[str] = []
    raw_outputs: list[str] = []
    device = next(model.parameters()).device
    for start in tqdm(
        range(0, len(premises), batch_size), desc="Recovered adapter inference"
    ):
        stop = min(start + batch_size, len(premises))
        prompts = []
        for premise, hypothesis in zip(
            premises[start:stop], hypotheses[start:stop]
        ):
            if prompt_mode == "legacy_duplicated":
                messages = build_messages(premise, hypothesis, "ternary")
                messages[1]["content"] = (
                    f"{prompt_for_task('ternary')}\n{messages[1]['content']}"
                )
                prompts.append(
                    apply_chat_template(
                        tokenizer, messages, add_generation_prompt=True
                    )
                )
            else:
                prompts.append(
                    build_generation_prompt(
                        tokenizer, premise, hypothesis, "ternary"
                    )
                )
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        inputs.pop("token_type_ids", None)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        raw_outputs.extend(decoded)
        predictions.extend(
            [extract_legacy_label(output) or "invalid" for output in decoded]
        )
    return predictions, raw_outputs


def _direct_evaluation(frame, *, model_id: str, scope: str):
    rows = frame.rename(columns={"recovered_prediction": "prediction"})
    return _evaluate(rows, model_id=model_id, task="ternary", scope=scope)


def _with_pair_set(table, pair_set: str):
    result = table.copy()
    result.insert(0, "pair_set", pair_set)
    return result


def run_recovered_adapter_evaluation(
    adapter_dir: str | Path,
    current_results_workbook: str | Path,
    *,
    current_autotest_dir: str | Path = "autotest/Dialogue",
    current_docx_dir: str | Path = "test_docx/Dialogue",
    legacy_autotest_dir: str | Path = ".legacy",
    test_dataset: str = "Dialogue",
    base_model_id: str | None = None,
    output_dir: str | Path = "local/experiments/lora/series_00_legacy_adapters/recovered/results",
    batch_size: int = 4,
    max_length: int = 4096,
    max_new_tokens: int = 64,
    quantization: bool = True,
    compute_dtype: str = "bfloat16",
    prompt_mode: str = "canonical",
    include_source_prefix_diagnostic: bool = True,
    hf_token: str | None = None,
) -> Path:
    """Run one recovered ternary adapter on current and exact legacy pairs.

    The model is loaded once. Current pairs are read from a saved
    ``document_pairs`` sheet, so RAG is not rerun. Legacy pairs retain the exact
    strings stored in the raw ``sentence/article/premise/answer`` workbooks.
    """
    import pandas as pd

    if compute_dtype not in {"bfloat16", "float16"}:
        raise ValueError("compute_dtype must be 'bfloat16' or 'float16'")
    if prompt_mode not in {"canonical", "legacy_duplicated"}:
        raise ValueError("prompt_mode must be 'canonical' or 'legacy_duplicated'")
    adapter_path = Path(adapter_dir)
    results_path = Path(current_results_workbook)
    config_path = adapter_path / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    resolved_base_model = base_model_id or str(config["base_model_name_or_path"])
    pairs = pd.read_excel(
        results_path, sheet_name="document_pairs", engine="openpyxl"
    )
    if "test_dataset" in pairs.columns:
        pairs = pairs[pairs["test_dataset"].astype(str).eq(test_dataset)].copy()
    pairs = pairs[pairs["task"].astype(str).eq("ternary")].copy()
    if pairs.empty:
        raise ValueError(f"No ternary {test_dataset!r} pairs in {results_path}")
    legacy_pairs = load_legacy_autotest_pairs(legacy_autotest_dir)
    token = hf_token or os.environ.get("HF_TOKEN")
    model_id = f"recovered:{adapter_path.name}"
    model, tokenizer = _load_adapter(
        adapter_path,
        base_model_id=resolved_base_model,
        token=token,
        quantization=quantization,
        compute_dtype=compute_dtype,
    )

    current_predictions, current_raw = _predict_pairs(
        model,
        tokenizer,
        pairs["premise"].astype(str).tolist(),
        pairs["hypothesis"].astype(str).tolist(),
        batch_size=batch_size,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        prompt_mode=prompt_mode,
    )
    current_predicted_pairs = pairs.copy()
    current_predicted_pairs["prediction"] = current_predictions
    current_predicted_pairs["raw_output"] = current_raw
    documents, _ = discover_autotest_cases(
        current_autotest_dir, current_docx_dir
    )
    current_tables = score_autotest_predictions(
        current_predicted_pairs,
        documents,
        model_id=model_id,
        task="ternary",
        autotest_dir=current_autotest_dir,
        docx_dir=current_docx_dir,
        test_dataset="current_bare_premise",
    )

    legacy_predictions, legacy_raw = _predict_pairs(
        model,
        tokenizer,
        legacy_pairs["premise"].astype(str).tolist(),
        legacy_pairs["hypothesis"].astype(str).tolist(),
        batch_size=batch_size,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        prompt_mode=prompt_mode,
    )
    legacy_pairs["recovered_prediction"] = legacy_predictions
    legacy_pairs["raw_output"] = legacy_raw
    legacy_scores, legacy_per_class, legacy_confusion = _direct_evaluation(
        legacy_pairs, model_id=model_id, scope="legacy_exact_pairs"
    )

    score_frames = [_with_pair_set(current_tables.scores, "current_bare_premise")]
    per_class_frames = [
        _with_pair_set(current_tables.per_class, "current_bare_premise")
    ]
    confusion_frames = [
        _with_pair_set(current_tables.confusion_matrix, "current_bare_premise")
    ]
    tables: dict[str, Any] = {
        "current_predictions": current_tables.alignment,
        "legacy_predictions": legacy_pairs,
    }

    if include_source_prefix_diagnostic:
        prefixed_premises = [
            f"{str(source).strip()} {str(premise).strip()}".strip()
            for source, premise in zip(pairs["source"], pairs["premise"])
        ]
        prefixed_predictions, prefixed_raw = _predict_pairs(
            model,
            tokenizer,
            prefixed_premises,
            pairs["hypothesis"].astype(str).tolist(),
            batch_size=batch_size,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            prompt_mode=prompt_mode,
        )
        prefixed_pairs = pairs.copy()
        prefixed_pairs["prediction"] = prefixed_predictions
        prefixed_pairs["raw_output"] = prefixed_raw
        prefixed_tables = score_autotest_predictions(
            prefixed_pairs,
            documents,
            model_id=model_id,
            task="ternary",
            autotest_dir=current_autotest_dir,
            docx_dir=current_docx_dir,
            test_dataset="current_source_prefixed",
        )
        score_frames.append(
            _with_pair_set(prefixed_tables.scores, "current_source_prefixed")
        )
        per_class_frames.append(
            _with_pair_set(prefixed_tables.per_class, "current_source_prefixed")
        )
        confusion_frames.append(
            _with_pair_set(
                prefixed_tables.confusion_matrix, "current_source_prefixed"
            )
        )
        tables["prefixed_predictions"] = prefixed_tables.alignment
        original_predictions = pairs.loc[
            :, ["document", "hypothesis_id", "premise", "source", "prediction"]
        ].rename(columns={"prediction": "current_adapter_prediction"})
        tables["adapter_prediction_comparison"] = (
            prefixed_tables.alignment[
                prefixed_tables.alignment["alignment_status"].eq("retrieved")
            ]
            .rename(columns={"prediction": "recovered_adapter_prediction"})
            .merge(
                original_predictions,
                on=["document", "hypothesis_id", "premise", "source"],
                how="left",
                validate="one_to_one",
            )
        )
        tables["current_prompt_comparison"] = pd.DataFrame(
            {
                "document": pairs["document"],
                "hypothesis_id": pairs["hypothesis_id"],
                "hypothesis": pairs["hypothesis"],
                "premise": pairs["premise"],
                "source": pairs["source"],
                "bare_prediction": current_predictions,
                "source_prefixed_prediction": prefixed_predictions,
                "prediction_changed": [
                    left != right
                    for left, right in zip(current_predictions, prefixed_predictions)
                ],
            }
        )

    score_frames.append(_with_pair_set(legacy_scores, "legacy_exact"))
    per_class_frames.append(_with_pair_set(legacy_per_class, "legacy_exact"))
    confusion_frames.append(_with_pair_set(legacy_confusion, "legacy_exact"))
    tables = {
        "scores": pd.concat(score_frames, ignore_index=True),
        "per_class": pd.concat(per_class_frames, ignore_index=True),
        "confusion": pd.concat(confusion_frames, ignore_index=True),
        **tables,
    }
    metadata = {
        "workflow": "recovered_adapter_evaluation",
        "adapter_dir": adapter_path,
        "adapter_sha256": file_sha256(adapter_path / "adapter_model.safetensors"),
        "adapter_config_sha256": file_sha256(config_path),
        "base_model_id": resolved_base_model,
        "current_results_workbook": results_path,
        "current_results_sha256": file_sha256(results_path),
        "current_pair_count": len(pairs),
        "legacy_pair_count": len(legacy_pairs),
        "prompt_sha256": prompt_sha256(prompt_for_task("ternary")),
        "batch_size": batch_size,
        "max_length": max_length,
        "max_new_tokens": max_new_tokens,
        "quantization": quantization,
        "compute_dtype": compute_dtype,
        "prompt_mode": prompt_mode,
        "source_prefix_diagnostic": include_source_prefix_diagnostic,
    }
    return write_results_workbook(
        "recovered_adapter_evaluation", tables, metadata, output_dir=output_dir
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter_dir", type=Path)
    parser.add_argument("current_results_workbook", type=Path)
    parser.add_argument("--current-autotest-dir", type=Path, default=Path("autotest/Dialogue"))
    parser.add_argument("--current-docx-dir", type=Path, default=Path("test_docx/Dialogue"))
    parser.add_argument("--legacy-autotest-dir", type=Path, default=Path(".legacy"))
    parser.add_argument("--test-dataset", default="Dialogue")
    parser.add_argument("--base-model-id")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("local/experiments/lora/series_00_legacy_adapters/recovered/results"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--no-quantization", action="store_true")
    parser.add_argument(
        "--compute-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("canonical", "legacy_duplicated"),
        default="canonical",
    )
    parser.add_argument("--no-source-prefix-diagnostic", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    print(
        run_recovered_adapter_evaluation(
            arguments.adapter_dir,
            arguments.current_results_workbook,
            current_autotest_dir=arguments.current_autotest_dir,
            current_docx_dir=arguments.current_docx_dir,
            legacy_autotest_dir=arguments.legacy_autotest_dir,
            test_dataset=arguments.test_dataset,
            base_model_id=arguments.base_model_id,
            output_dir=arguments.output_dir,
            batch_size=arguments.batch_size,
            max_length=arguments.max_length,
            max_new_tokens=arguments.max_new_tokens,
            quantization=not arguments.no_quantization,
            compute_dtype=arguments.compute_dtype,
            prompt_mode=arguments.prompt_mode,
            include_source_prefix_diagnostic=(
                not arguments.no_source_prefix_diagnostic
            ),
        )
    )
