"""Qwen3-8B full-document QLoRA preparation, training, and disposable 8k probe."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from LCD.shared.common import file_sha256
from LCD.shared.prompting import build_training_texts
from .data import COLUMNS, DEFAULT_OUTPUT, LABELS

DEFAULT_MODEL = "Qwen/Qwen3-8B"
MAX_TOKENS = 8192
LOSS_MODE = "selective_response_logits_v1"


def read_export(path: Path) -> list[dict[str, str]]:
    """Read the isolated Full training copy and validate required fields and labels."""
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(COLUMNS).issubset(reader.fieldnames or ()):
            raise ValueError(f"{path}: missing export columns")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no training examples")
    for row in rows:
        if row["expert_label"] not in LABELS or any(not row[column] for column in COLUMNS):
            raise ValueError(f"{row.get('workbook')}:{row.get('workbook_row')}: invalid export row")
    return rows


def _ids(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def tokenize_example(tokenizer: Any, row: dict[str, str], max_tokens: int = MAX_TOKENS,
                     *, context: str | None = None) -> tuple[dict[str, list[int]], int]:
    """Mask the prompt; shorten only the document's beginning if the cap requires it."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    original = row["document_context"] if context is None else context
    premise = f"{row['article_number']}\n{row['premise']}"

    def render(document: str) -> tuple[list[int], list[int]]:
        prompt, full = build_training_texts(tokenizer, premise, row["hypothesis"],
                                            row["expert_label"], "ternary",
                                            document_context=document)
        prompt_ids, full_ids = _ids(tokenizer, prompt), _ids(tokenizer, full)
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError("Qwen chat template does not preserve the generation prompt prefix")
        if len(full_ids) == len(prompt_ids):
            raise ValueError("No response tokens")
        return prompt_ids, full_ids

    prompt_ids, full_ids = render(original)
    retained = len(original)
    if len(full_ids) > max_tokens:
        empty_prompt, empty_full = render("")
        if len(empty_full) > max_tokens:
            raise ValueError(f"Non-context fields need {len(empty_full)} tokens; cap is {max_tokens}")
        # Search for the longest character suffix that fits. Token boundaries can
        # make counts locally nonmonotonic, so verify each chosen candidate.
        low, high = 0, len(original)
        best = (empty_prompt, empty_full, 0)
        while low <= high:
            middle = (low + high) // 2
            candidate_prompt, candidate_full = render(original[-middle:] if middle else "")
            if len(candidate_full) <= max_tokens:
                best = (candidate_prompt, candidate_full, middle)
                low = middle + 1
            else:
                high = middle - 1
        prompt_ids, full_ids, retained = best
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}, retained


def prepare_rows(tokenizer: Any, rows: list[dict[str, str]],
                 max_tokens: int = MAX_TOKENS) -> tuple[list[dict[str, list[int]]], dict[str, int]]:
    """Tokenize all natural examples and summarize complete sequence lengths."""
    results = [tokenize_example(tokenizer, row, max_tokens) for row in rows]
    prepared = [example for example, _ in results]
    lengths = [len(row["input_ids"]) for row in prepared]
    return prepared, {"rows": len(lengths), "min_tokens": min(lengths),
                      "max_tokens": max(lengths), "truncated_context_rows": sum(
                          len(row["document_context"]) != retained
                          for row, (_, retained) in zip(rows, results))}


def exact_probe_example(tokenizer: Any, row: dict[str, str],
                        target_tokens: int = MAX_TOKENS) -> dict[str, list[int]]:
    """Repeat source text until a disposable example has exactly target_tokens."""
    document = row["document_context"]
    if not document:
        raise ValueError("Probe requires source document text")
    # A dummy response prevents any expert Full label from entering the probe.
    dummy = dict(row, expert_label="not mentioned")
    unit = document + "\n"
    base, _ = tokenize_example(tokenizer, dummy, target_tokens, context="")
    if len(base["input_ids"]) >= target_tokens:
        raise ValueError("Prompt without context leaves no room for repeated document text")
    repetitions = 1
    while True:
        candidate, _ = tokenize_example(tokenizer, dummy, target_tokens * 2,
                                        context=unit * repetitions)
        if len(candidate["input_ids"]) >= target_tokens:
            break
        repetitions *= 2
    low, high = 0, len(unit) * repetitions
    full_context = unit * repetitions
    while low <= high:
        middle = (low + high) // 2
        candidate, _ = tokenize_example(tokenizer, dummy, target_tokens * 2,
                                        context=full_context[:middle])
        count = len(candidate["input_ids"])
        if count == target_tokens:
            return candidate
        if count < target_tokens:
            low = middle + 1
        else:
            high = middle - 1
    # Token counts may skip a value at a boundary. Try nearby offsets.
    for size in range(max(0, high - 32), min(len(full_context), low + 32) + 1):
        candidate, _ = tokenize_example(tokenizer, dummy, target_tokens * 2,
                                        context=full_context[:size])
        if len(candidate["input_ids"]) == target_tokens:
            return candidate
    raise ValueError("Could not construct an exact-length probe from repeated document text")


def load_model(model_id: str, revision: str | None, device: str, seed: int):
    """Load the Qwen base with NF4, SDPA, and checkpointed LoRA layers."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    if not torch.cuda.is_available() or not device.startswith("cuda"):
        raise RuntimeError("QLoRA requires a CUDA device")
    set_seed(seed)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, device_map={"": device}, dtype=dtype,
        attn_implementation="sdpa", quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True))
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear",
        bias="none", task_type="CAUSAL_LM"))
    return model, tokenizer, dtype


def selective_response_loss(model: Any, batch: dict[str, Any], *,
                            num_items_in_batch: Any = None,
                            return_outputs: bool = False):
    """Score only the final response tokens, with the usual causal shift.

    This experiment uses one unpadded example per microbatch. Keeping one
    preceding logit lets position i - 1 predict response token i; the final
    logit has no target. ``num_items_in_batch`` is the supervised-token count
    across a Trainer gradient-accumulation group.
    """
    import torch
    import torch.nn.functional as F

    labels = batch["labels"]
    if labels.ndim != 2 or labels.shape[0] != 1 or labels.shape != batch["input_ids"].shape:
        raise ValueError("Selective loss requires one unpadded sequence per microbatch")
    if "attention_mask" in batch and not torch.all(batch["attention_mask"] == 1):
        raise ValueError("Selective loss requires an unpadded attention mask")
    supervised = labels.ne(-100)[0]
    count = int(supervised.sum().item())
    if count < 1 or count >= labels.shape[1] or not bool(supervised[-count:].all()) \
            or bool(supervised[:-count].any()):
        raise ValueError("Supervised response must be a nonempty contiguous suffix with a prompt")
    outputs = model(input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    logits_to_keep=count + 1)
    logits = outputs.logits
    if logits.shape[:2] != (1, count + 1):
        raise ValueError("Model did not honor logits_to_keep")
    # Match ForCausalLMLoss: upcast logits, shift targets, and average only
    # supervised tokens (or normalize by the full accumulation group's count).
    loss = F.cross_entropy(logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
                           labels[:, -count:].reshape(-1), reduction="sum")
    loss = loss / (count if num_items_in_batch is None else num_items_in_batch)
    return (loss, outputs) if return_outputs else loss


def compare_loss_gradients(csv_path: Path, output_path: Path, *,
                           model_id: str = DEFAULT_MODEL,
                           revision: str | None = None, device: str = "cuda:0",
                           seed: int = 42, row_index: int = 0,
                           second_loss_mode: str = "selective",
                           loss_atol: float = 1e-4,
                           gradient_atol: float = 1e-4,
                           rtol: float = 1e-3,
                           gradient_norm_rtol: float = 2e-2) -> dict[str, Any]:
    """Compare full and selective logits on one context-free QLoRA example.

    Both passes use the same model weights, batch, and RNG state. The first
    pass stores only LoRA gradients on CPU; no optimizer step or adapter save
    occurs. The BF16 gradient check uses relative L2 norm because elementwise
    relative error is unstable near zero. The JSON report stays under the
    caller's chosen output path.
    """
    import torch
    import torch.nn.functional as F

    if second_loss_mode not in {"selective", "full"}:
        raise ValueError("second_loss_mode must be selective or full")
    rows = read_export(csv_path)
    if not 0 <= row_index < len(rows):
        raise ValueError(f"row_index must be between 0 and {len(rows) - 1}")
    row = rows[row_index]
    model, tokenizer, dtype = load_model(model_id, revision, device, seed)
    example, _ = tokenize_example(tokenizer, row, context="")
    batch = {key: torch.tensor([value], device=device) for key, value in example.items()}
    model.train()
    parameters = [(name, parameter) for name, parameter in model.named_parameters()
                  if parameter.requires_grad and "lora_" in name]
    if not parameters:
        raise RuntimeError("Loaded model has no trainable LoRA parameters")
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device)

    full_outputs = model(**batch)
    full_loss = full_outputs.loss
    response_count = int(batch["labels"].ne(-100).sum().item())
    full_response_logits = full_outputs.logits[:, -response_count - 1:, :].detach().float().cpu()
    masked_response_loss = F.cross_entropy(
        full_response_logits[:, :-1, :].reshape(-1, full_response_logits.shape[-1]),
        batch["labels"][:, -response_count:].cpu().reshape(-1), reduction="sum") / response_count
    full_loss.backward()
    del full_outputs
    full_gradients = {name: parameter.grad.detach().float().cpu().clone()
                      for name, parameter in parameters}
    model.zero_grad(set_to_none=True)
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng, device)

    if second_loss_mode == "selective":
        second_loss, second_outputs = selective_response_loss(model, batch,
                                                              return_outputs=True)
        second_response_logits = second_outputs.logits.detach().float().cpu()
    else:
        second_outputs = model(**batch)
        second_loss = second_outputs.loss
        second_response_logits = second_outputs.logits[:, -response_count - 1:, :].detach().float().cpu()
    logits_difference = full_response_logits - second_response_logits
    logits_max_difference = logits_difference.abs().max().item()
    logits_relative_l2_difference = (
        torch.linalg.vector_norm(logits_difference)
        / torch.linalg.vector_norm(full_response_logits).clamp_min(1e-12)).item()
    second_loss.backward()
    del second_outputs
    gradient_details = []
    total_squared_difference = 0.0
    total_squared_full = 0.0
    for name, parameter in parameters:
        full = full_gradients[name]
        selective = parameter.grad.detach().float().cpu()
        difference = (full - selective).abs()
        total_squared_difference += torch.sum(difference.double().square()).item()
        total_squared_full += torch.sum(full.double().square()).item()
        relative_l2_difference = (torch.linalg.vector_norm(difference)
                                  / torch.linalg.vector_norm(full).clamp_min(1e-12)).item()
        gradient_details.append({
            "name": name,
            "max_absolute_difference": difference.max().item(),
            "relative_l2_difference": relative_l2_difference,
            "elementwise_close": bool(torch.allclose(full, selective,
                                                       atol=gradient_atol, rtol=rtol)),
            "norm_close": relative_l2_difference <= gradient_norm_rtol,
        })
    full_value = full_loss.detach().float().item()
    second_value = second_loss.detach().float().item()
    loss_close = bool(torch.isclose(torch.tensor(full_value), torch.tensor(second_value),
                                   atol=loss_atol, rtol=rtol))
    record = {
        "model": model_id, "revision": revision, "device": device, "seed": seed,
        "second_loss_mode": second_loss_mode,
        "dtype": str(dtype), "quantization": "4-bit NF4 double quantization",
        "row_index": row_index, "workbook": row["workbook"],
        "workbook_row": row["workbook_row"], "source_document": row["document"],
        "hypothesis": row["hypothesis"], "premise": row["premise"],
        "citation": row["article_number"], "expert_label": row["expert_label"],
        "retrieval_method": "provided reviewed pair; no retrieval in this experiment",
        "prediction": None,
        "document_context": "omitted for this comparison",
        "tokens": len(example["input_ids"]),
        "supervised_tokens": sum(label != -100 for label in example["labels"]),
        "full_loss": full_value, "second_loss": second_value,
        "full_logits_masked_loss": masked_response_loss.item(),
        "max_response_logit_absolute_difference": logits_max_difference,
        "response_logits_relative_l2_difference": logits_relative_l2_difference,
        "absolute_loss_difference": abs(full_value - second_value),
        "loss_close": loss_close,
        "gradient_atol": gradient_atol, "loss_atol": loss_atol, "rtol": rtol,
        "gradient_norm_rtol": gradient_norm_rtol,
        "lora_tensors": len(gradient_details),
        "max_gradient_absolute_difference": max(
            item["max_absolute_difference"] for item in gradient_details),
        "max_gradient_relative_l2_difference": max(
            item["relative_l2_difference"] for item in gradient_details),
        "overall_gradient_relative_l2_difference": (
            total_squared_difference / max(total_squared_full, 1e-24)) ** 0.5,
        "gradient_tensors_elementwise_close": sum(
            item["elementwise_close"] for item in gradient_details),
        "gradient_tensors_norm_close": sum(item["norm_close"] for item in gradient_details),
        "gradients_close": all(item["norm_close"] for item in gradient_details),
        "gradient_details": gradient_details,
    }
    record["status"] = "pass" if loss_close and record["gradients_close"] else "fail"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return record


def train(csv_path: Path, output_dir: Path, *, model_id: str = DEFAULT_MODEL,
          revision: str | None = None, device: str = "cuda:0", seed: int = 42,
          max_tokens: int = MAX_TOKENS, epochs: float = 3,
          learning_rate: float = 2e-4, accumulation: int = 16,
          probe_report: Path) -> dict[str, Any]:
    """Train only natural Full rows; save a LoRA adapter and resource metadata."""
    import torch
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    report = json.loads(Path(probe_report).read_text(encoding="utf-8"))
    if (report.get("status") != "success" or report.get("tokens") != max_tokens
            or report.get("loss_mode") != LOSS_MODE
            or report.get("model") != model_id or report.get("revision") != revision
            or report.get("device") != device or report.get("seed") != seed
            or report.get("csv_sha256") != file_sha256(Path(csv_path))
            or report.get("peak_reserved_bytes", float("inf")) > report.get("device_capacity_bytes", 0)):
        raise ValueError("A successful matching exact-length probe within GPU memory is required")
    rows = read_export(csv_path)
    model, tokenizer, dtype = load_model(model_id, revision, device, seed)
    examples, lengths = prepare_rows(tokenizer, rows, max_tokens)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(output_dir / "trainer"), num_train_epochs=epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=accumulation,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit", learning_rate=learning_rate, lr_scheduler_type="cosine",
        warmup_ratio=0.05, bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
        save_strategy="no", eval_strategy="no", report_to="none", remove_unused_columns=False,
        seed=seed, data_seed=seed)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model,
                                     label_pad_token_id=-100, return_tensors="pt")

    class ResponseOnlyTrainer(Trainer):
        """Use the same selective-logit loss as the disposable probe."""

        def __init__(self, *trainer_args, **trainer_kwargs):
            super().__init__(*trainer_args, **trainer_kwargs)
            self.model_accepts_loss_kwargs = True

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            return selective_response_loss(model, inputs,
                                           num_items_in_batch=num_items_in_batch,
                                           return_outputs=return_outputs)

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    trainer = ResponseOnlyTrainer(model=model, args=args, train_dataset=examples,
                                  data_collator=collator)
    outcome = trainer.train()
    metrics = {"model": model_id, "revision": revision, "seed": seed, "device": device,
               "token_cap": max_tokens,
               "loss_mode": LOSS_MODE,
               "attention_backend": model.config._attn_implementation,
               "optimizer": "paged_adamw_8bit",
               "quantization": "4-bit NF4 double quantization", "training_role": "Full",
               "evaluation": "none", "elapsed_seconds": time.perf_counter() - started,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
               "train_metrics": outcome.metrics, "lengths": lengths}
    model.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "adapter")
    (output_dir / "training_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


def probe(csv_path: Path, output_path: Path, *, model_id: str = DEFAULT_MODEL,
          revision: str | None = None, device: str = "cuda:0", seed: int = 42,
          target_tokens: int = MAX_TOKENS) -> dict[str, Any]:
    """Run one exact-length forward/backward/update and save metrics, never weights."""
    import torch

    record: dict[str, Any] = {"model": model_id, "revision": revision, "seed": seed,
                              "device": device, "target_tokens": target_tokens,
                              "csv_sha256": file_sha256(Path(csv_path)),
                              "attention_backend": "sdpa",
                              "optimizer": "PagedAdamW8bit", "quantization": "4-bit NF4 double quantization",
                              "gradient_checkpointing": True, "status": "started"}
    record["loss_mode"] = LOSS_MODE
    started = time.perf_counter()
    step_started: float | None = None
    try:
        import bitsandbytes as bnb

        # The probe discards the expert label and uses a dummy response.
        row = read_export(csv_path)[0]
        record["source_document"] = row["document"]
        model, tokenizer, _ = load_model(model_id, revision, device, seed)
        record["attention_backend"] = model.config._attn_implementation
        example = exact_probe_example(tokenizer, row, target_tokens)
        record["tokens"] = len(example["input_ids"])
        record["device_capacity_bytes"] = torch.cuda.get_device_properties(device).total_memory
        optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=2e-4)
        torch.cuda.reset_peak_memory_stats(device)
        step_started = time.perf_counter()
        model.train()
        batch = {key: torch.tensor([value], device=device) for key, value in example.items()}
        loss = selective_response_loss(model, batch)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        record.update(status="success", update_completed=True,
                      loss=float(loss.detach().cpu()))
    except Exception as error:
        record.update(status="failed", failure_type=type(error).__name__, failure=str(error))
    finally:
        finished = time.perf_counter()
        record["elapsed_seconds"] = finished - started
        record["step_seconds"] = None if step_started is None else finished - step_started
        if torch.cuda.is_available():
            record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            record["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
            if (record["status"] == "success" and record["peak_reserved_bytes"]
                    > record["device_capacity_bytes"]):
                record.update(status="failed", failure_type="GpuMemoryLimitExceeded",
                              failure="Peak reserved memory exceeded GPU capacity")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    return record


def main() -> None:
    """CLI for token audit, disposable probe, and explicitly requested training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "probe", "compare-loss", "train"))
    parser.add_argument("--csv", type=Path, default=DEFAULT_OUTPUT / "full_document_train.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--tokenizer", default=DEFAULT_MODEL,
                        help="Tokenizer path or model ID for audit; train/probe load the model tokenizer")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument("--row-index", type=int, default=0,
                        help="Zero-based Full CSV row for compare-loss")
    parser.add_argument("--second-loss-mode", choices=("selective", "full"),
                        default="selective", help="Second pass for compare-loss")
    parser.add_argument("--probe-report", type=Path,
                        help="Required matching successful probe JSON for training")
    args = parser.parse_args()
    if args.revision is not None and args.revision.strip().upper() == "REVISION":
        parser.error("REVISION is a placeholder; omit --revision or supply a real branch, tag, or commit ID")
    if args.mode == "audit":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision)
        _, report = prepare_rows(tokenizer, read_export(args.csv), args.max_tokens)
        print(json.dumps(report, ensure_ascii=True, indent=2))
    elif args.mode == "probe":
        result = probe(args.csv, args.output_dir / "probe.json", model_id=args.model,
                       revision=args.revision, device=args.device, seed=args.seed,
                       target_tokens=args.max_tokens)
        print(json.dumps(result, ensure_ascii=True, indent=2))
        if result["status"] != "success":
            raise SystemExit(1)
    elif args.mode == "compare-loss":
        result = compare_loss_gradients(
            args.csv, args.output_dir / "loss_comparison.json", model_id=args.model,
            revision=args.revision, device=args.device, seed=args.seed,
            row_index=args.row_index, second_loss_mode=args.second_loss_mode)
        summary_keys = ("status", "tokens", "supervised_tokens", "full_loss",
                        "second_loss", "absolute_loss_difference", "loss_close",
                        "lora_tensors", "overall_gradient_relative_l2_difference",
                        "max_gradient_relative_l2_difference",
                        "gradient_tensors_elementwise_close",
                        "gradient_tensors_norm_close", "gradients_close")
        print(json.dumps({key: result[key] for key in summary_keys}, indent=2))
        if result["status"] != "pass":
            raise SystemExit(1)
    else:
        if args.probe_report is None:
            parser.error("train requires --probe-report")
        result = train(args.csv, args.output_dir, model_id=args.model, revision=args.revision,
                       device=args.device, seed=args.seed, max_tokens=args.max_tokens,
                       epochs=args.epochs, learning_rate=args.learning_rate,
                       accumulation=args.accumulation, probe_report=args.probe_report)
        print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
