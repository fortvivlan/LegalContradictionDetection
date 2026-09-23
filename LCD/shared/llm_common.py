"""Shared causal-LLM loading and batched prediction helpers."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Sequence

from .common import ModelSpec, Task
from .inference import ModelPrediction
from .prompting import build_generation_prompt, parse_generated_label


@dataclass
class LoadedCausalModel:
    """A loaded model/tokenizer pair plus resolved precision metadata."""

    model: Any
    tokenizer: Any
    compute_dtype: str


def require_cuda():
    """Return torch after verifying that a CUDA GPU is available."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA-capable local Python environment is required."
        )
    return torch


def load_causal_model(
    spec: ModelSpec,
    *,
    revision: str | None,
    token: str | None,
    quantization: bool = True,
    device_map: Any = "auto",
    precision: str = "auto",
) -> LoadedCausalModel:
    """Load a supported causal LM with configurable 4-bit quantization."""
    torch = require_cuda()
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    normalized_precision = precision.strip().lower()
    if normalized_precision not in {"auto", "bfloat16", "float16"}:
        raise ValueError("precision must be 'auto', 'bfloat16', or 'float16'")
    use_bf16 = normalized_precision == "bfloat16" or (
        normalized_precision == "auto" and bool(torch.cuda.is_bf16_supported())
    )
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("bfloat16 was requested but is unsupported by this GPU")
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_id,
        revision=revision,
        token=token,
        trust_remote_code=spec.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    kwargs: dict[str, Any] = {
        "revision": revision,
        "token": token,
        "trust_remote_code": spec.trust_remote_code,
        "device_map": device_map,
        "dtype": compute_dtype,
    }
    if quantization:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(spec.model_id, **kwargs)
    model.eval()
    return LoadedCausalModel(
        model=model,
        tokenizer=tokenizer,
        compute_dtype="bfloat16" if use_bf16 else "float16",
    )


def model_input_device(model: Any):
    """Find the input device for regular and accelerate-dispatched models."""
    try:
        return model.get_input_embeddings().weight.device
    except AttributeError:
        return model.device


class CausalPredictor:
    """Deterministic batched predictor for one task prompt."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        task: Task,
        *,
        batch_size: int = 1,
        max_input_length: int = 4096,
        max_new_tokens: int = 16,
        prompt_text: str | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.task = task
        self.batch_size = batch_size
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens
        self.prompt_text = prompt_text

    def predict_examples(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
        *,
        progress_description: str | None = None,
    ) -> list[ModelPrediction]:
        """Predict aligned examples, optionally displaying batch progress."""
        if len(premises) != len(hypotheses):
            raise ValueError("Premises and hypotheses must have equal length")
        torch = require_cuda()
        prompts = [
            build_generation_prompt(
                self.tokenizer,
                premise,
                hypothesis,
                self.task,
                prompt_text=self.prompt_text,
            )
            for premise, hypothesis in zip(premises, hypotheses)
        ]
        predictions: list[ModelPrediction] = []
        input_device = model_input_device(self.model)
        batch_starts = range(0, len(prompts), self.batch_size)
        if progress_description is not None:
            from tqdm.auto import tqdm

            batch_starts = tqdm(
                batch_starts,
                desc=progress_description,
                unit="batch",
            )
        for start in batch_starts:
            batch = prompts[start : start + self.batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_length,
            ).to(input_device)
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
            decoded = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            predictions.extend(
                ModelPrediction(parse_generated_label(text, self.task), text)
                for text in decoded
            )
        return predictions

    def predict_pairs(
        self, premises: Sequence[str], hypothesis: str
    ) -> list[ModelPrediction]:
        """Predict several premises against one hypothesis."""
        return self.predict_examples(premises, [hypothesis] * len(premises))


def release_gpu_objects(*objects: Any) -> None:
    """Drop references supplied by callers and clear CUDA allocator caches."""
    del objects
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ModuleNotFoundError:
        pass
