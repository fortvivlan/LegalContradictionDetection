# Classifier instructions

The 8k LoRA experiment is documented but has no new CLI yet. The current shared trainer's implementation and 2,560-token defaults are described in [shared instructions](../../shared/instructions.md). Run future local Python entrypoints only after they exist; keep model weights, probe logs, checkpoints, and results under ignored `local/` paths.

## Feasibility gate

1. For each of Llama-3.1-8B, Ministral-8B-Instruct-2410, Qwen3-8B, and T-lite-it-2.1, follow a source link in the current `Full` benchmark to obtain the original full-document text. Compose a representative pair prompt with document context and a dummy response label, without reading or using its held-out expert label. Verify that the complete prompt and response tokenize to exactly 8,192 tokens with the model's own tokenizer; do not silently truncate.
2. Run a disposable 4-bit LoRA training step at microbatch one: forward, backward, then optimizer update. Measure peak allocated and reserved GPU memory and step time. Start from the corrected shared trainer configuration; test attention and activation-memory settings if the baseline step exceeds 24 GB. Record failures and the setting that fits.
3. Delete the probe's updated model and optimizer state when the process ends. Do not save an adapter, evaluate prediction quality, select hyperparameters by `Full` labels, or mix Full examples into later training.

The gate passes only when each named model completes a real 8,192-token update within 24 GB. Record per-model outcomes even if some fail; a partial pass is not a four-model success.

## Later classifier training

Once data creation has released the new expert-reviewed splits, build the production 8k trainer around their `train` and `dev` data. The rebuilt `test` split stays held out until final evaluation. Keep model revisions, seeds, token limits, device, output locations, and the selected memory recipe configurable. No production training command or result is claimed here.
