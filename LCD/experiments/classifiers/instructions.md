# Classifier instructions

Run from the repository root with a local Python interpreter. The commands below prepare the [Qwen3-8B full-document experiment](series_01_qwen3_full_document/README.md); the export and audit do not load a model onto the GPU.

```bash
python -m LCD.experiments.classifiers.series_01_qwen3_full_document.data \
  --workbooks local/data/benchmarks/autotest/Full \
  --documents local/data/benchmarks/test_docx/Full \
  --output-dir local/experiments/classifiers/series_01_qwen3_full_document

python -m LCD.experiments.classifiers.series_01_qwen3_full_document.training audit \
  --csv local/experiments/classifiers/series_01_qwen3_full_document/full_document_train.csv \
  --tokenizer local/experiments/baselines/series_01_classifiers/lora_retraining/baseline_full_context_v1/artifacts/qwen/models/lora/Qwen_Qwen3-8B/ternary \
  --max-tokens 8192
```

Inspect `manifest.json` and confirm that the audit reports 1,236 rows, a maximum at or below 8,192, and the expected context truncation count. Defaults for source paths and output paths are built into the entrypoints. `--tokenizer` may also name a model ID. The saved local tokenizer is convenient for offline audit and does not import an old adapter or checkpoint.

Run the disposable exact-length update **before** training, or use an existing matching successful report:

```bash
python -m LCD.experiments.classifiers.series_01_qwen3_full_document.training probe \
  --model Qwen/Qwen3-8B --device cuda:0 --seed 42 \
  --max-tokens 8192 \
  --output-dir local/experiments/classifiers/series_01_qwen3_full_document/probe_selective
```

The probe writes only `probe.json`. Check `status: success`, `loss_mode: selective_response_logits_v1`, `tokens: 8192`, peak allocated and reserved memory below the device capacity, and step time. It uses a dummy label and saves no weights. Failures are recorded in the same JSON file. The completed selective-logit probe passed this gate with 18.7 GB peak reserved memory; the earlier full-logit probe failed it.

To compare the two response-loss implementations on one natural pair without its document context, run:

```bash
python -m LCD.experiments.classifiers.series_01_qwen3_full_document.training compare-loss \
  --model Qwen/Qwen3-8B --device cuda:0 --seed 42 --row-index 0 \
  --output-dir local/experiments/classifiers/series_01_qwen3_full_document/loss_comparison
```

This makes two forward and backward passes on the same loaded QLoRA model, restores the random state between passes, and writes `loss_comparison.json`; it does not update or save weights. The report includes both losses and per-tensor LoRA gradient differences. `status: pass` requires the losses to agree within `atol=1e-4, rtol=1e-3` and every gradient tensor's relative L2 difference to be at most 2%, the BF16 comparison tolerance. Use `--second-loss-mode full` and a separate output directory to check exact same-path reproducibility. The first pair passed the BF16 norm comparison; see the [experiment description](series_01_qwen3_full_document/README.md) for the measured differences.

If training is explicitly requested, use natural rows:

```bash
python -m LCD.experiments.classifiers.series_01_qwen3_full_document.training train \
  --model Qwen/Qwen3-8B --device cuda:0 --seed 42 \
  --max-tokens 8192 --epochs 3 --learning-rate 0.0002 --accumulation 16 \
  --probe-report local/experiments/classifiers/series_01_qwen3_full_document/probe_selective/probe.json \
  --output-dir local/experiments/classifiers/series_01_qwen3_full_document/run_01
```

The commands use the model's default revision. To pin a revision, add `--revision` followed by a real branch, tag, or commit ID to **both** the probe and training commands. Do not pass the word `REVISION` literally. The trainer requires a successful selective-logit probe report for the same CSV, model, revision, seed, device, and token cap. It saves `adapter/` and `training_metrics.json` under the output directory. No `Full` quality score is produced, because the same rows are used for training. Do not use repeated probe context as training data. The recipe starts with built-in SDPA; do not install `flash-attn` for this initial run.
