# Baselines and comparisons

These runnable workflows are retained for historical reproduction and comparison.
The current classifier goal is the separate [8k LoRA experiment](../classifiers/README.md).

`series_01_classifiers` contains the three-class BERT, untuned-LLM, and LoRA baseline campaign and the full-pipeline evaluator. Use `python -m LCD.experiments.baselines.series_01_classifiers.run_campaign --help` or `python -m LCD.experiments.baselines.series_01_classifiers.full_pipeline_evaluation --help`. New runs use `local/data/classification/train.csv` and `val.csv`, the local benchmark `Full` folder, and explicit ignored output directories. Historical reports are preserved under the matching ignored series directories.

## Full-context LoRA baseline retraining

The [retraining experiment](series_01_classifiers/README.md) is separate from
the historical campaign. From the repository root, inspect the four planned
jobs and then run the retraining command when GPU training is desired:

```bash
python -m LCD.experiments.baselines.series_01_classifiers.retrain_lora --dry-run
python -m LCD.experiments.baselines.series_01_classifiers.retrain_lora
```

Use `--models qwen` (or any subset) to train one job at a time. Repeating the
same command skips completed jobs and resumes evaluation from a compatible
saved adapter. The default run ID is `baseline_full_context_v1`; outputs are
under `local/experiments/baselines/series_01_classifiers/lora_retraining/`.
Keep a new `--run-id` for any changed input, revision, or training setting.
Use `--repo-root`, `--train-path`, `--val-path`, `--rag-dir`, `--rag-revision`,
`--autotest-dir`, `--test-docx-dir`, or `--output-root` to change paths.
`--revisions-json` and `--hyperparameters-json` accept JSON objects for model
revisions and shared LoRA parameter overrides. The default recipe has
`max_seq_length=2560`, `batch_size=1`, and `gradient_accumulation_steps=16`.
Overlong rows fail instead of truncating.

After all four jobs finish, compile the descriptive comparison:

```bash
python -m LCD.experiments.baselines.series_01_classifiers.compare_lora_retraining
```

The comparison reads saved `Full/autotest_model` scores from the new run,
`full_pipeline_evaluation_baseline`, `campaigns/results/full_pipeline_v1`, and
the balanced coordinate search's dropout-stage winners. It writes a CSV and
workbook under the new run's `results/comparison/` folder. Pass
`--new-results`, `--legacy-scores`, `--campaign-scores`, `--grid-scores`,
`--grid-state`, or `--output-dir` to use other ignored local locations.
The historical campaign has 1,607 scored pairs, whereas the later saved
reports have 1,207. The workbook labels each cohort and its support count;
it does not infer improvement across those different pair sets.

`series_02_lora_rag_comparison` contains a local legacy-adapter evaluator and a compact result compiler. The evaluator requires an explicit three-class `--models-source` artifact tree or JSON model list and can use the local `rag-qwen` bundle. It does not mount or read cloud storage. Historical mixed reports remain read-only evidence; new plots and workbooks contain only three-class scores.

`series_03_qwen38` holds the Qwen3.8 model-list configuration for the full-pipeline evaluator. Supply its `models.json` through `--models-source` and configure the local llama.cpp service with the evaluator's CLI options. Keep model weights and service artifacts outside tracked source.
