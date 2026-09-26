# LoRA experiments

`series_00_legacy_adapters` contains historical recovery and evaluation tools. The completed `lora_coordinate_balanced_val` run and its adapters, checkpoints, hashes, and reports remain historical evidence in ignored local folders. The fresh search uses the separate ID `lora_coordinate_imbalanced_val` and never imports those checkpoints.

## Fresh coordinate search

Run these commands from the repository root in the listed order, with the local Python environment and GPU dependencies installed. Prepare `local/data/classification/train.csv`, the imbalanced `local/data/classification/val.csv`, the paired `Full` benchmark under `local/data/benchmarks/autotest/Full` and `local/data/benchmarks/test_docx/Full`, and the pinned `dms-rag` repository and index. Set `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` for Llama access.

First inspect the initial candidate matrix:

```bash
python -m LCD.experiments.lora.series_01_coordinate_search.target_modules --dry-run
```

Then run each stage. **Repeat the current command** whenever it prints `paused`; move to the next command only after it prints `complete`. Each invocation attempts at most four model runs by default, so completing a stage normally requires several invocations. Saved results are skipped, interrupted or failed runs are retried, and a stage finishes only when every model candidate has valid validation and test scores.

```bash
python -m LCD.experiments.lora.series_01_coordinate_search.target_modules
python -m LCD.experiments.lora.series_01_coordinate_search.rank
python -m LCD.experiments.lora.series_01_coordinate_search.learning_rate
python -m LCD.experiments.lora.series_01_coordinate_search.alpha
python -m LCD.experiments.lora.series_01_coordinate_search.dropout
python -m LCD.experiments.lora.series_01_coordinate_search.compare_llm
```

Repeat `compare_llm` if it pauses. The comparison starts only after dropout completes. Each command uses all four models (`qwen`, `llama`, `ministral`, `t-lite`) and the ternary task by default. To change the per-invocation cap, add `--nruns N` (for example, `--nruns 1`); `nruns` is also a keyword parameter on the Python entrypoint functions. Failed attempts and automatic retries count toward this cap. `--max-retries` changes the retries within a sweep invocation.

Sweep stages retry a failed recipe once immediately by default. When investigating a failure, use `--max-retries 0` to stop after the first failure; rerunning later uses the saved adapter if its manifest matches the recipe. Review ZIP workbooks are assembled in memory, and long ZIP filenames are shortened to fit the Windows path limit.

When a later stage repeats the same training settings, it reuses a completed adapter from this search. If that earlier adapter's validation hash differs from the later recipe's recorded hash, the adapter is scored again on the current validation file. Reuse is limited to runs that did not select their final checkpoint using validation. The later recipe receives its own validation and `Full` scores; the original adapter and its manifest remain in place.

Training uses `train.csv`; model validation and stage winner selection use only `val.csv`. The winner is chosen by validation **contradiction F1**, then fewer invalid predictions, then grid order. Test scoring uses only the `Full` dataset and its `autotest_model` scope; it does not choose winners. `autotest_total` is excluded because premise retrieval is fixed in this search.

Progress is saved under `local/experiments/lora/series_01_coordinate_search/lora_coordinate_imbalanced_val/results/search_state.json`; stage workbooks are in its `stages/<stage>/results.xlsx` folders. Rerun with the same search ID, code, inputs, model set, and training settings to resume. You may change `--nruns` or `--max-retries` between invocations. If any locked input or training setting changes, use a new `--search-id` for all stages of the new search.

## Revalidate a saved adapter

Evaluate an existing adapter on a changed validation CSV without resuming training or changing the original search reports:

```bash
python -m LCD.experiments.lora.series_01_coordinate_search.validate_saved_adapter \
  --adapter-dir local/experiments/lora/series_01_coordinate_search/lora_coordinate_imbalanced_val/artifacts/experiments/2ab5410c41de3378ff27/models/lora/Qwen_Qwen3-8B/ternary \
  --val-path local/data/classification/val.csv \
  --output-dir local/experiments/lora/series_01_coordinate_search/lora_coordinate_imbalanced_val/results/revalidation
```

Use the local GPU Python environment. The command uses the saved base-model revision and inference settings and writes a workbook with scores and row-level predictions.
