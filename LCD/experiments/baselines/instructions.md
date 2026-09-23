# Baselines and comparisons

`series_01_classifiers` contains the three-class BERT, untuned-LLM, and LoRA baseline campaign and the full-pipeline evaluator. Use `python -m LCD.experiments.baselines.series_01_classifiers.run_campaign --help` or `python -m LCD.experiments.baselines.series_01_classifiers.full_pipeline_evaluation --help`. New runs use `local/data/classification/train.csv` and `val.csv`, the local benchmark `Full` folder, and explicit ignored output directories. Historical reports are preserved under the matching ignored series directories.

`series_02_lora_rag_comparison` contains a local legacy-adapter evaluator and a compact result compiler. The evaluator requires an explicit three-class `--models-source` artifact tree or JSON model list and can use the local `rag-qwen` bundle. It does not mount or read cloud storage. Historical mixed reports remain read-only evidence; new plots and workbooks contain only three-class scores.

`series_03_qwen38` holds the Qwen3.8 model-list configuration for the full-pipeline evaluator. Supply its `models.json` through `--models-source` and configure the local llama.cpp service with the evaluator's CLI options. Keep model weights and service artifacts outside tracked source.
