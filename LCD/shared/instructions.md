# Shared workflows

Install the project in the native local Python environment with `python -m pip install -e ".[ml]"`; install the test dependency with `python -m pip install "pytest>=8,<10"` or use `uv sync --extra ml --dev`. Run commands from the repository root or install the package editable before running them elsewhere. Model downloads and GPU initialization happen only when a workflow is called.

Private classification inputs belong in `local/data/classification/`. `train.xlsx` and `val.xlsx` are three-class sources; `train.csv` is the active training CSV. `val_balanced.xlsx` and `val_balanced.csv` are retained separately. The active `val.csv` starts with every row of `val_balanced.csv` and appends the checked `not mentioned` pairs from `val_added_for_human_expert_done.xlsx`. Rebuild it from the repository root with `python -m LCD.shared.rebuild_validation`; pass `--balanced`, `--expert`, `--output`, or `--provenance` to change paths. The active provenance file is `val_expert_provenance.csv`, which records the workbook row and source document for each addition. The earlier `augment_val_not_mentioned.py`, `val_added_provenance.csv`, and `val_excluded_fragments.csv` record candidate generation before expert review. The source workbooks can be converted with `python -m LCD.shared.conversion --train local/data/classification/train.xlsx --val local/data/classification/val.xlsx --output-dir local/data/classification`, but this overwrites the active reviewed `val.csv` and changes its experiment input hash.

Audit validation and reviewed Full examples against training with `python -m LCD.shared.dataset_similarity`. The default output under `local/analysis/train_val_full_neighbors/` contains a row-level nearest-neighbour CSV, a summary by split and label, and a method JSON. Use `--train`, `--val`, `--full-dir`, `--output-dir`, or `--batch-size` to change inputs or output. Scores are character 3–5 gram TF-IDF cosine, fitted on train only; pair scores average premise and hypothesis similarity to one training row. Exact matches use Unicode NFKC, case folding, and whitespace collapse.

The full pipeline extracts sentences after the last `ПОСТАНОВИЛ` marker, resolves citations before semantic fallback, and writes row-level source and prediction evidence. Use the baseline group's full-pipeline runner with explicit `--models-source`, `--rag-source`, and `--results-dir` paths. Model lists must contain three-class jobs. New evaluations select the `Full` benchmark by default; pass another dataset explicitly only for historical reproduction.

The environment token is read from `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` only when needed. Pin model and corpus revisions in run metadata. Keep generated workbooks, checkpoints, and input documents under ignored local directories.

Shared LoRA training uses complete prompt-plus-label examples with a default
2,560-token context, a microbatch of one, and 16 gradient accumulation steps.
It raises an error identifying any row that exceeds `max_seq_length`; adjust
the configurable limit and memory settings for other datasets. The baseline
group's [LoRA run instructions](../experiments/baselines/instructions.md)
describe the new four-model comparison workflow.
