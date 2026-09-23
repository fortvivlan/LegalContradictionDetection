# Shared workflows

Install the project in the native local Python environment with `python -m pip install -e ".[ml]"`; install the test dependency with `python -m pip install "pytest>=8,<10"` or use `uv sync --extra ml --dev`. Run commands from the repository root or install the package editable before running them elsewhere. Model downloads and GPU initialization happen only when a workflow is called.

Private classification inputs belong in `local/data/classification/`. `train.xlsx` and `val.xlsx` are three-class sources; `train.csv` and `val.csv` are their active CSV inputs. `val_balanced.xlsx` and `val_balanced.csv` are retained separately. Convert the primary workbooks with `python -m LCD.shared.conversion --train local/data/classification/train.xlsx --val local/data/classification/val.xlsx --output-dir local/data/classification`. This validates both workbooks before writing either CSV. Do not regenerate the migrated CSVs unless intentionally changing the experiment input hashes.

The full pipeline extracts sentences after the last `ПОСТАНОВИЛ` marker, resolves citations before semantic fallback, and writes row-level source and prediction evidence. Use the baseline group's full-pipeline runner with explicit `--models-source`, `--rag-source`, and `--results-dir` paths. Model lists must contain three-class jobs. New evaluations select the `Full` benchmark by default; pass another dataset explicitly only for historical reproduction.

The environment token is read from `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` only when needed. Pin model and corpus revisions in run metadata. Keep generated workbooks, checkpoints, and input documents under ignored local directories.
