# LoRA coordinate search

This historical experiment studied how much adaptation a base language model needs to classify a court ruling and a lawbook provision as `contradiction`, `entailment`, or `not mentioned`. It varied where and how strongly LoRA changed the model, then compared the selected adapter with its untuned base model. Further grid search is abandoned as a current direction; the scripts and results remain archived as historical evidence.

The historical search covered Qwen3-8B, Llama-3.1-8B, Ministral-8B-Instruct-2410, and T-lite-it-2.1. It was a **staged coordinate search**: each stage tested a small grid for one setting and passed its winner to the next stage. It did not evaluate every combination of all settings. Identical training settings reused an adapter from an earlier stage of the same search; each validation-data version received its own scores. Winners were selected separately for each model using validation contradiction F1; ties favored fewer invalid predictions, then grid order.

| Stage | Candidates | What it probes |
| --- | --- | --- |
| Target modules | Query and value projections (`q_proj`, `v_proj`); query, key, and value projections; or all linear layers | How broadly the adapter must modify the base model. |
| Rank | 8, 16, 32; alpha was held at twice the rank | The adapter's capacity while keeping its scaling rule fixed. |
| Learning rate | `1e-5`, `2e-5`, `1e-4`, `2e-4` | How large the training updates should be. |
| Alpha | One or two times the winning rank | The strength of the LoRA update relative to the base weights. |
| Dropout | 0, 0.05, 0.1 | Whether regularizing the adapter improves validation performance. |

Training used `train.csv`; stage selection used only `val.csv`. The `Full` benchmark's `autotest_model` scope provided a separate test score and did not select winners. The saved comparison evaluated each winning LoRA adapter against its matching untuned base model with the same inference settings.

The [data-creation experiment](../../data_creation/README.md) now defines the new expert-reviewed splits. The [8k classifier experiment](../../classifiers/README.md) is the current LoRA goal; it does not resume this search or import its adapters.

The original scripts are under `.legacy/lora_experiments/series_01_coordinate_search/`; [LoRA instructions](../instructions.md) explain the archive. Their adapters remain historical evidence and are not reused in the planned 8k training.
