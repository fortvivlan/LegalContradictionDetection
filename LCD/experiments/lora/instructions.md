# Historical LoRA experiments

The old adapter recovery, emergency retraining, and coordinate-search scripts
are archived locally under `.legacy/lora_experiments/`. Their original run
instructions are preserved there. The archived scripts and saved runs used
historical training behavior and are not entrypoints for new training.

The balanced and imbalanced search results, checkpoints, and reports remain in
ignored `local/experiments/lora/series_01_coordinate_search/`. Further grid
search is abandoned. The current [8k classifier experiment](../classifiers/README.md)
has no runner yet; the older [baseline retraining workflow](../baselines/instructions.md)
is retained for historical reproduction.
