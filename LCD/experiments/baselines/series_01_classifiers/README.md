# Full-context LoRA baseline retraining

This series reruns the four ternary baseline LoRAs with the original baseline
adapter recipe and complete training examples. The shared trainer now allows
2,560 tokens, uses a microbatch of one with 16 accumulation steps, and errors
if a prompt, source-prefixed premise, hypothesis, or response would be cut.
Model and RAG revisions are pinned; inputs and outputs can be overridden.

The retraining runner writes new adapters and scores under a separate ignored
run directory. It evaluates `val.csv` and the `Full` benchmark's
`autotest_model` scope. Validation scores describe the new run but do not
select a winner. A separate compiler compares these scores with the historical
incorrect-loss adapters, the original `full_pipeline_v1` campaign, and the
balanced coordinate-search winners. The campaign's saved `Full` score has
1,607 pairs; the two later historical reports have 1,207, so the comparison
shows support counts and avoids cross-cohort improvement claims.

Commands and input paths are in the [baseline instructions](../instructions.md).
