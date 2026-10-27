# Series 01: portable NLI and RAG datasets

This series builds new `train`, `val`, and `test` datasets from the 1,000 reviewed
original rulings in `local/data/classification/dataset0929/decisions/main`. The
split is frozen at 600/200/200 documents before hypotheses are extracted. A
source decision, every copy derived from it, and every resulting pair always
remain in the same split.

Hypotheses come from the final `ПОСТАНОВИЛ` section, split with `razdel` and
filtered with an audited dataset-specific filter. Explicit КоАП citations are
resolved before semantic retrieval from the October 2026 CodeRAG index. Each
hypothesis normally receives ten unique premises, with a hard limit of twenty.
Context is retrieved separately for each hypothesis-premise pair from the active
court-decision text with Qwen embeddings and this query instruction:

> Given a legal claim and a provision of Russian law, retrieve sentences from
> the same court decision that provide the factual and procedural context needed
> to interpret the claim and assess its consistency with the provision.

The five best distinct sentences are retained, excluding the hypothesis itself,
and serialized in document order. Fewer are used only when the document has
fewer eligible sentences.

## Balance and annotation policy

The complete labeled pair pool is retained. Training code uses weights rather
than deleting useful pairs: NLI sampling targets 50% `contradiction`, 25%
`entailment`, and 25% `not mentioned`; RAG sampling targets 50% `related` and
50% `unrelated`. Relevance is judged independently of the NLI label, so a
relevant but inconclusive pair is not silently turned into an unrelated negative.

Training may contain up to two distinct synthetic variants of an eligible
hypothesis, and a source document may contribute several modified hypotheses.
Validation targets about 120 contradiction-bearing documents; test targets about
40. Validation and test use at most one active variant and one target scenario
per source document. Naturally occurring contradictions count toward those
targets. The expert percentages are relative targets among feasible
contradiction types rather than reasons to force an incorrect label.

Only rows whose normalized value in `Добавляем в текущую выборку (октябрь 2026)`
is exactly `Да` are eligible. The expert scenario and the NLI judgment remain
separate: a valid control, malformed citation, or internal document inconsistency
does not automatically contradict the retrieved premise. Uncertain cases remain
pending and do not enter a released CSV.

## Portable split bundles

Each split is self-contained and can be moved without the original DOCX corpus:

```text
train/
  train.csv
  metadata.jsonl
  sampling_weights.csv
  manifest.json
  decisions/
    originals/<source_id>.txt
    variants/<variant_id>.txt
```

The same structure is used for `val` and `test`. Every CSV row has the columns
`row_id,hypothesis,premise,article_number,context,label,error_type,source_filename,decision_text_path`.
`row_id` is assigned from a persistent registry, is never renumbered, and is
never reused after an expert deletes the row. `decision_text_path` is relative
to the split directory and identifies the exact original or edited UTF-8 text
used to build that row.

Original rulings are never edited. A subagent proposes minimal offset-based edit
operations; the coordinator applies them to a new text copy, then rebuilds
hypotheses, premises, and context. All synthetic examples and all validation and
test pairs require human review. Training additionally requires review of every
uncertain pair and a stratified sample of the remaining pairs.

Independent verification records evidence without granting human approval. An
accepted annotation remains `proposed`; a rejected or uncertain annotation
returns to pending review. The pilot verifier found both a residual bank-detail
fragment and a compound hypothesis whose premise supported only part of the
claim. Consequently, entailment requires support for every material atom of a
compound hypothesis; a premise/context-supported incompatibility in any material
atom makes the pair contradictory, while partial support without incompatibility
is `not mentioned`. Filtering residues are flagged for extraction review.

Generation assignments are coordinator-owned and persisted. Each generation
task supplies exactly one original hypothesis and one selected eligible rule;
the subagent does not select either. Relative rule quotas determine assignments,
subject to the train two-variant limit and the validation/test one-source limit.
An infeasible proposal remains `needs_review` and a later assignment fills the
reported shortage from an unused candidate. After applying any accepted variant,
run preparation again for its split so every pair and context refers to the new
copied text.

Commands and review procedure are documented in the [data-creation
instructions](../instructions.md). The generation, annotation, and verification
contracts are in [`prompts/`](prompts/).
