# Data-creation instructions

Expert-selected articles are now available for document collection. Contradiction-modification rules remain an expert input for later dataset construction. Do not substitute agent-chosen articles or rules.

## Collection scope and sources

- Main article group: `18.8`, `18.9`, `18.10`, `18.11`, `18.12`, `18.15`, `18.16`, `18.17`, `18.18`, `18.19`, `18.20`, `19.27` КоАП РФ. Collect at least 1,000 distinct original judicial rulings across this group, seeking broad coverage and at least roughly ten per article where published decisions permit.
- Domain-shift group: `20.20`, `20.21`, `6.9` КоАП РФ. Collect about 30 original rulings, aiming at roughly ten per article. Keep this pool separate as test candidates; do not place its cases into train or dev.
- Collect final decisions on complaints separately for **both** groups, aiming for at least 30 verified appeals per article where published decisions permit and at least one for each article part where available. An appeal is evidence for expert review, not an original ruling or an automatically labeled contradiction.

Available public sources include [ГАС «Правосудие»](https://sudrf.ru/), which links to federal general-jurisdiction courts and magistrates; the [Moscow courts search](https://mos-gorsud.ru/mgs/search), which includes administrative-offence cases and review proceedings; [SudAct general courts](https://sudact.ru/regular/) and [SudAct magistrates](https://sudact.ru/magistrate/); and [ZakonRF's court-act archive](https://www.zakonrf.info/gorsud/). The current automated collectors use SudAct. Check official court portals and other public archives for coverage gaps and provenance confirmation where feasible. Record the exact document URL and source for every saved item. Search result titles, case-number prefixes, and page headings are discovery hints only; acceptance depends on the full published text. Use polite request intervals, bounded retries, and resume from saved provenance.

Store collected material under ignored `local/data/classification/dataset0929/decisions/` and `local/data/classification/dataset0929/appeals/`. Within each, keep `main/` and `domain_shift/` separate, then an article-number folder containing DOCX copies. Retain the original published HTML or text and a provenance manifest with retrieval time, source URL, article, court, case identifier, document type, content hash, and the passages supporting acceptance. Record inaccessible pages and shortages; never fill a target with duplicates, incomplete documents, or a different article.

Run the local collectors from the repository root with the project Python environment:

```bash
./.venv/bin/python -m LCD.experiments.data_creation.collect_originals --group all \
  --count-main 1000 --count-domain-shift 30 --minimum-per-article 10
./.venv/bin/python -m LCD.experiments.data_creation.collect_appeals --group all \
  --target 30 --max-pages 8
./.venv/bin/python -m LCD.experiments.data_creation.audit_collection
```

Both commands are resumable against their provenance manifests. Use `--help` for search depth, timeouts, request intervals, year or article selection, and output paths. The DOCX files are text copies of published acts; the corresponding HTML and plain-text files preserve the fetched source. Inspect `provenance.csv` and `failures.csv` before expert review. A shortfall is a source-availability or retrieval result, not permission to include a document that fails the checks below.

Run appeal searches into a separate staging directory. Before copying any candidate into the active `appeals/` collection, inspect its published opening, charged original ruling, complaint, substantive reasoning, and final operative ruling. Confirm that the *same* article and part govern the appealed ruling; dates such as `18.11.2025`, case numbers such as `3/18.19-33/2025`, historical citations, and source typos do not establish the target article. Record per-document acceptance evidence or rejection reasons. Keep missing parts and per-article shortfalls visible in the review report.

## Collection and expert review

1. The original-rulings collector finds first-instance *постановления о привлечении к административной ответственности* under the selected articles. The published text must identify a КоАП РФ case, apply the target article in the final `ПОСТАНОВИЛ` section, and give an actual disposition such as guilt and penalty. Reject complaint reviews even when titled `Постановление`. The Chapter 30 complaint requirements below do not apply to original rulings. Record a stable source URL, retrieval date, document identifier, and full text or a local copy for expert review.
2. Subagents separately collect final court decisions on complaints against administrative-offence rulings under Chapter 30 of the КоАП РФ. The appeal pool supports expert analysis of errors and does not become the source-document corpus. Record any verified link to an original ruling, but matching is not required.
3. Subagents document the text passages supporting every appeal's inclusion. Human experts review the candidate rulings and appeals, and select corpus rulings before splitting them.

The expert appeal-search instructions are:

> **Задание для поиска.** Найти и собрать судебные акты РФ по делам об административных правонарушениях, в которых суд рассмотрел жалобу на постановление по делу об административном правонарушении в порядке главы 30 КоАП РФ. Нужны именно итоговые акты пересмотра постановлений: решения по жалобам на постановления, а не исходные постановления о привлечении к ответственности.
>
> Включать документы, в которых одновременно есть все признаки:
>
> 1. Прямо указано, что дело рассматривается по КоАП РФ / в порядке главы 30 КоАП РФ.
> 2. Есть обжалуемое постановление по делу об административном правонарушении: должностного лица, административного органа, комиссии или суда.
> 3. Суд рассматривает жалобу на это постановление.
> 4. В документе содержится итог рассмотрения жалобы: постановление оставлено без изменения, жалоба — без удовлетворения; постановление изменено; постановление отменено, производство прекращено; постановление отменено, дело возвращено на новое рассмотрение; либо постановление отменено, дело направлено по подведомственности.
>
> **Ключевое правило:** не доверять заголовку документа. Нужно по тексту подтвердить три вещи: есть постановление по КоАП, есть жалоба именно на него, и есть итоговое решение по этой жалобе.

## Apply the expert exclusions to original rulings

`review_decisions` defaults to `decisions/main/`; `domain_shift/` and appeals remain separate. It reads the DOCX text, uses the final `ПОСТАНОВИЛ` section and the existing `razdel` splitter, and preserves the indices removed by the irrelevant-text filter. Federal-law checks deliberately precede that filter because the experts also exclude payment boilerplate citing independent federal laws. A named/numbered law in the reasoning alone does not meet this exclusion. Embedded references in a cited КоАП provision are recorded as allowed contexts; another independent federal-law citation in the same document still excludes it.

The default `--ip-scope party` follows the clarified expert instruction: only the entrepreneur being prosecuted triggers criterion б. Check the opening and operative subject; an entrepreneur employer, witness, landlord, or generic statutory quotation does not suffice. `--ip-scope any` is available only for a deliberately broader review. Explicit expert exclusions are enforced for all eight supplied filenames, including the two documents with a detached departure explanation. Analogous uncited departure obligations are also checked. A payment warning or hypothetical future expulsion does not establish an imposed expulsion.

Create the report in a **new directory outside `decisions/`**:

```bash
./.venv/bin/python -m LCD.experiments.data_creation.review_decisions \
  --root local/data/classification/dataset0929/decisions \
  --output local/data/classification/dataset0929/decision_review_NEW \
  --ip-scope party
```

Inspect `review.xlsx`, `report.txt`, `summary.json`, and `plan.json`. Spacing and multiple-offence matches are candidates: brand-name capitalization, previous convictions, rejected joinder motions and generic legal quotations do not establish those exclusions. Record manual decisions in a local JSON object keyed by the exact manifest `filename`, for example:

```json
{
  "main/18.15/example.docx": {
    "note": "Joinder was refused; this ruling decides one episode.",
    "reviewed_flags": ["possible_multiple_offences"]
  }
}
```

An entry may also specify `add_reasons`, `remove_reasons`, and `evidence` (reason code to passage list). Available reason codes are documented by `REASONS` in the module. These records describe the agent's application of the experts' criteria, not a new human expert sign-off. Regenerate into another new output directory with `--overrides path/to/review_overrides.json`; pending flags prevent application. The completed 2026-10-25 local review and its overrides are under `local/data/classification/dataset0929/decision_review_20261025/`.

Apply a completed review with:

```bash
./.venv/bin/python -m LCD.experiments.data_creation.review_decisions \
  --apply local/data/classification/dataset0929/decision_review_NEW/plan.json
./.venv/bin/python -m LCD.experiments.data_creation.audit_collection
```

Application verifies the manifest and every reviewed DOCX/HTML/TXT hash before moving any files. Excluded bundles go to `decision_review_NEW/excluded/`, retaining their article directories and an independent `provenance.csv`. The active manifest contains only retained documents; its byte-for-byte original is in `before_review/provenance.csv`. `applied.json` records completion. On an ordinary move/write failure, the workflow rolls back completed moves and restores the original manifest. To reverse a completed review, move the archived bundles back using the original relative paths in `plan.json` and restore the saved manifest **only if no subsequent collection or curation has changed the active pool**. Do not resume a collector into this curated directory: use separate staging so excluded source URLs are not reintroduced.

## Date cutoff for the main original-ruling pool

The optional date filter defaults to retaining rulings dated **1 January 2024 onward**. The 2026-10-26 date-restricted collection was cancelled: the older rulings were restored and newly collected candidates were preserved. No date cutoff currently excludes those restored rulings. For a future date review, use the date printed in the opening, with an explicit archive-title date as fallback for anonymized openings. Search years and case-number years do not establish the ruling date. Inspect unresolved dates and any year disagreements in `plan.json` before applying. Use a new output directory outside `decisions/`:

```bash
./.venv/bin/python -m LCD.experiments.data_creation.filter_decisions_by_date \
  --cutoff 2024-01-01 \
  --output local/data/classification/dataset0929/decision_date_review_NEW
./.venv/bin/python -m LCD.experiments.data_creation.filter_decisions_by_date \
  --apply local/data/classification/dataset0929/decision_date_review_NEW/plan.json
./.venv/bin/python -m LCD.experiments.data_creation.audit_collection
```

Removed DOCX/HTML/TXT bundles are preserved in the date review's `excluded/`, with their provenance and the original manifest in `before_review/`. Preserve the earlier expert exclusions. Collect replacements into separate staging directories, apply the experts' exclusions again, and block previously collected URLs, text hashes and case identities before importing. Keep date evidence alongside the import review; report any shortage honestly.

## Build the portable NLI and RAG datasets

The native Series 01 workflow operates on accepted **original rulings** and keeps
appeals outside the dataset. It uses seed 42 to freeze 600/200/200 source
documents into `train`, `val`, and `test`; duplicate or related cases stay in a
single split. Run each command with `--help` before a production build and use a
new ignored bundle directory:

```bash
./.venv/bin/python -m LCD.experiments.data_creation.series_01_nli_rag initialize \
  --bundle local/data/classification/dataset0929/nli_rag_v1
./.venv/bin/python -m LCD.experiments.data_creation.series_01_nli_rag prepare \
  --bundle local/data/classification/dataset0929/nli_rag_v1 --split train
./.venv/bin/python -m LCD.experiments.data_creation.series_01_nli_rag tasks \
  --bundle local/data/classification/dataset0929/nli_rag_v1 \
  --split train --kind annotate --batch-size 20
```

Repeat `prepare` and `tasks` for `val` and `test`. Create each kind explicitly:

```bash
./.venv/bin/python -m LCD.experiments.data_creation.series_01_nli_rag tasks \
  --bundle local/data/classification/dataset0929/nli_rag_v1 \
  --split train --kind generate --batch-size 1
./.venv/bin/python -m LCD.experiments.data_creation.series_01_nli_rag tasks \
  --bundle local/data/classification/dataset0929/nli_rag_v1 \
  --split train --kind verify --batch-size 20
```

After independent verifier subagents return JSONL, import it explicitly for the
same split:

```bash
./.venv/bin/python -m LCD.experiments.data_creation.series_01_nli_rag verify \
  --bundle local/data/classification/dataset0929/nli_rag_v1 \
  --split train --input local/path/to/verifications.jsonl
```

An independent `accept` records supporting evidence but leaves an annotation
`proposed`; only a human review import can make it `approved`. `reject` and
`needs_review` return the affected row to pending review. A pilot verification
also caught a residual bank-detail hypothesis and partial entailment of compound
claims. The annotation and verification contracts therefore require entailment
support for the complete claim, contradiction when a supported incompatibility
affects any material atom, and explicit `extraction_filter_residue` review for
non-legal payment or bank fragments.

Initialization copies every source to the assigned split as UTF-8 text and
writes hashes and immutable assignments. Preparation extracts final
`ПОСТАНОВИЛ` hypotheses, audits every filtered sentence, resolves exact citations
first, fills semantic premises from `local/data/coderag`, and retrieves context.
The default premise depth is 10 unique provisions, with a hard maximum of 20.
Context retrieval uses Qwen with the exact instruction recorded in the
[series README](series_01_nli_rag/README.md), excludes the hypothesis, and writes
up to five sentences in document order.

Only expert-table rows whose normalized October 2026 inclusion value is exactly
`Да` enter task generation. Percentages are relative targets across feasible
contradiction scenarios. Train may use two distinct variants of one hypothesis
and several modified hypotheses from a source; `val` and `test` allow one active
variant and one scenario per source. The targets are pair-weighted
50/25/25 (`contradiction`/`entailment`/`not mentioned`) for train, about 120
contradiction-bearing validation documents, and about 40 test documents.
The coordinator persists each generation assignment before task export. One
generation task names one source, one original hypothesis, and one selected
expert rule; the agent must not choose among rules. Infeasible assignments return
`needs_review` and remain visible as quota shortages until the coordinator makes
a new assignment to an unused candidate.

Give the generated tasks to subagents using the checked-in Russian contracts:

- [`generate_variant.txt`](series_01_nli_rag/prompts/generate_variant.txt) for
  offset-based changes to a copied decision;
- [`annotate_pairs.txt`](series_01_nli_rag/prompts/annotate_pairs.txt) for
  independent NLI and relevance judgments;
- [`verify_proposals.txt`](series_01_nli_rag/prompts/verify_proposals.txt) for an
  independent check.

Apply an accepted variant proposal through the workflow rather than editing a
TXT directly. The coordinator verifies `before` against the source offsets,
writes a new `decisions/variants/` copy, and rebuilds all affected artifacts:

```bash
./.venv/bin/python -m LCD.experiments.data_creation.series_01_nli_rag apply-variant \
  --bundle local/data/classification/dataset0929/nli_rag_v1 \
  --input local/path/to/variant.jsonl
# Rebuild pairs and context from the new copied variant.
./.venv/bin/python -m LCD.experiments.data_creation.series_01_nli_rag prepare \
  --bundle local/data/classification/dataset0929/nli_rag_v1 --split train
./.venv/bin/python -m LCD.experiments.data_creation.series_01_nli_rag annotate \
  --bundle local/data/classification/dataset0929/nli_rag_v1 \
  --split train --input local/path/to/annotations.jsonl
```

The same `apply-variant` command records a subagent's `needs_review` response
when it has an empty `edits` list and a concrete rationale. It does not edit a
document in that case. Re-running `tasks --kind generate` preserves the failed
assignment in the audit ledger, releases its reservation and tries another
candidate without forcing a contradiction label. Excluded original pair IDs
deactivate their old generation assignments instead of invalidating the ledger.

The expert scenario is audit metadata, not a predetermined NLI label. Relevance
is also separate: a relevant neutral pair may be `not mentioned` and `related`.
Missing target provisions, insufficient context, conflicts, and uncertain legal
application remain `needs_review`; do not add a target provision to retrieval or
force a label to satisfy balance.

## Human review, stable IDs, and export

Run only one canonical writer (`prepare`, imports, filtering or export) for a
bundle at a time. Subagents can read tasks and write their own proposal files
while preparation runs. Keep agent outputs under the bundle's `agent_outputs/`
directory so they survive session interruptions.

For train, create a deterministic stratified human-review scope after annotation:

```bash
./.venv/bin/python -m LCD.experiments.data_creation.series_01_nli_rag review-plan \
  --bundle local/data/classification/dataset0929/nli_rag_v1 --split train \
  --fraction 0.1 --seed 42
# Experts review human_review.csv; keep the untouched original export separately.
./.venv/bin/python -m LCD.experiments.data_creation.series_01_nli_rag review \
  --bundle local/data/classification/dataset0929/nli_rag_v1 --split train \
  --sampled --input local/path/to/expert-reviewed-human_review.csv
```

The configurable default selects 10% from every populated premise-article,
NLI-label and error-type stratum (at least one), plus every synthetic or flagged
row. `--sampled` restricts deletion handling to the sampled snapshot; it does not
exclude the rest of train. `human_review_pending.jsonl` contains uncertain rows;
resolve or explicitly exclude those with `annotate --human` JSONL, including an
independent relevance judgment and rationale. Relevance corrections also use
human JSONL because relevance is stored in metadata. A completed audit permits
release of the remaining proposed natural train rows. Val/test still require
human approval of every included row. New unreviewed rows invalidate a train
audit until its scope is refreshed.

Use `prepare --device cuda --batch-size 16 --max-batch-tokens 8192` through the
Windows interpreter for the local GPU. The padded-token limit prevents long
provisions from causing shared-memory spillover and never truncates an input.
Preparation resumes from completed document checkpoints. `refresh-filter` can
exclude audited extraction remnants without reallocating existing row IDs;
stale annotation imports may use `--skip-excluded` to preserve those exclusions.

Inspect `preparation_review.jsonl` for hypotheses whose explicit citations exceed
the twenty-premise limit. These cases are held for review rather than silently
truncating their citations. After preparing edited decisions, run the target
audit to compare the intended provision with the actual retrieved candidates:

```bash
./.venv/bin/python -m LCD.experiments.data_creation.series_01_nli_rag audit-targets \
  --bundle local/data/classification/dataset0929/nli_rag_v1
```

`target_retrieval_review.jsonl` records missing target provisions. The audit never
inserts a reference provision into retrieval results or assigns an NLI label.
When generation tasks are refreshed, a validation/test variant with a missing
target provision or disputed independent verification releases its planned
positive-document slot for another source. Its existing copy and row history are
retained, and its source cannot receive a second variant.
The same applies when all retrieved target pairs have been annotated and none
has a contradiction label; neutral examples are retained without counting them
toward a positive-document target.
Use `refresh-rules --expert-table PATH` to refresh supplementary expert comments
without changing the pinned eligible rules or their core generation instructions.

Export a draft CSV, let experts delete or reorder rows and correct permitted
review fields, then import it with the review command. `row_id`, not line number,
is the identity. The persistent registry never renumbers surviving rows and never
reuses an ID deleted by an expert.

```bash
./.venv/bin/python -m LCD.experiments.data_creation.series_01_nli_rag export \
  --bundle local/data/classification/dataset0929/nli_rag_v1 --split train
# Experts edit the exported CSV files.
./.venv/bin/python -m LCD.experiments.data_creation.series_01_nli_rag review \
  --bundle local/data/classification/dataset0929/nli_rag_v1 \
  --split train --input local/path/to/expert-edited.csv
./.venv/bin/python -m LCD.experiments.data_creation.series_01_nli_rag validate \
  --bundle local/data/classification/dataset0929/nli_rag_v1
./.venv/bin/python -m LCD.experiments.data_creation.series_01_nli_rag export \
  --bundle local/data/classification/dataset0929/nli_rag_v1 \
  --split train --released
```

Repeat draft review and released export for `val` and `test`. The `annotate` and
`review` commands deliberately require `--split`; this prevents a row from being
silently imported into the wrong registry.

Every split folder contains `{split}.csv`, `metadata.jsonl`,
`sampling_weights.csv`, `manifest.json`, and `decisions/originals` and
`decisions/variants` text directories. CSV columns are
`row_id,hypothesis,premise,article_number,context,label,error_type,source_filename,decision_text_path`.
The relative `decision_text_path` makes the folder movable. Release validation
checks all referenced files and hashes and excludes unresolved rows. Human review
is required for every synthetic variant and every `val`/`test` pair; review all
uncertain train pairs and a stratified sample of the rest. Historical
`local/data/classification/val.csv` and `Full` remain separate.

## Load and sample a moved split

Training consumers do not need the build registry. Point the portable loader at
the CSV after moving the complete split directory; decision paths are resolved
relative to the CSV. Sampling probabilities are recomputed from surviving rows,
so an expert may delete or reorder CSV rows without invalidating IDs or retaining
stale weights:

```python
from LCD.experiments.data_creation.series_01_nli_rag.loading import (
    load_split, sample_row_ids, sampling_probabilities,
)

csv_path = "local/path/to/moved-train/train.csv"
rows = load_split(csv_path)
detection = sampling_probabilities(csv_path, task="detection")
rag = sampling_probabilities(csv_path, task="rag")
batch_ids = sample_row_ids(csv_path, task="detection", count=64, seed=42)
```

Detection requires all three NLI classes and targets 50/25/25. RAG reads
`metadata.jsonl`, requires relevance for every surviving row plus both relevance
classes, and targets 50/50. Missing classes or judgments fail explicitly; the
loader never substitutes a fallback distribution. Sampling uses only Python's
standard library and returns permanent `row_id` values with replacement.
