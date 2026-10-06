# Lawbook experiments

`series_01_embedding_reranking` contains the completed legal sentence-embedding experiment, retrieval-depth sweep, embedding-model sweep, and deterministic citation audit. The corpus checkout is `local/resources/dms-rag/` when movable, with a supported local fallback at the ignored root `dms-rag/`; the selected self-contained bundle is `local/bundles/rag-qwen/`. Annotation workbooks and decision DOCX files are under `local/data/benchmarks/`.

Entry points are `python -m LCD.experiments.lawbook.series_01_embedding_reranking.run_experiment`, `run_depth_sweep`, `run_embedding_sweep`, and `run_citation_audit` with that same module prefix. Use `--help` for explicit input, model, revision, device, and output options. New retrieval reports evaluate `Full` only and omit Dialogue columns. Historical Dialogue comparisons require explicitly selecting both datasets through the Python evaluation interface.

The first series compares baseline and tuned embeddings, optional rerankers, and candidate/final retrieval depths. Citation matches retain priority over semantic candidates. Runs record corpus and model revisions and write ignored artifacts and results under `local/experiments/lawbook/series_01_embedding_reranking/`.

`series_02_planned` is deferred. Its [README](series_02_planned/README.md) lists candidate directions; no final design or runner has been selected yet.

## Series 03: legal-code inventory and coverage audit

The [code-update experiment](series_03_code_update/README.md) prepares expert review before replacing the premise-search corpus. From the repository root, with the project dependencies installed, run:

```bash
python -m LCD.experiments.lawbook.series_03_code_update.audit_codes \
  --codes-dir local/data/codes \
  --decisions-dir local/data/classification/dataset0929/decisions \
  --output-dir local/experiments/lawbook/series_03_code_update
```

The audit reads all decision DOCX files in both dataset groups, including tables, and splits their full text into sentences with `razdel`. It does not apply hypothesis extraction or irrelevant-text filtering. Appeal directories and HTML/TXT mirrors are excluded. Every matching occurrence is retained with its decision path and sentence index; no models or external services are used. An unreadable document stops the run rather than being silently skipped.

Outputs are UTF-8 `federal_law_sentences.txt` with one representative sentence per identifiable law for the expert, `federal_law_sentences_full.txt` preserving every matching occurrence, `code_versions.json` recording source filenames, edition labels and SHA-256 hashes, and `coverage_summary.json` recording counts, decision hashes and explicit law numbers absent from the supplied filenames. Compact grouping prefers the first dated citation and normalizes numeric and written Russian dates. Different laws sharing a number retain separate dates; matching descriptive titles merge date variants for the same number. Undated references merge when the law number identifies only one known law. Quoted titles are normalized (including `РФ`) and merged by containment or spelling similarity of at least 0.8; generic amendment titles are matched exactly. Titles of orders and decrees following a law reference are excluded. Generic references without an identifiable number or title appear only in the full export. A sentence citing several laws can represent several groups without being printed repeatedly. Grouping is provisional and requires expert verification for typos, redactions and ambiguous references.

Edition labels come from filenames and are not independently verified effective dates. Number-only comparisons are candidate omissions, not resolved legal identities; the expert should check titles/dates, redactions, and sentences without explicit numbers. All outputs remain under ignored `local/`. After expert review, add any required law editions, rerun the inventory/audit, then build and integrate the replacement corpus as a separate step.

## Series 04: КоАП remarks and application entities

The [remarks experiment](series_04_remarks/README.md) uses local Python only. The completed supplied-document annotations live under `local/experiments/lawbook/series_04_remarks/`. Reproduce the curated export with:

```bash
python -m LCD.experiments.lawbook.series_04_remarks.annotate_remarks \
  --source 'local/data/codes/Кодекс Российской Федерации об административных правонарушениях_редакция 01.10.2026.docx' \
  --annotations local/experiments/lawbook/series_04_remarks/scope_annotations.json \
  --output-dir local/experiments/lawbook/series_04_remarks
```

For another source or an initial rules-based suggestion, omit `--annotations` and use a new output directory. Do not omit it when reproducing the curated supplied-document export: that would replace the manual scope interpretations with initial suggestions. Input paths, annotation paths and output paths are explicit. Source and per-remark text hashes must match curated annotations; changed documents require fresh annotation. The exporter writes its current annotations as `scope_annotations.json`, allowing repeatable local regeneration.

Open `koap_remarks_review.xlsx`: the first columns contain full remark text, category, and application entities. Filter `Приоритет проверки` to `высокий` first, then review all remaining rows. The last three columns are for expert category/entity corrections and comments. Very long remarks remain fully stored in their cells; expand the formula bar to inspect text exceeding Excel's maximum row height. Expert edits to the workbook do not automatically change the machine-readable JSON or CSV; reconcile them into the annotation JSON before generating final links. The category guide is also exported as `remark_categories.txt` and included on the workbook's `Категории` sheet.

`koap_remarks.csv` preserves one row per numbered or unnumbered remark, full multiline text, category, application entities, source article/title/chapter/section, one-based DOCX XML paragraph positions (including tables and empty paragraphs), and document/text hashes. `remark_targets.csv` and the workbook's `Связи` sheet contain one immediate relation per target. Entity IDs distinguish code, chapter, article, part, paragraph, other remark, and thematic group. An edition-independent structural ID such as `koap_rf:article:1.5:part:3` must be paired with the document hash/edition for later linking.

All scope annotations are provisional. Explicit article scopes are retained at article level; some otherwise unnumbered definitions are matched to parts using the article's dispositions and marked for priority review. Citations describing exceptions or sources of definitions are not automatically application targets. Open thematic groups keep candidate article entities separately from actual links. Repealed placeholders preserve locations without active links, and applicability through another remark is not expanded transitively. No corpus database is changed.
