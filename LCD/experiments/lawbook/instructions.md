# Lawbook experiments

`series_01_embedding_reranking` contains the completed legal sentence-embedding experiment, retrieval-depth sweep, embedding-model sweep, and deterministic citation audit. The corpus checkout is `local/resources/dms-rag/` when movable, with a supported local fallback at the ignored root `dms-rag/`; the selected self-contained bundle is `local/bundles/rag-qwen/`. Annotation workbooks and decision DOCX files are under `local/data/benchmarks/`.

Entry points are `python -m LCD.experiments.lawbook.series_01_embedding_reranking.run_experiment`, `run_depth_sweep`, `run_embedding_sweep`, and `run_citation_audit` with that same module prefix. Use `--help` for explicit input, model, revision, device, and output options. New retrieval reports evaluate `Full` only and omit Dialogue columns. Historical Dialogue comparisons require explicitly selecting both datasets through the Python evaluation interface.

The first series compares baseline and tuned embeddings, optional rerankers, and candidate/final retrieval depths. Citation matches retain priority over semantic candidates. Runs record corpus and model revisions and write ignored artifacts and results under `local/experiments/lawbook/series_01_embedding_reranking/`.

`series_02_planned` is deferred. Its [README](series_02_planned/README.md) lists candidate directions; no final design or runner has been selected yet.
