# Legal Contradiction Detection

LCD studies contradictions between the operative rulings of Russian court decisions and provisions of law. The pipeline extracts hypotheses from the final `ПОСТАНОВИЛ` section, retrieves relevant lawbook premises using explicit citations or semantic retrieval, and classifies each hypothesis–premise pair as `contradiction`, `entailment`, or `not mentioned`. Reports retain each source document, hypothesis, premise, citation, retrieval route, and prediction for review.

Reusable extraction, retrieval, classification, evaluation, and XLSX-to-CSV conversion code lives in `LCD/shared`; [shared instructions](LCD/shared/instructions.md) describe its local interfaces. Private datasets, models, checkpoints, and results live under the ignored `local/` tree. Historical archives live under the ignored `.legacy/` tree.

## Experiments

| Experiment | Description and instructions | Status |
| --- | --- | --- |
| [Data creation](LCD/experiments/data_creation/README.md) | Rebuild expert-reviewed train, dev, and test sets from original court rulings, with a separate appeal-evidence pool. [Instructions](LCD/experiments/data_creation/instructions.md). | Candidate documents collected; expert review pending |
| [Classifiers: Qwen3-8B full-document training](LCD/experiments/classifiers/series_01_qwen3_full_document/README.md) | Copy reviewed `Full` pairs with source decisions; prepare an 8k QLoRA trainer and disposable exact-length probe. [Classifier catalogue](LCD/experiments/classifiers/README.md); [Instructions](LCD/experiments/classifiers/instructions.md). | Export complete; first 8k probe exceeded GPU memory; training pending |
| [Lawbook: embedding and reranking](LCD/experiments/lawbook/series_01_embedding_reranking/README.md) | Historical embedding, reranking, and retrieval-depth study. [Instructions](LCD/experiments/lawbook/instructions.md). | Completed |
| [Lawbook: next retrieval series](LCD/experiments/lawbook/series_02_planned/README.md) | Candidate improvements to lawbook retrieval. [Instructions](LCD/experiments/lawbook/instructions.md). | Deferred |
| [Document-context retrieval](LCD/experiments/document_context_retrieval/series_01_planned/README.md) | Retrieve passages from court decisions to supplement hypothesis checking. [Instructions](LCD/experiments/document_context_retrieval/instructions.md). | Deferred |
| [LoRA: legacy adapters](LCD/experiments/lora/series_00_legacy_adapters/README.md) | Historical adapter recovery and evaluation. [Archive instructions](LCD/experiments/lora/instructions.md). | Archived |
| [LoRA: coordinate search](LCD/experiments/lora/series_01_coordinate_search/README.md) | Historical staged hyperparameter search; no further grid search is planned. [Archive instructions](LCD/experiments/lora/instructions.md). | Abandoned as a current direction; archives preserved |
| [Baseline classifiers and LoRA retraining](LCD/experiments/baselines/series_01_classifiers/README.md) | Historical classifier campaign and separate full-context retraining workflow. [Instructions](LCD/experiments/baselines/instructions.md). | Historical; runnable for reproduction |
| [LoRA/RAG comparison](LCD/experiments/baselines/series_02_lora_rag_comparison/README.md) | Historical three-class adapter evaluation and comparison. [Instructions](LCD/experiments/baselines/instructions.md). | Historical; reusable evaluator |
| [Qwen3.8 evaluation](LCD/experiments/baselines/series_03_qwen38/README.md) | Model-list configuration for the existing full-pipeline evaluator. [Instructions](LCD/experiments/baselines/instructions.md). | Historical configuration |
