# Legal Contradiction Detection

LCD studies contradictions between the operative rulings of Russian court decisions and provisions of law. The pipeline extracts hypotheses from the final `ПОСТАНОВИЛ` section, retrieves relevant lawbook premises using explicit citations or semantic retrieval, and classifies each hypothesis–premise pair as `contradiction`, `entailment`, or `not mentioned`. Reports retain each source document, hypothesis, premise, citation, retrieval route, and prediction for review.

Reusable extraction, retrieval, classification, evaluation, and XLSX-to-CSV conversion code lives in `LCD/shared`; [shared instructions](LCD/shared/instructions.md) describe its local interfaces. Private datasets, models, checkpoints, and results live under the ignored `local/` tree. Historical archives live under the ignored `.legacy/` tree.

## Experiments

| Group and series | Description | Status |
| --- | --- | --- |
| [Lawbook: embedding and reranking](LCD/experiments/lawbook/series_01_embedding_reranking/README.md) | Test embeddings, rerankers, and retrieval depth; document the recall, runtime, and classifier-quality tradeoff. [Run instructions](LCD/experiments/lawbook/instructions.md). | Completed series; reusable evaluation workflows |
| [Lawbook: next retrieval series](LCD/experiments/lawbook/series_02_planned/README.md) | Explore hybrid retrieval, adaptive final depth, richer embeddings, distillation, and late interaction. [Lawbook overview](LCD/experiments/lawbook/README.md) and [instructions](LCD/experiments/lawbook/instructions.md). | Planned |
| [LoRA: legacy adapters](LCD/experiments/lora/series_00_legacy_adapters/README.md) | Historical adapters, recovery, and evaluation against baseline retrieval. [Archive instructions](LCD/experiments/lora/instructions.md). | Archived |
| [LoRA: coordinate search](LCD/experiments/lora/series_01_coordinate_search/README.md) | Historical staged tuning across four language models. [Archive instructions](LCD/experiments/lora/instructions.md). | Archived |
| [Document-context retrieval](LCD/experiments/document_context_retrieval/instructions.md) | Retrieve relevant passages from the court decision to add context to hypothesis checking. This planned series replaces the earlier summarization direction. | Planned |
| [Baseline LoRA retraining](LCD/experiments/baselines/series_01_classifiers/README.md) | Retrain four LoRAs with complete context and compare with three saved cohorts. [Run instructions](LCD/experiments/baselines/instructions.md). | Ready to run |
| [Baselines and comparisons](LCD/experiments/baselines/instructions.md) | BERT and untuned-LLM baselines, full-pipeline evaluations, LoRA/RAG comparisons, and a Qwen3.8 experiment. | Historical and reusable evaluation workflows |
