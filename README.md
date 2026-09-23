# Legal Contradiction Detection

LCD studies contradictions between the operative rulings of Russian court decisions and provisions of law. The pipeline extracts hypotheses from the final `ПОСТАНОВИЛ` section, retrieves relevant lawbook premises using explicit citations or semantic retrieval, and classifies each hypothesis–premise pair as `contradiction`, `entailment`, or `not mentioned`. Reports retain each source document, hypothesis, premise, citation, retrieval route, and prediction for review.

Reusable extraction, retrieval, classification, evaluation, and XLSX-to-CSV conversion code lives in `LCD/shared`; [shared instructions](LCD/shared/instructions.md) describe its local interfaces. Private datasets, models, checkpoints, and results live under the ignored `local/` tree. Historical archives live under the ignored `.legacy/` tree.

## Experiments

| Group and series | Description | Status |
| --- | --- | --- |
| [Lawbook: embedding and reranking](LCD/experiments/lawbook/instructions.md) | Compare legal sentence embeddings, rerankers, retrieval depths, and citation resolution. The selected self-contained `rag-qwen` bundle is stored locally. | Completed series; reusable evaluation workflows |
| [Lawbook: next retrieval series](LCD/experiments/lawbook/instructions.md) | Further premise-retrieval experiments on the lawbook corpus. | Planned |
| [LoRA: legacy adapters](LCD/experiments/lora/instructions.md) | Historical adapters, recovery, and evaluation against baseline retrieval. | Historical |
| [LoRA: coordinate search](LCD/experiments/lora/instructions.md) | Compare target modules, rank, learning rate, alpha, and dropout across four language models. `lora_coordinate_balanced_val` preserves the completed balanced-validation run; `lora_coordinate_imbalanced_val` is the next fresh run. | Historical run preserved; new run prepared |
| [Document-context retrieval](LCD/experiments/document_context_retrieval/instructions.md) | Retrieve relevant passages from the court decision to add context to hypothesis checking. This planned series replaces the earlier summarization direction. | Planned |
| [Baselines and comparisons](LCD/experiments/baselines/instructions.md) | BERT and untuned-LLM baselines, full-pipeline evaluations, LoRA/RAG comparisons, and a Qwen3.8 experiment. | Historical and reusable evaluation workflows |
