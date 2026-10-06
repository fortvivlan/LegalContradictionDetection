# Lawbook retrieval experiments

These experiments retrieve lawbook provisions for the contradiction classifier. Explicit article references are resolved before semantic retrieval, and the retrieved premises are then checked against hypotheses from court rulings.

- [Series 01: embedding and reranking](series_01_embedding_reranking/README.md) tested whether tuned embeddings, reranking, and retrieval depth could improve recall. Increasing retrieval depth improved recall but made inference slower and reduced end-to-end contradiction-detection quality when too many premises reached the classifier.
- [Series 02: deferred retrieval improvements](series_02_planned/README.md) proposes hybrid candidate generation, adaptive result counts, stronger representations, and a broader cost-quality evaluation. It is not a current implementation goal.
- [Series 03: update the premise-search lawbook](series_03_code_update/README.md) replaces the `dms-rag` corpus with expert-supplied legal texts, logs their editions, and audits federal-law mentions in dataset0929 decisions for missing laws. The local audit is implemented; expert review and database integration are pending.
- [Series 04: КоАП remarks and application scopes](series_04_remarks/README.md) extracts individual remarks and exports provisional categories and structured application entities for expert review and later linking.

Local entry points and input/output guidance are in the [lawbook instructions](instructions.md).
