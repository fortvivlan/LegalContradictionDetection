# Lawbook retrieval experiments

These experiments retrieve lawbook provisions for the contradiction classifier. Explicit article references are resolved before semantic retrieval, and the retrieved premises are then checked against hypotheses from court rulings.

- [Series 01: embedding and reranking](series_01_embedding_reranking/README.md) tested whether tuned embeddings, reranking, and retrieval depth could improve recall. Increasing retrieval depth improved recall but made inference slower and reduced end-to-end contradiction-detection quality when too many premises reached the classifier.
- [Series 02: planned retrieval improvements](series_02_planned/README.md) proposes hybrid candidate generation, adaptive result counts, stronger representations, and a broader cost-quality evaluation.

Local entry points and input/output guidance are in the [lawbook instructions](instructions.md).
