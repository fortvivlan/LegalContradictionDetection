# Series 01: embedding and reranking

This series investigated whether fine-tuning legal sentence embeddings and adding a reranker could improve lawbook retrieval **recall@N**. It compared baseline and tuned embeddings, pretrained and tuned rerankers, and different sizes for the initial candidate pool (`candidate_top_k`) and final set of premises (`final_top_k`). A citation audit examined the separate, higher-priority path for explicit article references.

The main practical finding was that increasing `top_k` improved retrieval recall, but the larger retrieval and reranking workload made inference too slow. Sending more retrieved premises to the contradiction-detection model also increased its opportunities to make errors, reducing overall quality. Higher recall@N alone therefore did not give a useful operating point for the full pipeline.

A paper about this work is in preparation. Publication details and a fuller account of the results can be added here when available.

The [lawbook instructions](../instructions.md) describe the local workflows and reports.
