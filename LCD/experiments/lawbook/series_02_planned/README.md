# Series 02: planned retrieval improvements

This series is planned. Its aim is to recover relevant lawbook provisions without sending an unnecessarily large set of premises to the contradiction classifier. The following are candidate directions, not a fixed implementation plan:

1. Combine Qwen-based semantic retrieval and BM25 with explicit article-reference matches. Fuse ranked candidates with reciprocal rank fusion (RRF), include reference matches, then apply a tuned reranker.
2. Choose the final number of premises adaptively for each hypothesis instead of always returning 60.
3. Enrich sentence embeddings with surrounding legal context.
4. Compare recall against the average number of candidates and runtime, alongside downstream contradiction-detection quality, rather than reporting recall@N alone.
5. Distil the tuned reranker into a Qwen-based or other bi-encoder, using mined hard negatives.
6. If recall remains limited, investigate late interaction with BGE-M3 multi-vector retrieval or Jina-ColBERT.

No Series 02 runner or results exist yet. Local workflow instructions will be added to the [lawbook instructions](../instructions.md) when the design is selected.
