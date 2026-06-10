from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever


def reciprocal_rank_fusion(bm25_results, dense_results, k=60):
    """
    RRF formula: score = sum of 1/(k + rank) across all retrievers.
    
    k=60 is the standard constant — it dampens the impact of very high 
    ranked results so no single retriever dominates.
    """
    scores = {}  # doc_id -> fused score
    doc_map = {}  # doc_id -> actual doc object

    # Score BM25 results by rank position
    for rank, result in enumerate(bm25_results):
        doc_id = result["doc"]["id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        doc_map[doc_id] = result["doc"]

    # Score dense results by rank position
    for rank, result in enumerate(dense_results):
        doc_id = result["doc"]["id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        doc_map[doc_id] = result["doc"]

    # Sort by fused score descending
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)

    return [
        {"score": scores[doc_id], "doc": doc_map[doc_id]}
        for doc_id in sorted_ids
    ]


class HybridRetriever:

    def __init__(self, corpus_path):
        print("Initialising BM25...")
        self.bm25 = BM25Retriever(corpus_path)

        print("Initialising Dense retriever...")
        self.dense = DenseRetriever(corpus_path)

    def retrieve(self, query, top_k=3):
        # Get results from both retrievers
        # Fetch more than top_k so RRF has enough to re-rank
        bm25_results = self.bm25.retrieve(query, top_k=10)
        dense_results = self.dense.retrieve(query, top_k=10)

        # Fuse
        fused = reciprocal_rank_fusion(bm25_results, dense_results)

        return fused[:top_k]