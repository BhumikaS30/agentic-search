from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid import HybridRetriever


def test_retriever(name, retriever, queries):
    print(f"\n===== {name} =====")
    for query in queries:
        print(f"\n  Query: {query}")
        print("  " + "-" * 40)
        results = retriever.retrieve(query, top_k=2)
        for r in results:
            print(f"    [{r['score']:.4f}] {r['doc']['title']}")


def main():
    corpus = "data/corpus/documents.json"

    queries = [
        "how do neural nets update parameters",
        "what is a react agent",
        "combining semantic and keyword retrieval",
    ]

    bm25 = BM25Retriever(corpus)
    dense = DenseRetriever(corpus)
    hybrid = HybridRetriever(corpus)

    test_retriever("BM25 (keyword)", bm25, queries)
    test_retriever("Dense (semantic)", dense, queries)
    test_retriever("Hybrid (RRF fusion)", hybrid, queries)


if __name__ == "__main__":
    main()