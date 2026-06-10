from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever


def test_retriever(retriever, queries):
    for query in queries:
        print(f"\n  Query: {query}")
        print("  " + "-" * 40)
        results = retriever.retrieve(query, top_k=2)
        for r in results:
            print(f"    [{r['score']:.3f}] {r['doc']['title']}")


def main():
    queries = [
        "how do neural nets update parameters",
        "what is a react agent",
        "combining semantic and keyword retrieval",
    ]

    print("\n===== BM25 (keyword) =====")
    bm25 = BM25Retriever("data/corpus/documents.json")
    test_retriever(bm25, queries)

    print("\n===== Dense (semantic) =====")
    dense = DenseRetriever("data/corpus/documents.json")
    test_retriever(dense, queries)


if __name__ == "__main__":
    main()