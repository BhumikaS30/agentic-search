from src.retrieval.bm25_retriever import BM25Retriever

def main():
    retriever = BM25Retriever("data/corpus/documents.json")

    queries = [
        "how does keyword search work",
        "what is a react agent",
        "combining semantic and keyword retrieval",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        results = retriever.retrieve(query, top_k=2)
        for r in results:
            print(f"  [{r['score']:.3f}] {r['doc']['title']}")

if __name__ == "__main__":
    main()