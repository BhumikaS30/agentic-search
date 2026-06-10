import json
from rank_bm25 import BM25Okapi


def load_corpus(path: str) -> list[dict]:
    """Load Documents from a JSON File"""
    with open(path, "r") as f:
        return json.load(f)
    
def tokenize(text) :
    return text.lower().split()

class BM25Retriever:
    def __init__(self, corpus_path: str):
        #Load Documents
        self.documents = load_corpus(corpus_path)

        # BM25 expects a list of tokenized documents
        tokenized_corpus = [tokenize(doc["title"] + " " + doc["content"]) for doc in self.documents]

        # Build the index
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Given a query string, return top_k most relevant documents."""
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Pair each document with its score, sort descending
        scored_docs = [
            {"score": scores[i], "doc": self.documents[i]}
            for i in range(len(self.documents))
        ]
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:top_k]