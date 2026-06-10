import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def load_corpus(path):
    with open(path, "r") as f:
        return json.load(f)


class DenseRetriever:

    def __init__(self, corpus_path, model_name="all-MiniLM-L6-v2"):
        self.documents = load_corpus(corpus_path)

        # Load a small but powerful embedding model
        # This runs locally, no API key needed
        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)

        # Convert every document to a vector
        print("Encoding corpus...")
        texts = [doc["title"] + " " + doc["content"] for doc in self.documents]
        embeddings = self.model.encode(texts, show_progress_bar=True)

        # Normalize so cosine similarity = dot product (faster)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # IP = inner product
        self.index.add(embeddings.astype(np.float32))
        print(f"Index built with {self.index.ntotal} documents")

    def retrieve(self, query, top_k=3):
        # Convert query to vector
        query_embedding = self.model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Search FAISS index
        scores, indices = self.index.search(
            query_embedding.astype(np.float32), top_k
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "score": float(score),
                "doc": self.documents[idx]
            })

        return results