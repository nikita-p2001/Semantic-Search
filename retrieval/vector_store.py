import faiss
import numpy as np
import pickle


class FaissVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine sim (with normalized vectors)
        self.metadata = []

    def add(self, embeddings: np.ndarray, metadatas: list[dict]):
        self.index.add(embeddings)
        self.metadata.extend(metadatas)

    def search(self, query_emb: np.ndarray, top_k: int = 5):
        query_emb = np.expand_dims(query_emb, axis=0)
        scores, indices = self.index.search(query_emb, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append({
                "score": float(score),
                "metadata": self.metadata[idx]
            })

        return results

    def save(self, path: str):
        faiss.write_index(self.index, path + ".index")
        with open(path + ".meta.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, path: str):
        self.index = faiss.read_index(path + ".index")
        with open(path + ".meta.pkl", "rb") as f:
            self.metadata = pickle.load(f)
