from retrieval.embedder import Embedder
from retrieval.vector_store import FaissVectorStore


def main():
    embedder = Embedder()

    store = FaissVectorStore(dim=384)
    store.load("artifacts/company_kb")

    while True:
        query = input("\nEnter query (or 'exit'): ")
        if query == "exit":
            break

        q_emb = embedder.embed_query(query)
        results = store.search(q_emb, top_k=5)

        print("\nTop Results:")
        for i, r in enumerate(results):
            print(f"\n#{i+1} | score={r['score']:.4f}")
            print(r["metadata"]["text"][:500])


if __name__ == "__main__":
    main()
