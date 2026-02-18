from retrieval.embedder import Embedder
from retrieval.vector_store import FaissVectorStore
import pickle


def main():
    # Load chunks from Phase 2 output
    with open("artifacts/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    embedder = Embedder()

    texts = [c.text for c in chunks]
    metadatas = [{
        "chunk_id": c.chunk_id,
        "parent_doc_id": c.parent_doc_id,
        "chunk_index": c.chunk_index,
        "text": c.text
    } for c in chunks]

    print("Embedding chunks...")
    embeddings = embedder.embed_texts(texts)

    dim = embeddings.shape[1]
    store = FaissVectorStore(dim)
    store.add(embeddings, metadatas)

    store.save("artifacts/company_kb")

    print(f"Indexed {len(chunks)} chunks with dim={dim}")


if __name__ == "__main__":
    main()
