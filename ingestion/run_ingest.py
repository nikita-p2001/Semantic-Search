# ingestion/run_ingest.py
from ingestion.loader import load_txt_files
from ingestion.normalizer import normalize_text
from ingestion.chunker import SmartChunker

import pickle
import os


def main():
    docs = load_txt_files("data")

    for doc in docs:
        doc.text = normalize_text(doc.text)

    chunker = SmartChunker(
        model_name="BAAI/bge-small-en-v1.5",
        chunk_size=300,
        overlap=50
    )

    all_chunks = []
    for doc in docs:
        chunks = chunker.chunk_text(
            text=doc.text,
            parent_doc_id=doc.doc_id,
            base_metadata=doc.metadata
        )
        all_chunks.extend(chunks)

    print(f"Loaded {len(docs)} documents")
    print(f"Generated {len(all_chunks)} chunks")

    if not all_chunks:
        print("WARNING: No chunks generated!")
        return

    sample_chunk = all_chunks[0]
    print("Sample chunk summary:")
    print(f"chunk_id: {sample_chunk.chunk_id}")
    print(f"parent_doc_id: {sample_chunk.parent_doc_id}")
    print(f"chunk_index: {sample_chunk.chunk_index}")
    print(f"text_length: {len(sample_chunk.text)} chars")
    print(f"metadata: {sample_chunk.metadata}")

    # ✅ SAVE CHUNKS (INSIDE main)
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    print("Saved chunks to artifacts/chunks.pkl")


if __name__ == "__main__":
    main()
