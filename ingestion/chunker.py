# ingestion/chunker.py
from transformers import AutoTokenizer
from ingestion.schema import Chunk
import uuid

class SmartChunker:
    def __init__(self, model_name: str, chunk_size: int = 300, overlap: int = 50):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, parent_doc_id: str, base_metadata: dict):
        tokens = self.tokenizer.encode(text)
        chunks = []

        start = 0
        chunk_index = 0

        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]

            chunk_text = self.tokenizer.decode(chunk_tokens)

            chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                parent_doc_id=parent_doc_id,
                text=chunk_text,
                chunk_index=chunk_index,
                metadata={
                    **base_metadata,
                    "token_start": start,
                    "token_end": end
                }
            )

            chunks.append(chunk)

            chunk_index += 1
            start += self.chunk_size - self.overlap

        return chunks
