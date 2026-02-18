# ingestion/schema.py
from dataclasses import dataclass
from typing import Dict

@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    source: str
    metadata: Dict

@dataclass
class Chunk:
    chunk_id: str
    parent_doc_id: str
    text: str
    chunk_index: int
    metadata: Dict