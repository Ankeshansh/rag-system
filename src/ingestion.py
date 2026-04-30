import os
from typing import List, Dict
from dataclasses import dataclass


# ---------- Data Structure ----------

@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_id: int


# ---------- Load Documents ----------

def load_documents(data_path: str) -> Dict[str, str]:
    """
    Loads all .txt files from data/docs/

    Returns:
        dict -> {filename: text}
    """
    documents = {}

    for file in os.listdir(data_path):
        if file.endswith(".txt"):
            path = os.path.join(data_path, file)

            with open(path, "r", encoding="utf-8") as f:
                documents[file] = f.read()

    return documents


# ---------- Basic Cleaning ----------

def clean_text(text: str) -> str:
    """
    Minimal cleaning:
    - removes very short lines
    - strips extra whitespace
    """
    lines = text.split("\n")

    cleaned = [
        line.strip()
        for line in lines
        if len(line.strip()) > 30
        and "ISBN" not in line
        and "Retrieved from" not in line
    ]

    return "\n".join(cleaned)


# ---------- Chunking ----------

def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> List[str]:
    """
    Splits text into overlapping chunks
    """

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ---------- Full Ingestion Pipeline ----------

def ingest(data_path: str) -> List[Chunk]:
    """
    Full pipeline:
    load → clean → chunk → attach metadata
    """

    documents = load_documents(data_path)

    all_chunks = []

    for doc_id, text in documents.items():
        cleaned = clean_text(text)
        chunks = chunk_text(cleaned)

        for i, chunk in enumerate(chunks):
            all_chunks.append(
                Chunk(
                    text=chunk,
                    doc_id=doc_id,
                    chunk_id=i
                )
            )

    return all_chunks


# ---------- Debug Run ----------

if __name__ == "__main__":
    chunks = ingest("data/docs")

    print(f"Total chunks: {len(chunks)}")
    print("\nSample chunk:\n")
    print(chunks[0].text[:500])