import numpy as np
import faiss
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from src.ingestion import Chunk, ingest

# ---------- Load Embedding Model ----------

model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------- Build Embeddings ----------

def build_embeddings(chunks: List[Chunk]) -> np.ndarray:
    texts = [chunk.text for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    return np.array(embeddings)



# ---------- Build FAISS Index ----------

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


# ---------- Build BM25 Index ----------

def build_bm25(chunks: List[Chunk]) -> BM25Okapi:
    tokenized_chunks = [chunk.text.split() for chunk in chunks]
    return BM25Okapi(tokenized_chunks)


# ---------- Dense Retrieval ----------

def dense_search(
    query: str,
    index: faiss.IndexFlatL2,
    chunks: List[Chunk],
    top_k: int = 5
) -> List[Tuple[Chunk, float]]:

    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append((chunks[idx], dist))

    return results


# ---------- BM25 Retrieval ----------

def bm25_search(
    query: str,
    bm25: BM25Okapi,
    chunks: List[Chunk],
    top_k: int = 5
) -> List[Tuple[Chunk, float]]:

    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[-top_k:][::-1]

    results = [(chunks[i], scores[i]) for i in top_indices]
    return results


# ---------- Hybrid Retrieval ----------

def hybrid_search(
    query: str,
    index: faiss.IndexFlatL2,
    bm25: BM25Okapi,
    chunks: List[Chunk],
    top_k: int = 5
) -> List[Chunk]:

    dense_results = dense_search(query, index, chunks, top_k=top_k * 2)
    sparse_results = bm25_search(query, bm25, chunks, top_k=top_k * 2)

    combined = {}

    # Add dense results
    for chunk, _ in dense_results:
        combined[(chunk.doc_id, chunk.chunk_id)] = chunk

    # Add BM25 results
    for chunk, _ in sparse_results:
        combined[(chunk.doc_id, chunk.chunk_id)] = chunk

    # Convert to list
    candidates = list(combined.values())

    return candidates[:top_k]


# ---------- Debug Run ----------

if __name__ == "__main__":

    # Step 1: Load chunks
    chunks = ingest("data/docs")

    # Step 2: Build embeddings
    embeddings = build_embeddings(chunks)

    # Step 3: Build FAISS index
    index = build_faiss_index(embeddings)

    # Step 4: Build BM25 index
    bm25 = build_bm25(chunks)

    query = "Define overfitting in machine learning?"

    print("\n--- Dense Retrieval ---\n")
    dense_results = dense_search(query, index, chunks, top_k=5)

    for i, (chunk, score) in enumerate(dense_results):
        print(f"Result {i+1} (score={score:.4f})")
        print(f"Doc: {chunk.doc_id}")
        print(chunk.text)
        print("-" * 50)

    print("\n--- Hybrid Retrieval ---\n")
    hybrid_results = hybrid_search(query, index, bm25, chunks, top_k=5)

    for i, chunk in enumerate(hybrid_results):
        print(f"Result {i+1}")
        print(f"Doc: {chunk.doc_id}")
        print(chunk.text)
        print("-" * 50)