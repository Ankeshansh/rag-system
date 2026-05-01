from typing import List, Tuple
from sentence_transformers import CrossEncoder

from src.ingestion import Chunk
from src.retrieval import hybrid_search


# ---------- Load Cross Encoder ----------

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ---------- Rerank Function ----------

def rerank(
    query: str,
    candidates: List[Chunk],
    top_k: int = 5
) -> List[Tuple[Chunk, float]]:

    # Prepare pairs (query, chunk)
    pairs = [(query, chunk.text) for chunk in candidates]

    # Get relevance scores
    scores = reranker.predict(pairs)

    # Combine and sort
    scored_chunks = list(zip(candidates, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    return scored_chunks[:top_k]


# ---------- Full Pipeline ----------

def retrieve_and_rerank(
    query: str,
    index,
    bm25,
    chunks,
    top_k: int = 5
):
    candidates = hybrid_search(query, index, bm25, chunks, top_k=top_k * 5)
    reranked = rerank(query, candidates, top_k=top_k)
    return reranked


# ---------- Debug Run ----------

if __name__ == "__main__":

    from retrieval import build_embeddings, build_faiss_index, build_bm25
    from ingestion import ingest

    chunks = ingest("data/docs")

    embeddings = build_embeddings(chunks)
    index = build_faiss_index(embeddings)
    bm25 = build_bm25(chunks)

    query = "Define overfitting in machine learning?"

    results = retrieve_and_rerank(query, index, bm25, chunks, top_k=5)

    print(f"\nQuery: {query}\n")

    for i, (chunk, score) in enumerate(results):
        print(f"Result {i+1} (score={score:.4f})")
        print(f"Doc: {chunk.doc_id}")
        print(chunk.text)
        print("-" * 50)