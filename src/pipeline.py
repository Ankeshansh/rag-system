from ingestion import ingest
from retrieval import build_embeddings, build_faiss_index, build_bm25
from rerank import retrieve_and_rerank
from generate import generate_answer

import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------- Build System (one-time setup) ----------

def build_system(data_path="data/docs"):

    # Load chunks
    chunks = ingest(data_path)

    # Build embeddings
    embeddings = build_embeddings(chunks)

    # Build FAISS index
    index = build_faiss_index(embeddings)

    # Build BM25
    bm25 = build_bm25(chunks)

    return chunks, index, bm25


# ---------- Context Builder ----------

# def build_context(reranked_chunks, max_chars=2000):

#     context = ""

#     for chunk, _ in reranked_chunks:
#         if len(context) + len(chunk.text) <= max_chars:
#             context += chunk.text.strip() + "\n\n"
#         else:
#             break

#     return context.strip()

def build_context(query, reranked_chunks, model, max_chars=2000):

    query_emb = model.encode([query])[0]

    scored = []

    for chunk, _ in reranked_chunks:
        chunk_emb = model.encode([chunk.text])[0]

        # cosine similarity
        score = np.dot(query_emb, chunk_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb)
        )

        scored.append((chunk, score))

    # sort again based on alignment
    scored.sort(key=lambda x: x[1], reverse=True)

    context = ""
    selected = []

    for chunk, score in scored:
        if len(context) + len(chunk.text) <= max_chars:
            selected.append(chunk)
            context += chunk.text + "\n\n"

    return context.strip()


# ---------- Retrieval Pipeline ----------

def retrieve_context(query, chunks, index, bm25, top_k=5):

    # Step 1: retrieve + rerank
    reranked = retrieve_and_rerank(query, index, bm25, chunks, top_k=top_k)

    # Step 2: build context
    context = build_context(query, reranked, model)

    return context, reranked


# Generate Answer Pipeline

def run_pipeline(query, chunks, index, bm25):

    # Step 1: Retrieve context
    context, reranked = retrieve_context(query, chunks, index, bm25)

    # Step 2: Generate answer
    answer = generate_answer(query, context)

    return answer, context, reranked


# ---------- Debug Run ----------

if __name__ == "__main__":

    # Build system once
    chunks, index, bm25 = build_system()

    # Test query
    query = "Define overfitting in machine learning?"

    answer, context, reranked = run_pipeline(query, chunks, index, bm25)

    print("\n--- ANSWER ---\n")
    print(answer)

    print("\n--- CONTEXT (truncated) ---\n")
    print(context)