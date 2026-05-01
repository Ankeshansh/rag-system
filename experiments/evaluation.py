import json
import numpy as np
import ollama
import re


from collections import defaultdict

from src.pipeline import build_system, run_pipeline
from src.retrieval import dense_search, hybrid_search
from src.rerank import retrieve_and_rerank


# ---------- Load Queries ----------

def load_queries(path="data/queries.json"):
    with open(path, "r") as f:
        return json.load(f)


# ---------- Metrics ----------

def recall_at_k(retrieved_docs, relevant_docs):
    return any(doc in retrieved_docs for doc in relevant_docs)


def mrr(retrieved_docs, relevant_docs):
    for i, doc in enumerate(retrieved_docs):
        if doc in relevant_docs:
            return 1 / (i + 1)
    return 0

def normalize(doc):
    return doc.lower().replace(".txt", "").strip()


def context_recall_score(reranked_chunks, relevant_docs):
    """
    Returns fraction of relevant docs present in retrieved chunks

    Example:
    relevant_docs = [A, B]
    retrieved_docs = [A, C]

    → score = 1/2 = 0.5
    """

    # normalize both sides
    retrieved_docs = [normalize(chunk.doc_id) for chunk, _ in reranked_chunks]
    relevant_docs = [normalize(doc) for doc in relevant_docs]

    # count matches
    hits = sum(1 for doc in relevant_docs if doc in retrieved_docs)

    return hits / len(relevant_docs)


def keyword_score(answer, keyword_groups):
    if not keyword_groups:
        return None

    score = 0
    for group in keyword_groups:
        if any(word.lower() in answer.lower() for word in group):
            score += 1

    return score / len(keyword_groups)

def length_penalty(answer, query):
    answer_len = len(answer.split())
    query_len = len(query.split())

    # expected answer length grows with query complexity
    expected_len = max(20, query_len * 3)

    if answer_len >= expected_len:
        return 1.0  # no penalty
    elif answer_len >= expected_len * 0.6:
        return 0.8
    elif answer_len >= expected_len * 0.4:
        return 0.6
    else:
        return 0.4

def judge_answer(query, answer, context):
    prompt = f"""
                You are a STRICT evaluator for a question-answering system.

                Your job is to evaluate the answer along three dimensions:

                IMPORTANT:
                - Correctness and completeness should be judged using your own knowledge.
                - Grounding should be judged ONLY based on the provided context.
                - Do NOT assume the context is complete or correct.
                - Be strict. Do not give high scores for partial or vague answers.

                Rules:
                - If the answer says "I don't know" → correctness = 0, completeness = 0
                - If the answer is partially correct → correctness < 1
                - If key details are missing → completeness < 0.6
                - If the answer is not supported by the context → grounding < 0.5

                Question: {query}
                Answer: {answer}
                Context: {context}

                Evaluate:

                1. Correctness (0-1): Is the answer factually correct based on your knowledge?
                2. Completeness (0-1): Does the answer fully address the question?
                3. Grounding (0-1): Is the answer supported by the provided context?

                Also provide a SHORT REASON.

                Return ONLY JSON:
                {{
                "correctness": float,
                "completeness": float,
                "grounding": float,
                "overall": float,
                "reason": "brief explanation"
                }}
            """

    try:
        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )

        content = response["message"]["content"]

        # 🔥 Extract JSON using regex
        match = re.search(r"\{.*\}", content, re.DOTALL)

        if match:
            return json.loads(match.group())

        return None

    except Exception as e:
        print(f"Error in judge_answer: {e}")
        return None


# ---------- Evaluation ----------

def evaluate():

    USE_JUDGE = True

    queries = load_queries()
    chunks, index, bm25 = build_system()

    metrics = defaultdict(list)

    results_list = []

    for item in queries:

        query = item["query"]
        relevant_docs = item["relevant_docs"]
        keyword_groups = item.get("expected_answer_keywords", [])

        # =========================
        # DENSE
        # =========================
        dense_results = dense_search(query, index, chunks, top_k=5)
        dense_docs = [chunk.doc_id for chunk, _ in dense_results]

        metrics["dense_recall"].append(recall_at_k(dense_docs, relevant_docs))
        metrics["dense_mrr"].append(mrr(dense_docs, relevant_docs))

        # =========================
        # HYBRID
        # =========================
        hybrid_results = hybrid_search(query, index, bm25, chunks, top_k=5)
        hybrid_docs = [
            chunk.doc_id if not isinstance(chunk, tuple) else chunk[0].doc_id
            for chunk in hybrid_results
        ]

        metrics["hybrid_recall"].append(recall_at_k(hybrid_docs, relevant_docs))
        metrics["hybrid_mrr"].append(mrr(hybrid_docs, relevant_docs))

        # =========================
        # RERANK
        # =========================
        rerank_results = retrieve_and_rerank(query, index, bm25, chunks, top_k=5)
        rerank_docs = [chunk.doc_id for chunk, _ in rerank_results]

        metrics["rerank_recall"].append(recall_at_k(rerank_docs, relevant_docs))
        metrics["rerank_mrr"].append(mrr(rerank_docs, relevant_docs))

        # =========================
        # FINAL PIPELINE
        # =========================
        answer, context, reranked = run_pipeline(query, chunks, index, bm25)

        final_docs = [chunk.doc_id for chunk, _ in reranked]


        metrics["final_recall"].append(recall_at_k(final_docs, relevant_docs))
        metrics["final_mrr"].append(mrr(final_docs, relevant_docs))

        # Context recall
        score = context_recall_score(reranked, relevant_docs)
        metrics["context_recall"].append(score)

        # Answer quality (keyword proxy)
        keyword_score_val = keyword_score(answer, keyword_groups)

        if keyword_score_val is not None:
            metrics["answer_score"].append(keyword_score_val)


        # LLM-based evaluation
        judge_scores = None

        if USE_JUDGE:
            judge_scores = judge_answer(query, answer, context)

            if judge_scores:
                c = judge_scores.get("correctness", 0)
                comp = judge_scores.get("completeness", 0)
                g = judge_scores.get("grounding", 0)

                # 🔥 apply length penalty ONLY to completeness
                lp = length_penalty(answer, query)
                comp = comp * lp

                # recompute overall (VERY IMPORTANT)
                overall = 0.4 * c + 0.4 * comp + 0.2 * g

                # penalize incomplete answers
                if comp < 0.7:
                    overall *= 0.6

                # penalize "I don't know"
                answer_lower = answer.lower()
                if "i don't know" in answer_lower or len(answer.strip()) < 20:
                    overall *= 0.3
                    c = min(c, 0.2)

                # penalize hallucination
                if g < 0.5:
                    overall *= 0.7

                # update back
                judge_scores["correctness"] = c
                judge_scores["overall"] = overall

            if judge_scores and "overall" in judge_scores:
                metrics["judge_score"].append(judge_scores["overall"])


        results_list.append({
            "query": query,
            "relevant_docs": relevant_docs,
            "retrieved_docs": final_docs,
            "answer": answer,
            "context_recall_score": metrics["context_recall"][-1],
            "answer_score": keyword_score_val if keyword_score_val is not None else None,
            "judge_scores": judge_scores
        })

    # ---------- Aggregate ----------
    print("\n\n===== FINAL METRICS =====")

    for key, values in metrics.items():
        print(f"{key}: {np.mean(values):.3f}")

    with open("experiments/results.json", "w") as f:
        json.dump(results_list, f, indent=4)


if __name__ == "__main__":
    evaluate()