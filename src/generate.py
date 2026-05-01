import requests


# ---------- Generate Answer using Ollama ----------

def generate_answer(query: str, context: str) -> str:

    prompt = f"""
                You are a helpful AI assistant.

                Answer clearly and completely ONLY using the context.
                Combine information from multiple parts if needed.
                Explain the concept fully in 2–4 sentences.
                Do not give very short answers.
                If the answer is not present, say "I don't know."

                Context:
                {context}

                Question:
                {query}

                Answer:
            """

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    result = response.json()

    return result["response"].strip()