# 🔍 Retrieval-Augmented Generation (RAG) System from Scratch with Hybrid Search, Reranking & LLM Evaluation

A fully modular **RAG pipeline** built from scratch with hybrid retrieval, reranking, query-aware context selection, and **LLM-based evaluation**, designed to study the gap between retrieval quality and final answer quality.

---

## 🚀 Overview

This project implements an end-to-end **Retrieval-Augmented Generation (RAG)** system:

```
Query
 → Hybrid Retrieval (Dense + BM25)
 → Cross-Encoder Reranking
 → Query-aware Context Selection
 → Local LLM Generation (Ollama - LLaMA3)
 → Evaluation (Retrieval + LLM-as-Judge)
```

The goal was not just to build a pipeline, but to **analyze where RAG systems actually fail** — retrieval vs context vs generation.

---

## 🔥 Key Features

### 🔹 Hybrid Retrieval

* Dense embeddings using Sentence Transformers
* Sparse retrieval using BM25
* Combines semantic + lexical search

### 🔹 Reranking

* Cross-encoder reranking of top candidates
* Improves relevance ordering

### 🔹 Query-Aware Context Selection

* Selects most relevant chunks dynamically
* Avoids naive top-k concatenation
* Improves signal-to-noise ratio

### 🔹 Diversity-Aware Retrieval

* Limits dominance of a single document
* Ensures multi-document coverage

### 🔹 Local LLM Generation

* Uses **Ollama (LLaMA3)** for inference
* No external APIs required

### 🔹 Advanced Evaluation System

* Retrieval metrics: Recall@K, MRR
* Context-level evaluation
* LLM-as-a-Judge (correctness, completeness, grounding)
* Length-aware penalties for realistic scoring

---

## 📊 Final Results

| Metric                 | Value           |
| ---------------------- | --------------- |
| Recall@5               | **1.000**       |
| MRR                    | **0.84**        |
| Context Recall         | **0.90**        |
| Answer Score (Keyword) | **0.67**        |
| Judge Score (LLM Eval) | **0.88**        |

---

## 🧠 Important Insights

### 💡 1. Perfect Retrieval ≠ Perfect Answers

Despite achieving **Recall@5 = 1.0**, answer quality was initially poor due to:

* incomplete context
* shallow generation

---

### 💡 2. Context Coverage Matters

Even with perfect retrieval:

```
Context Recall = 0.9
```

Some multi-document queries still missed key information.

---

### 💡 3. Evaluation Must Be Multi-Dimensional

We decouple evaluation into:

* **Correctness** → LLM knowledge
* **Completeness** → answer depth
* **Grounding** → context support

---

### 💡 5. Length-Aware Scoring is Crucial

Short answers were initially over-scored.
We introduced a **query-aware length penalty** to reflect true completeness.

---

## ⚠️ Note on Recall@5 = 1.0

This result is influenced by:

* Small curated dataset (~10 documents)
* Well-aligned queries

To avoid misleading conclusions, we additionally evaluate:

* Context Recall (multi-doc coverage)
* LLM-based answer quality

---

## 🏗️ Project Structure

```
hybrid-rag-system-with-reranking-and-llm-based-evaluation/
│
├── src/
│   ├── ingestion.py        # chunking & preprocessing
│   ├── retrieval.py        # dense + BM25 retrieval
│   ├── rerank.py           # cross-encoder reranking
│   ├── pipeline.py         # full RAG pipeline
│   ├── generate.py         # LLM generation (Ollama)
│
├── experiments/
│   ├── evaluation.py       # evaluation pipeline
│
├── data/
│   ├── *.txt               # knowledge base documents
│
├── requirements.txt
├── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/rag-system.git
cd hybrid-rag-system-with-reranking-and-llm-based-evaluation
```

---

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Install Ollama

```bash
brew install ollama
```

Start server:

```bash
ollama serve
```

Pull model:

```bash
ollama pull llama3
```

---

## ▶️ Running the System

### Run RAG pipeline

```bash
python src/pipeline.py
```

---

### Run evaluation

```bash
python -m experiments.evaluation
```

---

## 🧪 Example Query

```text
Define overfitting in machine learning
```

### Output:

* Retrieved relevant chunks
* Generated structured answer
* Evaluated via LLM judge

---

## 📈 Evaluation Details

### Retrieval Metrics

* Recall@K
* MRR

### Context Metrics

* Context Recall Score (multi-document coverage)

### Answer Metrics

* Keyword-based score
* LLM-based evaluation:

  * correctness
  * completeness
  * grounding

---

## 🔍 Key Learnings

* Hybrid retrieval significantly improves recall
* Reranking improves precision but not completeness
* Context construction is critical for answer quality
* LLM-based evaluation is powerful but requires strict control

---

## 🚀 Future Improvements

* Larger and noisier datasets
* Better multi-hop retrieval
* Answer aggregation strategies
* Multi-model evaluation (reduce judge bias)
* Deployment as API

---

## 🧑‍💻 Author

**Ankesh Ansh**
IIT (BHU) Varanasi
