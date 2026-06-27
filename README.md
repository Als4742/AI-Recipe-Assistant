# AI-Recipe-Assistant

# 🍳 RAG Recipe Assistant

The **RAG Recipe Assistant** is a lightweight, framework-free Retrieval-Augmented Generation (RAG) system designed to help users discover and understand recipes using natural language queries. By moving away from traditional keyword-based searching, this system leverages advanced semantic understanding to match complex user queries with relevant culinary data, rerank them for optimal accuracy, and generate clean, grounded cooking recommendations.

---

## 🧠 Core Capabilities

Instead of rigid keyword filters, the system intelligently interprets user intent, allowing for natural language queries like:
* *“What can I cook with chicken and eggs?”*
* *“Which recipes use garlic and soy sauce?”*
* *“Show me quick chicken recipes”*

The internal pipeline translates these queries into exact context matches, extracts the matching cooking instructions, and produces a highly relevant, naturally phrased output.

---

## ⚙️ Architecture & Pipeline

The project implements a modern, modular, multi-stage RAG workflow:


```

[ User Query ]
│
▼

1. Embeddings (Bi-Encoder) ──► Semantic Vector Search (Top Matches)
│
▼
2. Cross-Encoder Reranking ──► Re-scored & Re-ordered Results
│
▼
3. Context Builder         ──► Structured Prompt Formatting
│
▼
4. LLM Generation          ──► Grounded, Hallucination-Free Answer

```

### 1. Data Preparation
Recipe datasets are cleaned, parsed, and synthesized into a structured textual format optimized for embedding ingestion. Each document contains:
* **Recipe Name**
* **Ingredients**
* **Cooking Time**
* **Tags**

### 2. Embedding-Based Retrieval (Bi-Encoder)
* Raw documents are mapped into a high-dimensional vector space using `SentenceTransformers`.
* When a query is received, it is dynamically embedded using the same model.
* A fast similarity search computes distance metrics to fetch the initial top-K matching candidate recipes.

### 3. Cross-Encoder Re-ranking
To overcome the semantic limitations of independent vector lookups, a secondary **Cross-Encoder** model evaluates the user query and candidate recipe pairs simultaneously.
* **Benefits:** Drastically improves precision, corrects subtle ranking misalignments, and filters out noisy or marginal matches before sending text to the LLM.

### 4. Context Building
The highest-scoring re-ranked recipes are formatted into a clean, structured context block, complete with clear field delineations to guide the language model.

### 5. LLM Answer Generation
A HuggingFace-hosted **Qwen2.5-Instruct** model serves as the synthesis engine. It reads the strict context block and translates the underlying data into a well-structured, user-friendly response.

---

## 🛡️ Hallucination Control & Grounding

This pipeline is engineered from the ground up for strict informational fidelity:
* **System Prompt Constraints:** The LLM is strictly prohibited from pulling outside knowledge or introducing unverified recipes/ingredients.
* **Fallback Design:** If no relevant matching recipes cross the validation thresholds during retrieval and reranking, the engine automatically responds with a standardized fallback message:
  > *“Not found in provided recipes.”*

---

## 🎯 Key Features

* **True Semantic Search:** Captures intent, synonyms, and ingredients contextually without brittle exact-match strings.
* **Dual-Stage Ranking:** Bi-Encoder for scalable retrieval paired with a Cross-Encoder for pinpoint accuracy.
* **Structured Context Synthesis:** Clean prompt engineering pipeline ensuring absolute structural adherence.
* **Zero-Framework Blueprint:** Built raw and lightweight without heavy orchestrators (like LangChain or LlamaIndex), providing complete transparency over the pipeline logic.
* **Anti-Hallucination Guardrails:** Hard-grounded generation guarantees factual outputs tied exclusively to the input dataset.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* A Hugging Face API Token (for the Qwen2.5 inference client)


## 💡 Technical Significance

This project serves as a clear, production-ready implementation of core modern AI design patterns:

* **Retrieval-Augmented Generation (RAG):** Bridging static parametric knowledge with explicit external data streams.
* **Deep Text Re-ranking:** Maximizing information density inside LLM context windows to keep latency and API costs low.
* **Strict Prompt Engineering:** Practical implementation of grounding techniques to eliminate hallucinations in domain-specific use cases.

```

```
