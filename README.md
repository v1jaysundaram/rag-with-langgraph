<p align="center">
  <img src="assets/banner.png" alt="RAG with LangGraph Banner" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/USERNAME/rag-with-langgraph?style=flat&color=yellow" alt="Stars"/>
  <img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https://github.com/USERNAME/rag-with-langgraph&title=Views" alt="Views"/>
  <a href="YOUR_LINKEDIN_URL"><img src="https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin" alt="LinkedIn"/></a>
  <a href="YOUR_YOUTUBE_CHANNEL_URL"><img src="https://img.shields.io/badge/YouTube-Subscribe-red?style=flat&logo=youtube" alt="YouTube"/></a>
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat" alt="License"/>
</p>

---

## Build production-grade RAG pipelines from scratch — 11+ techniques using LangGraph, FAISS, and OpenAI

Each episode is a standalone Python script implementing one RAG technique end-to-end. The series progresses from a basic retrieve-augment-generate loop through query optimization, context enrichment, and retrieval enhancement strategies.

---

## Concepts Covered

`Retrieval-Augmented Generation` · `LangGraph` · `Query Optimization` · `HyDE` · `HyPE` · `Contextual Chunk Headers` · `Semantic Chunking` · `Contextual Compression` · `Reranking` · `MMR` · `FAISS`

---

## Episodes

| # | Topic | Code | Video |
|---|-------|------|-------|
| 01 | Basic RAG | [📂 Code](./1_basic_rag.py) | [▶️ Watch](YOUR_VIDEO_URL) |
| 02 | Query Optimization — Typo Correction + Broadening | [📂 Code](./2_query_optimizations.py) | [▶️ Watch](YOUR_VIDEO_URL) |
| 03 | Sub-Query Decomposition | [📂 Code](./3_query_optimization_sub_query.py) | [▶️ Watch](YOUR_VIDEO_URL) |
| 04 | HyDE — Hypothetical Document Embeddings | [📂 Code](./4_query_optimization_hyde.py) | [▶️ Watch](YOUR_VIDEO_URL) |
| 05 | HyPE — Hypothetical Prompt Embeddings | [📂 Code](./5_query_optimization_hype.py) | [▶️ Watch](YOUR_VIDEO_URL) |
| 06 | Contextual Chunk Headers (CCH) | [📂 Code](./6_context_enrichment_cch.py) | [▶️ Watch](YOUR_VIDEO_URL) |
| 07 | Context Window Enhancement | [📂 Code](./7_context_enrichment_context_window_enhancement.py) | [▶️ Watch](YOUR_VIDEO_URL) |
| 08 | Semantic Chunking | [📂 Code](./8_context_enrichment_semantic_chunking.py) | [▶️ Watch](YOUR_VIDEO_URL) |
| 09 | Contextual Compression | [📂 Code](./9_context_enrichment_contextual_compression.py) | [▶️ Watch](YOUR_VIDEO_URL) |
| 10 | LLM-based Reranking | [📂 Code](./10_retrieval_enhancement_reranking.py) | [▶️ Watch](YOUR_VIDEO_URL) |
| 11 | MMR — Maximal Marginal Relevance | [📂 Code](./11_retrieval_enhancement_mmr.py) | [▶️ Watch](YOUR_VIDEO_URL) |
| … | More coming soon | — | — |

> New episodes drop regularly — [subscribe to get notified](YOUR_YOUTUBE_CHANNEL_URL)

---

## Architecture

Every pipeline follows the same LangGraph pattern — a `TypedDict` state flows through a directed acyclic graph of nodes:

```
Query → [Pre-processing nodes] → Retrieve → Augment → Generate → Answer
```

- **Retrieve** — queries a FAISS vector store (`k=4`), returns documents + context string
- **Augment** — formats a `PromptTemplate` with the query and retrieved context
- **Generate** — calls `ChatOpenAI` and returns the final answer

Each episode introduces new nodes or modifies the retrieve step to implement a different technique. The graph is compiled with `graph.compile()` and invoked with `workflow.invoke(input_state)`.

---

## Prerequisites

Before starting this series you should be comfortable with:
- **LangChain / LangGraph** — chains, retrievers, state graphs
- **Python** — functions, classes, type hints
- **LLMs / Embeddings** — what they are and how they're used in RAG

---

## Getting Started

**Clone and set up**
```bash
git clone https://github.com/USERNAME/rag-with-langgraph.git
cd rag-with-langgraph
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
pip install langchain langchain-community langchain-openai langgraph faiss-cpu pypdf python-dotenv scikit-learn langchain-experimental langchain-classic
```

**Configure environment**
```bash
cp .env.example .env
# Add your OPENAI_API_KEY
```

**Build the vector store** (required before running scripts 1–4, 9, 11)
```bash
jupyter notebook 0_rag_components.ipynb
```

**Run any episode**
```bash
python 1_basic_rag.py
python 4_query_optimization_hyde.py
# etc.
```

> Scripts 5–10 build their own vector stores on first run automatically.

---

## License

MIT © Vijay

---

## Connect

<p align="center">
  <a href="YOUR_LINKEDIN_URL">LinkedIn</a> &nbsp;·&nbsp; <a href="YOUR_YOUTUBE_CHANNEL_URL">YouTube</a>
</p>
