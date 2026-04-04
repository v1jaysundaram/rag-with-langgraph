# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Requires an `OPENAI_API_KEY` in a `.env` file at the project root. The scripts load it via `python-dotenv`.

Activate the virtual environment before running anything:
```bash
source venv/Scripts/activate  # Windows Git Bash
```

Install dependencies (if setting up fresh):
```bash
pip install langchain langchain-community langchain-openai langgraph faiss-cpu pypdf python-dotenv scikit-learn langchain-experimental langchain-classic
```

## Running the scripts

Each numbered script is a standalone RAG pipeline. Run directly:
```bash
python 1_basic_rag.py
python 2_query_optimizations.py
python 3_query_optimization_sub_query.py
python 4_query_optimization_hyde.py
python 5_context_enrichment_hype.py
python 6_context_enrichment_cch.py
python 7_context_enrichment_context_window_enhancement.py
python "8_context_enrichment_semantic chunking.py"
python 9_context_enrichment_contextual_compression.py
```

The FAISS vector store at `rag_embeddings/` must exist before running scripts 1–4. It is built from `llm-book.pdf` (a 16-page PDF excerpt about LLMs) using `0_rag_components.ipynb` or `misc/rag_indexing.ipynb` — chunk size 400 characters, no overlap, `OpenAIEmbeddings`.

`5_context_enrichment_hype.py` builds its own store at `rag_embeddings_hype/` on first run (chunk size 1000, overlap 200). To rebuild it, delete `rag_embeddings_hype/` and rerun.

`6_context_enrichment_cch.py` builds its own store at `rag_embeddings_cch/` on first run (chunk size 1000, overlap 200). To rebuild it, delete `rag_embeddings_cch/` and rerun.

`7_context_enrichment_context_window_enhancement.py` builds its own store at `rag_embeddings_cwe/` on first run (chunk size 500, no overlap). To rebuild it, delete `rag_embeddings_cwe/` and rerun.

`8_context_enrichment_semantic chunking.py` builds its own store at `rag_embeddings_semantic/` on first run. To rebuild it, delete `rag_embeddings_semantic/` and rerun. Requires `langchain-experimental`.

`9_context_enrichment_contextual_compression.py` reuses the existing `rag_embeddings/` store (chunk size 400, no overlap). No separate store needed. Requires `langchain-classic`.

## Architecture

All pipelines follow the same LangGraph pattern:

**State** (`TypedDict`) flows through a directed acyclic graph of nodes:
1. Optional query pre-processing node(s)
2. `retrieve` — queries FAISS vector store (`k=4`), returns docs + context string
3. `augment` — formats a `PromptTemplate` with query + context
4. `generate` — calls `ChatOpenAI` (`gpt-4o` or `gpt-4o-mini`), returns `answer`

**Query optimization strategies** (each in its own file):
- `1_basic_rag.py` — vanilla retrieve → augment → generate
- `2_query_optimizations.py` — typo correction → query broadening → retrieve → augment → generate
- `3_query_optimization_sub_query.py` — decomposes query into 2–4 sub-queries via LLM (JSON list), retrieves for each, deduplicates docs by content
- `4_query_optimization_hyde.py` — HyDE: generates a hypothetical document of `chunk_size` characters, embeds it for retrieval instead of the raw query
- `5_context_enrichment_hype.py` — HyPE (Hypothetical Prompt Embeddings): at index time, generates 3 hypothetical questions per chunk and embeds those instead of the chunks; query matches against questions, original chunks are retrieved via metadata; uses `gpt-4o-mini` for question generation, `gpt-4o` for generation; separate FAISS store (`rag_embeddings_hype/`)
- `6_context_enrichment_cch.py` — Contextual Chunk Headers (CCH): enriches each chunk with an LLM-generated title + summary before embedding, using a separate FAISS store (`rag_embeddings_cch/`); uses `gpt-4o-mini` for enrichment, `gpt-4o` for generation
- `7_context_enrichment_context_window_enhancement.py` — Context Window Enhancement (CWE): retrieves top-k chunks then expands each result by fetching its neighbouring chunks (index ±1) from the docstore; stores chunk index in metadata at index time; separate FAISS store (`rag_embeddings_cwe/`)
- `8_context_enrichment_semantic chunking.py` — Semantic Chunking: splits text by semantic meaning using `SemanticChunker(OpenAIEmbeddings(), breakpoint_threshold_type='percentile', breakpoint_threshold_amount=90)` from `langchain_experimental`; chunks are variable-length, grouped by embedding similarity; separate FAISS store (`rag_embeddings_semantic/`)
- `9_context_enrichment_contextual_compression.py` — Contextual Compression: wraps the base FAISS retriever with `ContextualCompressionRetriever` + `LLMChainExtractor` (from `langchain_classic`); the compressor (`gpt-4o-mini`) extracts only the relevant portions of each retrieved chunk before passing context to the generator (`gpt-4o`); reuses `rag_embeddings/`

**Key shared pattern:** `RagState` is a `TypedDict` defined per-file. The graph is compiled with `graph.compile()` and invoked with `workflow.invoke(input_state)`.

## Notebooks

- `0_rag_components.ipynb` — walks through each RAG component (loader, splitter, embeddings, FAISS, retriever, augmentation) step by step; also rebuilds the vector store
- `misc/rag_indexing.ipynb` — standalone indexing notebook
- `misc/retrievers.ipynb` — experiments with different retriever configurations
- `similarity_comparison.ipynb` — paste two context strings (basic RAG vs CCH) and a query; computes TF-IDF cosine similarity scores to compare retrieval quality

## Skill routing

When the user's request matches an available skill, ALWAYS invoke it using the Skill
tool as your FIRST action. Do NOT answer directly, do NOT use other tools first.
The skill has specialized workflows that produce better results than ad-hoc answers.

Key routing rules:
- Product ideas, "is this worth building", brainstorming → invoke office-hours
- Bugs, errors, "why is this broken", 500 errors → invoke investigate
- Ship, deploy, push, create PR → invoke ship
- QA, test the site, find bugs → invoke qa
- Code review, check my diff → invoke review
- Update docs after shipping → invoke document-release
- Weekly retro → invoke retro
- Design system, brand → invoke design-consultation
- Visual audit, design polish → invoke design-review
- Architecture review → invoke plan-eng-review
- Save progress, checkpoint, resume → invoke checkpoint
- Code quality, health check → invoke health
