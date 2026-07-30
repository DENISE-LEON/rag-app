# RAG Document Q&A and Data Analysis API

A FastAPI backend that lets users upload documents and tabular files, ask grounded questions about them, and automatically route each request to the right workflow: **RAG** (document Q&A), **Analysis** (documents + tabular statistics), or **Quickstats** (deterministic pandas aggregations). Answers are generated only from user-provided evidence, with inline citations back to the uploaded material.

Frontend UI repository: [rag_app_ui](https://github.com/DENISE-LEON/rag_app_ui).

## Features

- **Multi-file upload** in a single request: PDF, CSV, XLSX/XLS, TSV, TXT, MD.
- **Content-aware file classification**: files are first classified by extension, then re-checked by content. PDFs are inspected with `pdfplumber` for real tables, and `.txt`/`.md`/unknown files are sniffed with `csv.Sniffer` for delimiters — so a `.txt` holding a delimited table is treated as tabular, not text.
- **Automatic mode suggestion**: `determine_best_mode` combines the `has_tabular` / `has_text` signals with keyword intent detection on the query, and returns both a suggested mode and a human-readable reason.
- **Mode-confirmation round trip**: if the suggested mode differs from the submitted mode, the API returns `awaiting_confirmation: true` instead of guessing. The client re-sends with `mode_confirmed` and `want_to_switch` to accept or decline the suggestion.
- **Two-layer caching** (see below): a content-keyed vectorstore cache and a session-keyed response cache.
- **Reranked retrieval**: 8 chunks retrieved from Chroma, then reranked with a `cross-encoder/ms-marco-MiniLM-L-6-v2` CrossEncoder down to the top 4, with rerank scores attached to each source.
- **Enforced citations**: prompts require every factual sentence to carry a `[n]` citation, and the API returns a structured `sources` list (source name, page, chunk id, rerank score, chunk content).
- **Query translation for analysis mode**: the raw question is rewritten by the LLM into a retrieval-optimized query using tabular schema hints (file names, column names, row counts) before retrieval.
- **Deterministic quickstats with LLM fallback**: filters and aggregates are first parsed with regex/keyword dispatch. Only if intent is still unknown does it fall back to `_smart_interpret_query`, which asks the LLM for a strict JSON plan (`aggregate`, `target_column`, `filter`) that is then validated before use.

## Caching

Caching lives in `backend/core/cache.py` and uses two separate in-memory dictionaries with different keys — this distinction matters:

| Cache | Key | Scope | What it stores |
|---|---|---|---|
| `vector_store_cache` | `source_signature` — SHA-256 over the sorted SHA-256 hashes of each file's **bytes** | Content only, no session | The built Chroma vectorstore, so identical file sets skip re-splitting and re-embedding |
| `response_cache` | SHA-256 over `session_id \|\| source_signature \|\| mode \|\| normalized_query` | Session-scoped | `{response, sources}` for an exact repeat request |

Consequences worth knowing:

- A response-cache hit **short-circuits the entire pipeline** — no retrieval, no rerank, no LLM call — and the response is returned with `cached: true`.
- The source signature hashes file **contents only**, not filenames or sizes. Two identically-named files with different bytes produce different signatures; two differently-named files with identical bytes produce the same signature.
- Queries are normalized (lowercased, whitespace-collapsed) before hashing, so casing and extra spaces still hit the cache.
- Same files + different `session_id` = vectorstore **hit**, response cache **miss**. Embeddings are reused, but the answer is regenerated.
- Both caches are plain process-local dicts: they reset on server restart, are not shared across workers, and are currently unbounded (no TTL or eviction).

## Repositories

| Repository | Purpose |
|---|---|
| [rag-app](https://github.com/DENISE-LEON/rag-app) | FastAPI backend for file ingestion, routing, caching, retrieval, and response generation. |
| [rag_app_ui](https://github.com/DENISE-LEON/rag_app_ui) | Frontend for selecting intent, uploading files, sending queries, and displaying answers and sources. |

## Stack

- **Backend:** Python, FastAPI, Pydantic, Uvicorn.
- **RAG / AI:** LangChain, Anthropic Claude (`claude-haiku-4-5`), HuggingFace `all-MiniLM-L6-v2` embeddings, Chroma vector store, sentence-transformers CrossEncoder reranker.
- **Data handling:** pandas, pdfplumber, extension- and content-based loaders.
- **Frontend:** HTML, CSS, JavaScript (separate repository).

## Project structure

```
backend/
  main.py                 # FastAPI app, CORS, /health, router mounted at /rag
  config.py               # loads ANTHROPIC_API_KEY, builds the shared ChatAnthropic llm
  api/rag_router.py       # /welcome_page and /ask_query endpoints, mode + cache orchestration
  core/file_loader.py     # ingestion, type detection/reclassification, LangChain docs + DataFrames
  core/mode_helper.py     # determine_best_mode
  core/cache.py           # source + response signatures, vectorstore and response caches
  core/rag.py             # rag_pipeline, analysis_pipeline, retrieval, rerank, prompts
  core/aggregates.py      # pandas_pipeline, filter/aggregate parsing, smart interpretation
```

## API

### `GET /rag/welcome_page?intent=data|rag|unsure`

Returns a welcome message, a `default_mode` for the chosen intent (`data` → quickstats, `rag`/`unsure` → rag), and the available intents with descriptions.

### `POST /rag/ask_query` (multipart form)

| Field | Type | Notes |
|---|---|---|
| `query_request` | string (JSON) | `{"query": "...", "mode": "rag" \| "analysis" \| "quickstats"}`; invalid JSON returns 422 |
| `files` | file[] | one or more uploads, required |
| `session_id` | string | required; scopes the response cache only |
| `want_to_switch` | bool | default `false`; accept the suggested mode |
| `mode_confirmed` | bool | default `false`; marks that the suggestion was already shown to the user |

Request flow:

1. Validate `query_request` JSON into a `QueryRequest`.
2. Read all file bytes and compute `source_signature`.
3. Ingest files → LangChain docs, DataFrames, `has_tabular` / `has_text`.
4. Suggest a mode; if it differs from the submitted mode and `mode_confirmed` is false, return `awaiting_confirmation`.
5. Compute the response-cache signature and return the cached answer if present.
6. Otherwise dispatch to `analysis_pipeline`, `pandas_pipeline`, or `rag_pipeline`, cache the result, and return it.

## Run locally

1. Clone the repository and create/activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   pip install langchain-chroma
   ```

   `backend/core/rag.py` imports `langchain_chroma`, which is not yet pinned in `requirements.txt`.
3. Create a `.env` file with `ANTHROPIC_API_KEY=...`.
4. Start the server:

   ```bash
   uvicorn backend.main:app
   ```

   Auto-reload is best avoided locally if it watches `.venv` and causes repeated restarts. Interactive docs are at `/docs`, and `/health` returns a readiness payload.

For the frontend, clone [rag_app_ui](https://github.com/DENISE-LEON/rag_app_ui) and point its API base URL at the backend (currently `http://localhost:8000`).

## Current limitations

- Both caches are in-memory and process-local: they clear on restart, are not shared across workers, and have no size limit, TTL, or eviction policy.
- Quickstats returns an empty `sources` list — a documented temporary gap rather than a design decision.
- The quickstats pipeline has no `summarize` intent branch yet; summary-style queries that parse to no filter or aggregate return "unable to answer".
- There are no automated tests in the repository, so changes rely on manual smoke testing of each mode.
- `hash_session_signature` in `cache.py` is currently unused.
- A debug `print("using cached response")` remains in the router instead of structured logging.
- Follow-up questions depend on the active upload flow rather than a persistent document memory layer.
- Latency varies with file size, embedding-model load time, and reranking.

## Future improvements

- Move caching to a shared store (e.g. Redis) with TTL and eviction so it survives restarts and works across workers.
- Return sources from quickstats and add a summarize intent.
- Add automated tests for ingestion, mode selection, cache behavior, and pipeline dispatch.
- Replace debug prints with structured logging and add error handling around LLM calls.
- Improve frontend file-state handling and conversation continuity.

## Why this project

This project reflects an interest in building tools that make information easier to access and reason over. It combines document retrieval, structured data handling, deterministic data analysis, and caching into a practical application that answers questions from user-provided evidence instead of unsupported guesses.
