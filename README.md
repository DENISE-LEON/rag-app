# RAG Document Q&A and Data Analysis API

A FastAPI-based application that lets users upload documents and tabular files, ask grounded questions about them, and route requests across RAG, analysis, and quick-stat workflows. The backend focuses on evidence-based responses over user-provided files, while a separate frontend repository provides the browser interface.[1][2]

Frontend UI repository: [rag_app_ui](https://github.com/DENISE-LEON/rag_app_ui).[3]

## Features

- Upload and process multiple files in a single request, including PDFs, CSV, XLSX, TXT, TSV, Markdown, and similar document/data formats supported by the pipeline.
- Route requests across different modes depending on the user’s goal: document Q&A, analysis, and quick stats over structured data.[1][2]
- Generate grounded answers using retrieval over uploaded content rather than relying on unscoped model responses.
- Surface source-backed outputs so users can trace responses back to uploaded materials.
- Provide a lightweight frontend experience for choosing intent, uploading files, selecting a mode, and viewing answers plus sources.[1][2]

## Repositories

| Repository | Purpose |
|---|---|
| [rag-app](https://github.com/DENISE-LEON/rag-app) | FastAPI backend for file ingestion, retrieval, routing, and response generation. |
| [rag_app_ui](https://github.com/DENISE-LEON/rag_app_ui) | Frontend interface for selecting intent, uploading files, sending queries, and displaying responses.[1][2] |

## Stack

- **Backend:** Python, FastAPI.[4]
- **RAG / AI:** LangChain, Anthropic Claude, HuggingFace embeddings, Chroma.
- **Data handling:** Pandas, pdfplumber, file-type specific loaders for uploaded content.
- **Frontend:** HTML, CSS, JavaScript in a separate UI repository.[1][2][5]

## How it works

1. The user chooses an intent and enters the chat workspace in the frontend UI.[1][2]
2. The user uploads one or more files and submits a question with a selected mode.[1][2]
3. The backend ingests the uploaded files, routes the request, and generates a response grounded in the uploaded material.
4. The UI displays the answer and, when available, the returned sources.[1]

## Run locally

### Backend

1. Clone the backend repository.
2. Create and activate a virtual environment.
3. Install project dependencies.
4. Set required environment variables, including `ANTHROPIC_API_KEY` for Claude-based responses.
5. Start the FastAPI server, preferably without auto-reload if local file-watching causes repeated restarts.

Example:

```bash
uvicorn backend.main:app
```

### Frontend

1. Clone the frontend repository: [rag_app_ui](https://github.com/DENISE-LEON/rag_app_ui).[3]
2. Open the frontend locally and ensure the API base URL points to the FastAPI backend, currently configured for `http://localhost:8000` in the provided UI logic.[1]

## Current limitations

- Follow-up questions currently depend on the active upload flow rather than a full session-based document memory layer.[1]
- Local development can be noisy if FastAPI is run with broad auto-reload settings and watches `.venv` files.
- Performance can vary depending on file size, model loading, and retrieval steps during local runs.

## Future improvements

- Add session-aware caching so follow-up questions can reuse previously uploaded document context without re-uploading files.
- Improve frontend file-state handling and conversation continuity.[1]
- Continue reducing latency and polishing the UX for local demos and future deployment.

## Why this project

This project reflects an interest in building tools that make information easier to access and reason over. It combines document retrieval, structured data handling, and a simple user interface into a practical application that answers questions based on user-provided evidence instead of unsupported guesses.
