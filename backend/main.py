from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.rag_router import router as rag_router

app = FastAPI(
    title="RAG API",
    description="Multi-mode document assistant for RAG, quick stats, and analysis.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rag_router, prefix="/rag", tags=["rag"])


@app.get("/")
def root():
    return {
        "message": "RAG API is running.",
        "docs_url": "/docs",
        "routes": {
            "welcome": "/rag/welcome_page",
            "ask_query": "/rag/ask_query",
        },
    }

#add readiness checks 
@app.get("/health")
def health_check():
    return {"status": "ok"}