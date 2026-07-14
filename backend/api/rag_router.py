from fastapi import APIRouter, UploadFile, File, Form, HTTPException  
import json
from pydantic import BaseModel
from enum import Enum
from backend file_loader import ingest_files
from backend.mode_pipelines import pandas_pipeline, rag_pipeline, determine_best_mode

class QueryMode(str, Enum):
    ANALYSIS = "analysis"
    QUICKSTATS = "quickstats"
    RAG = "rag"

    @property
    def description(self):
        descriptions = {
            QueryMode.ANALYSIS: "You want to analyze your data.",
            QueryMode.QUICKSTATS: "You want to get quickstats about your data.",
            QueryMode.RAG: "You want to ask questions about your documents.",
        }
        return descriptions.get(self, "")


class UserIntent(str, Enum):
    DATA = "data"
    RAG = "rag"
    UNSURE = "unsure"

    @property
    def description(self):
        descriptions = {
            UserIntent.DATA: "You want to analyze your data.",
            UserIntent.RAG: "You want to ask questions about your documents.",
            UserIntent.UNSURE: "You are unsure about what you want to do.",
        }
        return descriptions.get(self, "")

class QueryRequest(BaseModel):
    query: str
    mode: QueryMode


router = APIRouter()
mode = QueryMode
@router.get("/welcome_page")
async def welcome_pg(intent: UserIntent): 
    match intent:
        case UserIntent.DATA:
            default_mode = QueryMode.QUICKSTATS
        case UserIntent.RAG:
            default_mode = QueryMode.RAG
        case UserIntent.UNSURE:
            default_mode = QueryMode.RAG

    return {"message": "Welcome to the advanced AI chatbot!",
            "default_mode": default_mode,
            "intents": {intent.value: intent.description for intent in UserIntent}
            
            }

@router.post("/ask_query")
async def ask_query(
        query_request: QueryRequest,
        file: UploadFile = File(...),
        want_to_switch: bool = Form(...),
        ):
        mode = query_request.mode
    #1. Ingest files and classify them
    ingested = await ingest_files([file])
    all_docs = ingested["all_documents"]
    has_tabular = ingested["has_tabular"]
    has_text = ingested["has_text"]
    tabular_files = ingested["tabular_files"]
    text_files = ingested["text_files"]

    #2. Determine best mode based on file types + query
    suggested_mode, reason = determine_best_mode(query_request.query, has_tabular, has_text)
    if suggested_mode != mode:
        if not want_to_switch:
            return {
                "message": f"Suggested mode: {suggested_mode}",
                "reason": reason,
                "suggested_mode": suggested_mode,
                "awaiting_confirmation": True
            }
        else:
            mode = QueryMode(suggested_mode)
    #match mode to pipeline
    match mode:
        case QueryMode.ANALYSIS:
            # Handle analysis mode
            return {"message": "Analysis mode selected", "query": query_request.query}
        case QueryMode.QUICKSTATS:
            response = pandas_pipeline(query_request.query, tabular_files)
            return {"message": "Quickstats mode selected", "query": query_request.query, "response": response}
        case QueryMode.RAG: 
            response = rag_pipeline(query_request.query, all_docs)
            return {"message": "RAG mode selected", "query": query_request.query, "response": response}
        case _:
            raise HTTPException(status_code=400, detail="Invalid mode selected")