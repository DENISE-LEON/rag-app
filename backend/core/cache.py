#in charge of caching
import hashlib
from backend.core.file_loader import get_doc_metadata

vector_store_cache = {}
response_cache = {}

def compute_source_signature(files: list, contents_list: list[bytes]) -> str:
    file_hashes = []

    for file, contents in zip(files, contents_list):
        file_hash = hashlib.sha256(contents).hexdigest()
        file_hashes.append(file_hash)

    file_hashes.sort()
    combined = "||".join(file_hashes)

    return hashlib.sha256(combined.encode("utf-8")).hexdigest()

def compute_response_cache_signature(
    session_id: str,
    file_signature: str,
    mode: str,
    query: str,
) -> str:
    #normalize query so same queries with different casing are treated as the same
    normalized_query = " ".join(query.lower().split())

    combined = (
        f"{session_id}||"
        f"{file_signature}||"
        f"{mode}||"
        f"{normalized_query}"
    )

    return hashlib.sha256(combined.encode("utf-8")).hexdigest()

def get_vectorstore(file_session_signature: str):
    #if file signature not in cache, automatically return None with .get
    return vector_store_cache.get(file_session_signature)

def set_vectorstore(file_session_signature: str, vector_store):
    vector_store_cache[file_session_signature] = vector_store

def get_response(response_cache_signature:str) -> str:
    return response_cache.get(response_cache_signature)

def set_response(response_cache_signature: str, response: str, sources: list):
    response_cache[response_cache_signature] = 
    {"response": response, "sources": sources}


def hash_session_signature(session_id: str)-> str:
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()