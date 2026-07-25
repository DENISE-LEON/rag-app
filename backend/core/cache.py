#in charge of caching
import hashlib
from backend.core.file_loader import get_doc_metadata

vector_store_cache = {}
def compute_doc_signature(files: list, contents_list: list[bytes]) -> str:
    file_hashes = []

    for file, contents in zip(files, contents_list):
        file_hash = hashlib.sha256(contents).hexdigest()
        file_hashes.append(file_hash)

    file_hashes.sort()
    combined = "||".join(file_hashes)

    return hashlib.sha256(combined.encode("utf-8")).hexdigest()
#for future session aware
def compute_session_signature(session_id: str -> str:
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()

def get_vectorestore(file_signature: str) -> vector_store:
    #if file signature not in cache, automatically return None with .get
    return vector_store_cache.get(file_signature)



def set_vectorestore(file_signature: str, vector_store):
    vector_store_cache[file_signature] = vector_store
