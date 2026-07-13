#ingest files. detect types, load, hands back needed data frames, 
#langchain docs, and meta data needed by other functions
import io
import os 
import pandas as pd
import csv
import pdfplumber
import tempfile
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, CSVLoader, TextLoader, StructuredExcelLoader
from langchain.schema import Document
from fastapi import UploadFile

TABULAR_EXTENSIONS = {"csv", "xlsx", "xls", "tsv"}
TEXT_EXTENSIONS = {"pdf", "txt", "docx", "md"}

#1. load the docs

#dictionary to map file extensions to their respective loaders
loaders = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".xls": StructuredExcelLoader,
    ".xlsx": StructuredExcelLoader,
    ".tsv": CSVLoader,
}

#ingest pipeline
#1. receive the files
#2. load all files to langchain incase needed for RAG or Analysis mode
#3. classify files by type (tabular or text)
#4. load tabular files to dataframes
#5. return dict
# return metadat to be later used by other functions(list of tabular, text, langchain docs)
# return bools indicating if there are tabular or text files to be used by determine_best_mode
async def ingest_files(files: list[UploadFile]) -> dict:
    all_docs = []
    tabular_results = []
    text_results = []

    for file in files:
        contents = await file.read()
        docs = load_documents_to_langchain(file.filename, contents)
        all_docs.append(docs)

        file_type = get_file_type(file.filename)
        file_type = reclassify_if_tabular(file.filename, contents, file_type)
   
        if file_type == "tabular":
            tab_result = load_tabular_to_dfs(file.filename, contents)
            tabular_results.append(tab_result)
        elif file_type == "text":
            text_results.append(docs)

    return {
        "all_docs": all_docs,
        "has_tabular": len(tabular_results) > 0,
        "has_text": len(text_results) > 0,
        "tabular_files": tabular_results,
        "text_files": text_results
    }
#load docs to langchain document objects, which are used for analysis and RAG
def load_documents_to_langchain(filename: str, contents: bytes):
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        if ext in loaders:
            loader_cls = loaders[ext]
            loader = loader_cls(tmp_path)
            documents = loader.load()
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    finally:
        os.remove(tmp_path)

    return {
        "filename": filename,
        "type": documents[0].metadata.get("source", "unknown") if documents else "unknown",
        "documents": documents,
        "page_count": len(documents),
        "extension": ext,
    }

def load_tabular_to_dfs(file_name: str, contents: bytes) -> dict:
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else "csv"
    if ext in {"xlsx", "xls"}:
        df = pd.read_excel(io.BytesIO(contents))
        dfs = [df]  # Wrap the DataFrame in a list for consistency
    elif ext == "pdf":
        dfs = _extract_pdf_tables(contents)
    else:
        df = pd.read_csv(io.BytesIO(contents), sep=None, engine="python")
        dfs = [df]
    
    
    return {
        "filename": file_name,
        "type": "tabular",
        "dataframes": dfs,
        "columns": [list(df.columns) for df in dfs],
        "row_counts": [len(df) for df in dfs],
        "extension": ext
    }
#first classify by extension
def get_file_type(file_name: str) -> str:
    #rsplit starts from the right
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    if ext in TABULAR_EXTENSIONS:
        return "tabular"
    elif ext in TEXT_EXTENSIONS:
        return "text"
    else:
        return "unknown"

#catches txt or pdf files that have tabular content
def reclassify_if_tabular(file_name: str, contents: bytes, current_type:str) -> str:
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

    if ext == "pdf":
        return _check_pdf_for_tables(contents)
    if ext in {"txt", "md", ""} or current_type == "unknown":
        return _check_text_for_delimiters(contents)

    return current_type
def _check_text_for_delimiters(contents: bytes) -> bool:
    try:
        #reads the first 2048 bytes of the file, converts bytes to readable string, skip invalid characters
        sniff_content = contents[:2048].decode("utf-8", errors="ignore")
        #detect delimiters
        sniffer = csv.Sniffer().sniff(sniff_content)
        if sniffer.delimiter in {',', ';', '\t', '|'}:
            return "tabular"
    except csv.Error:
        pass
    return "text"
#pdf files are treated differently from text files
#cannot be directly loaded into a df 
def _check_pdf_for_tables(contents: bytes) -> str:
    try:
        #bytesIO allows us to treat bytes as a file-like object which can be read by pdfplumber
        with pdfplumber.open(io.BytesIO(contents)) as pdf: 
            for page in pdf.pages:
                tables = page.extract_tables()
                if tables:
                    return "tabular"
    except Exception as e:
        pass
    return "text"

def _extract_pdf_tables(contents: bytes) -> list:
    dataframes = []
    try:
        with pdfplumber.open(io.BytesIO(contents)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table in tables:
                    if not table:
                        continue
                    headers = table[0]
                    rows = table[1:]
                    headers = [
                        str(h) if h is not None else f"col_{i}"
                        for i, h in enumerate(headers)
                    ]

                    df = pd.DataFrame(rows, columns=headers)
                    df.attrs["source_page"] = page_num + 1
                    dataframes.append(df)

    except Exception as e:
        print(f"Warning: PDF table extraction failed — {e}")

    return dataframes

