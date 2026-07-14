#read files
import os 
#for reading .env file
from dotenv import load_dotenv
#text splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Embeddings turn text into mathematical vectors (numbers)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
#database that stores and searches those mathematical vectors
from langchain_community.vectorstores import Chroma
# Chain connects the database to the LLM to provide the final answer
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.core.messages import HumanMessage
#processes query + docs, calculates token level similarity
from sentence_transformers import CrossEncoder
#read the .env file and get the API key
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
llm = ChatAnthropic(model_name="claude-haiku-4-5", temperature=0)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rag_pipeline(query:str, all_docs):
    #retrieve the actual langchain docs from the all_docs list(which contains a dict of metadata)
    docs = _flatten_docs(all_docs)
    #split and embed the langchain docs
    vector_store = _embed_docs(docs)
    #prompt template
    template = rag_template
    #retrieve relevant docs and answer query
    retrieved_docs = _retrieve_docs(query,vector_store)
    #rerank retrieved docs 
    reranked_docs = _rerank_docs(query, retrieved_docs)
    #get the docs metadata and format for llm to cite
    context, sources = _build_cited_context(reranked_docs)
    response = _generate_answer(query, context)
    return {
        "query": query,
        "response": response,
        "sources": sources
    }

def analysis_pipeline():
    

#templates for diff modes
rag_template = PromptTemplate(
    input_variables=["context", "question"],
    template = """You are a document assistant. Answer questions using ONLY the context provided below.

Rules:
1. Answer only from the context. If the answer is not present, say: "I don't have that information in the uploaded documents."
2. Keep answers concise — one sentence for facts, up to 3 bullets for explanations.
3. Use inline citations in brackets, like [1] or [2], immediately after the sentence they support.
4. If context is partial, start with: "Based on available documents: "

Context:
{context}

Question: {question}
Answer:"""
)

analysis_template = PromptTemplate(
    
)


def _generate_answer(query: str, context: str):
    prompt = template.format(context=context, question=query)
    response = llm.invoke(prompt)
    return response.content

#take the retrieved docs metadata and format for the llm to cite
def _build_cited_context(retrieved_docs: list):
    context_parts = []
    sources = []

    for i, doc in enumerate(retrieved_docs, start=1):
        metadata = doc.metadata or {}
        source_name = metadata.get("source", "unknown source")
        page = metadata.get("page", None)
        chunk_id = metadata.get("chunk_id", None)
        rerank_score = metadata.get("rerank_score", None)

        display_page = page + 1 if isinstance(page, int) else None

        source_label = source_name
        if display_page is not None:
            source_label += f", page {display_page}"
        if chunk_id is not None:
            source_label += f", chunk {chunk_id}"

        context_parts.append(f"[{i}] {source_label}\n{doc.page_content}")

        sources.append({
            "citation": i,
            "source": source_name,
            "page": display_page,
            "chunk_id": chunk_id,
            "rerank_score": rerank_score,
            "content": doc.page_content
        })
    context = "\n\n".join(context_parts)
    return context, sources


def _rerank_docs(query: str, retrieved_docs: list):
    #rerank the chunks and return the top 4
    top_n = 4
    if not retrieved_docs:
        return []

    pairs = [(query, doc.page_content) for doc in retrieved_docs]
    scores = reranker.predict(pairs)

    scored_docs = list(zip(retrieved_docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    reranked_docs = []
    for rank, (doc, score) in enumerate(scored_docs[:top_n], start=1):
        if doc.metadata is None:
            doc.metadata = {}
        doc.metadata["rerank_score"] = float(score)
        doc.metadata["rerank_rank"] = rank
        reranked_docs.append(doc)

    return reranked_docs



def _retrieve_docs(query:str,vector_store):
    #retrieve 8 relevant chunks
    k = 8
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.invoke(query)
    return retrieved_docs


def _embed_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
     chunk_overlap=150,
     #structure aware text splitting
     separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],)

    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents( #turns vectors into searchable index used for semantics
        documents=splits, 
        embedding= embedding_model #converts text chunks into vectors
        )
    return vectorstore

#retrieve the actual docs
def _flatten_docs(all_docs):
    flat_docs = []
    for item in all_docs:
        flat_docs.extend(item["documents"])
    return flat_docs
