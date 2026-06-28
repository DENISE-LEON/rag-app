#read files
import os 
#for reading .env file
from dotenv import load_dotenv
#doc loaders, diff types of files
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, CSVLoader, TextLoader
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
#read the .env file and get the API key

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

#1. load the docs

#dictionary to map file extensions to their respective loaders
loaders = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader
}

#prompt template handles how AI responds(grounding, conciseness, and source citation)
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template = """You are a document assistant. Answer questions using ONLY the context provided below.

Rules:
1. Answer only from the context. If the answer is not present, say: "I don't have that information in the uploaded documents."
2. Keep answers concise — one sentence for facts, up to 3 bullets for explanations.
3. End every answer with: Source: [brief description of the relevant document or section]
4. If context is partial, start with: "Based on available documents: "

Context:
{context}

Question: {question}
Answer:"""
)

def query_translation(query):
    return

def create_directory_loader(file_ext, loader_cls):
    # This helper function creates a loader for a specific extension
    return DirectoryLoader(
        path="input_docs", #folder where the documents are stored
        glob=f"**/*{file_ext}", # This looks for the extension in the folder, glob is which files to load
        loader_cls=loader_cls #which loader to use for this extension
    )

# Create a list of all loaders we want to run
loader_list = [create_directory_loader(ext, cls) for ext, cls in loaders.items()] #shorthand for loop 
docs = []
for loader in loader_list:
    # Load the documents and add them to main list
    docs.extend(loader.load())
print(f"Loaded {len(docs)} document pages.")

loaded_extensions = set()
csv_file_paths = []

for doc in docs:
    source = doc.metadata.get("source", "")
    ext = os.path.splitext(source)[1].lower()
    loaded_extensions.add(ext)
    if ext == ".csv":
        csv_file_paths.append(source)

print(f"Loaded file extensions: {', '.join(loaded_extensions)}")
print(f"CSV file paths: {', '.join(csv_file_paths)}")

#2. split the docs into chunks 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = text_splitter.split_documents(docs)
#chunk_overlap ensures that the end of one piece and the start of the next share some text so context isn't lost.

#3. embedding & storage 
vectorstore = Chroma.from_documents( #turns vectors into searchable index used for semantics
        documents=splits, 
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") #converts text chunks into vectors
    )

#4. retrieval setup
llm = ChatAnthropic(model_name="claude-haiku-4-5", temperature=0)  #temperature how creative the AI's responses are, 0 means more factual
qa_chain = RetrievalQA.from_chain_type( #retrievalQA connects the database to the LLM to provide the final answer
    llm=llm,
    chain_type="stuff", #"stuff" the relevant notes into the prompt sent to the AI.
    retriever=vectorstore.as_retriever(), #retriever is the database that stores and searches those mathematical vectors
    chain_type_kwargs={"prompt": prompt_template}
    )

#5. query time
print("Ask a question about your documents or type 0 to quit:")
while True:
    query = input("Your question:")
    if query == "0":
        break
    response = qa_chain.invoke(query)
    print(f"\nAI Response: {response['result']}")
