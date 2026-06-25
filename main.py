#read files
import os 
#for reading .env file
from dotenv import load_dotenv
#doc loaders, diff types of files
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, CSVLoader, TextLoader
#text splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Embeddings turn text into mathematical vectors (numbers)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
#database that stores and searches those mathematical vectors
from langchain_community.vectorstores import Chroma
# Chain connects the database to the LLM to provide the final answer
from langchain.chains import RetrievalQA

#read the .env file and get the API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

#1. load the docs

#dictionary to map file extensions to their respective loaders
loaders = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader
}

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

#2. split the docs into chunks 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = text_splitter.split_documents(docs)
#chunk_overlap ensures that the end of one piece and the start of the next share some text so context isn't lost.

#3. embedding & storage 
vectorstore = Chroma.from_documents( #turns vectors into searchable index used for semantics
        documents=splits, 
        embedding=OpenAIEmbeddings() #converts text chunks into vectors
    )

#4. retrieval setup
llm = ChatOpenAI(model_name="gpt-4", temperature=0)  #temperature how creative the AI's responses are, 0 means more factual
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", #"stuff" the relevant notes into the prompt sent to the AI.
    retriever=vectorstore.as_retriever()
    )

#5. query time
print("Ask a question about your documents or type 0 to quit:")
while True:
    query = input("Your questimon:")
    if query == "0":
        break
    response = qa_chain.invoke(query)
    print(f"\nAI Response: {response['result']}")
