import os
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import chromadb
from agents.agents import GeminiEmbeddings

# Setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDlYWCbgRh9Vu17zk5T1yUDU5l3wGDTZ-E")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
embeddings = GeminiEmbeddings(client=gemini_client)

CHROMA_PATH = "./chroma_db_test"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name="resumes",
    embedding_function=embeddings
)

# Test File
pdf_path = r"C:\Users\kshit\Downloads\Kshitij_Dalvi_Online_Resume_Feb26.pdf"

try:
    print("Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print(f"Loaded {len(documents)} pages. Splitting...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"Split into {len(chunks)} chunks. Adding to ChromaDB...")
    
    # Store in ChromaDB
    collection.add(
        ids=[f"test_chunk_{i}" for i in range(len(chunks))],
        documents=[chunk.page_content for chunk in chunks],
        metadatas=[{"source": pdf_path} for chunk in chunks]
    )
    print("SUCCESS: Chunks added to ChromaDB.")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
