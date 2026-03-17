import asyncio
import os
import certifi
from motor.motor_asyncio import AsyncIOMotorClient
from google import genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from typing import List
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_PATH = "./chroma_db"
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "unthinkable")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

class GeminiEmbeddings(Embeddings):
    def __init__(self, client):
        self.client = client
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            response = self.client.models.embed_content(
                model="gemini-embedding-001", 
                contents=text,
                config={'output_dimensionality': 768}
            )
            embeddings.append(response.embeddings[0].values)
        return embeddings
    def embed_query(self, text: str) -> List[float]:
        response = self.client.models.embed_content(
            model="gemini-embedding-001", 
            contents=text,
            config={'output_dimensionality': 768}
        )
        return response.embeddings[0].values

embeddings = GeminiEmbeddings(gemini_client)

async def sync_all_resumes():
    print("--- Starting Global RAG Sync ---")
    
    # Init MongoDB
    client = AsyncIOMotorClient(MONGO_URI, tlsCAFile=certifi.where())
    db = client[MONGO_DB]
    col = db["candidates"]
    
    # Init Chroma
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="resumes"
    )

    # Get all candidates
    cursor = col.find({})
    candidates = await cursor.to_list(length=1000)
    
    for cand in candidates:
        cid = cand["_id"]
        pdf_path = cand.get("pdf_path")
        name = cand.get("name", "Unknown")
        
        if not pdf_path or not os.path.exists(pdf_path):
            print(f"[SKIP] No local PDF for {name} ({cid})")
            continue
            
        print(f"[SYNC] Processing {name} ({cid})...")
        
        # 1. Delete old chunks
        try:
            vectorstore.delete(where={"candidate_id": cid})
        except Exception:
            pass
            
        # 2. Ingest
        try:
            loader = PyPDFLoader(pdf_path)
            documents = await asyncio.to_thread(loader.load)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            docs = text_splitter.split_documents(documents)
            
            for doc in docs:
                doc.metadata['candidate_id'] = cid
                doc.metadata['candidate_name'] = name
            
            Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=CHROMA_PATH,
                collection_name="resumes"
            )
            print(f"  -> Ingested {len(docs)} chunks")
        except Exception as e:
            print(f"  -> ERROR: {e}")

    print("--- Global RAG Sync Complete ---")

if __name__ == "__main__":
    asyncio.run(sync_all_resumes())
