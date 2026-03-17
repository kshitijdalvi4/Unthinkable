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

# --- Re-implementing necessary pieces for standalone run ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_PATH = "./chroma_db"
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

class GeminiEmbeddings(Embeddings):
    def __init__(self, client):
        self.client = client
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            # Force 768 dimensions for Chroma compatibility
            response = self.client.models.embed_content(
                model="gemini-embedding-001", 
                contents=text,
                config={'output_dimensionality': 768}
            )
            embeddings.append(response.embeddings[0].values)
        return embeddings
    def embed_query(self, text: str) -> List[float]:
        # Force 768 dimensions for Chroma compatibility
        response = self.client.models.embed_content(
            model="gemini-embedding-001", 
            contents=text,
            config={'output_dimensionality': 768}
        )
        return response.embeddings[0].values

embeddings = GeminiEmbeddings(gemini_client)

async def recover_user_in_chroma(candidate_id, pdf_path):
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} does not exist.")
        return

    print(f"Starting ingestion for {candidate_id} from {pdf_path} using gemini-embedding-001...")
    
    loader = PyPDFLoader(pdf_path)
    documents = await asyncio.to_thread(loader.load)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    
    for doc in docs:
        doc.metadata['candidate_id'] = candidate_id
        doc.metadata['candidate_name'] = "Kshitij Dalvi"
    
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name="resumes"
    )
    print(f"Successfully ingested {len(docs)} chunks for {candidate_id}")

if __name__ == "__main__":
    cid = "candidate_20260314_164333_976875"
    path = "./uploads/candidate_20260314_214709_140620.pdf" 
    asyncio.run(recover_user_in_chroma(cid, path))
