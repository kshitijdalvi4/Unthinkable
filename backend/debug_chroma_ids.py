import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "./chroma_db"

def list_chroma_ids():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        collection = client.get_collection(name="resumes")
        results = collection.get()
        ids = set()
        if results['metadatas']:
            for meta in results['metadatas']:
                if 'candidate_id' in meta:
                    ids.add(meta['candidate_id'])
                else:
                    ids.add("MISSING")
        
        print("Candidate IDs in ChromaDB:")
        for cid in ids:
            print(f"- {cid}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_chroma_ids()
