import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "./chroma_db"

def inspect_chroma():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        collection = client.get_collection(name="resumes")
        print(f"--- ChromaDB 'resumes' count: {collection.count()} ---")
        
        # Get a sample of metadata
        results = collection.get(limit=5)
        if results['metadatas']:
            print("Sample Metadatas:")
            for meta in results['metadatas']:
                print(meta)
        else:
            print("No metadata found.")

        # Let's check for specific candidate ID
        target_id = "candidate_20260314_164333_976875"
        res = collection.get(where={"candidate_id": target_id}, limit=1)
        if res['ids']:
            print(f"FOUND {len(res['ids'])} docs for ID: {target_id}")
        else:
            print(f"NOT FOUND in Chroma for ID: {target_id}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_chroma()
