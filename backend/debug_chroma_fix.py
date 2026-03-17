import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "./chroma_db"

def check_id_in_chroma(target_ids):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        collection = client.get_collection(name="resumes")
        for tid in target_ids:
            res = collection.get(where={"candidate_id": tid}, limit=1)
            if res['ids']:
                print(f"FOUND in Chroma for ID: {tid}")
            else:
                print(f"NOT FOUND in Chroma for ID: {tid}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_id_in_chroma(["candidate_20260314_164333_976875", "candidate_20260314_214709_140620"])
