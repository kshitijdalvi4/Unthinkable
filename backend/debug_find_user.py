import asyncio
import db

async def search_candidate(email):
    col = db.candidates_col()
    doc = await col.find_one({"email": email})
    if doc:
        print(f"FOUND: {doc['_id']} | Email: {doc.get('email')} | Resume Path: {doc.get('pdf_path')}")
        return doc
    else:
        print("NOT FOUND by email.")
        return None

if __name__ == "__main__":
    email = "kshitijdalvi22@gmail.com"
    asyncio.run(search_candidate(email))
