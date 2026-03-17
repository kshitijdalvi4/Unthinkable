import asyncio
import db

async def list_candidates():
    col = db.candidates_col()
    cursor = col.find({})
    print("--- Candidate List ---")
    async for d in cursor:
        print(f"ID: {d.get('_id')} | Email: {d.get('email')} | Name: {d.get('name')}")
    print("----------------------")

if __name__ == "__main__":
    asyncio.run(list_candidates())
