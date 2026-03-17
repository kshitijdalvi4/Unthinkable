import asyncio
import db

async def main():
    col = db.candidates_col()
    docs = await col.find({}).to_list(length=100)
    print("--- Candidates in MongoDB ---")
    for d in docs:
        print(f"ID: {d['_id']} | Name: {d.get('name')} | Email: {d.get('email')}")
    print("----------------------------")

if __name__ == "__main__":
    asyncio.run(main())
