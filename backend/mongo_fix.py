import asyncio
import os
import sys
from datetime import datetime

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db import candidates_col, get_client

async def test_and_cleanup():
    print("Testing MongoDB Connection...")
    try:
        client = get_client()
        # The ismaster command is cheap and does not require auth.
        await client.admin.command('ismaster')
        print("✅ Connection Successful!")
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        return

    print("\nChecking for duplicates...")
    pipeline = [
        {"$group": {
            "_id": "$email",
            "count": {"$sum": 1},
            "ids": {"$push": "$_id"},
            "docs": {"$push": "$$ROOT"}
        }},
        {"$match": {"count": {"$gt": 1}, "_id": {"$ne": None}}}
    ]
    
    duplicates = await candidates_col().aggregate(pipeline).to_list(length=100)
    
    if not duplicates:
        print("No duplicates found with matching email.")
    else:
        print(f"Found {len(duplicates)} email address(es) with duplicates.")
        for group in duplicates:
            email = group["_id"]
            ids = group["ids"]
            print(f"Email: {email} | IDs: {ids}")
            
            # Keep the first one, delete the rest
            primary_id = ids[0]
            to_delete = ids[1:]
            
            # Merge additional_info if possible
            merged_info = {}
            for doc in group["docs"]:
                merged_info.update(doc.get("additional_info", {}))
            
            await candidates_col().update_one(
                {"_id": primary_id},
                {"$set": {"additional_info": merged_info, "updated_at": datetime.utcnow().isoformat()}}
            )
            
            for d_id in to_delete:
                await candidates_col().delete_one({"_id": d_id})
                print(f"  Deleted duplicate ID: {d_id}")
            
            print(f"  Merged info into Primary ID: {primary_id}")

    print("\nCleanup Complete.")

if __name__ == "__main__":
    asyncio.run(test_and_cleanup())
