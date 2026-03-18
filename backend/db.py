"""
MongoDB async client for Unthinkable backend.
Collections:
  - candidates: resume data + additional_info answers
  - applications: application log per job
"""
import os
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

import certifi
from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URI = os.getenv("MONGO_URI", "")
MONGO_DB  = os.getenv("MONGO_DB",  "unthinkable")

_client: Optional[AsyncIOMotorClient] = None

def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        # Use tls=True without specifying tlsCAFile so Linux (Render) uses system CA certs.
        # On Windows, certifi is needed but on Linux the system certs work fine.
        import sys
        kwargs = {
            "serverSelectionTimeoutMS": 5000,
            "connectTimeoutMS": 5000,
        }
        if sys.platform == "win32":
            kwargs["tlsCAFile"] = certifi.where()
        print(f"[DB] Creating MongoDB client (platform: {sys.platform})")
        _client = AsyncIOMotorClient(MONGO_URI, **kwargs)
    return _client

def get_db():
    return get_client()[MONGO_DB]

def candidates_col():
    return get_db()["candidates"]

def applications_col():
    return get_db()["applications"]


# ─── Candidate helpers ───────────────────────────────────────────────────────

async def get_candidate(candidate_id: str) -> Optional[dict]:
    """Return full candidate document or None."""
    print(f"[DB] get_candidate: {candidate_id}")
    doc = await candidates_col().find_one({"_id": candidate_id})
    print(f"[DB] get_candidate result: {'Found' if doc else 'Not Found'}")
    return doc

async def get_candidate_by_email(email: str) -> Optional[dict]:
    """Find a candidate by email for deduplication."""
    if not email:
        return None
    print(f"[DB] get_candidate_by_email: {email}")
    doc = await candidates_col().find_one({"email": email})
    print(f"[DB] get_candidate_by_email result: {'Found' if doc else 'Not Found'}")
    return doc

async def upsert_candidate(candidate_id: str, data: dict) -> None:
    """Insert or update a candidate document (initial upload)."""
    payload = {k: v for k, v in data.items()}
    payload["_id"] = candidate_id
    payload.setdefault("additional_info", {})
    payload.setdefault("knowledge_base", {})
    payload.setdefault("created_at", datetime.utcnow().isoformat())
    payload["updated_at"] = datetime.utcnow().isoformat()

    await candidates_col().update_one(
        {"_id": candidate_id},
        {"$set": payload},
        upsert=True,
    )

async def update_additional_info(candidate_id: str, new_answers: dict) -> None:
    """Merge new_answers into the candidate's additional_info dict."""
    if not new_answers:
        return
    set_fields = {f"additional_info.{k}": v for k, v in new_answers.items()}
    set_fields["updated_at"] = datetime.utcnow().isoformat()
    await candidates_col().update_one(
        {"_id": candidate_id},
        {"$set": set_fields},
    )

async def update_knowledge_base(candidate_id: str, profile_data: dict) -> None:
    """Merge profile_data into the candidate's core knowledge_base."""
    if not profile_data:
        return
    # Flattens some core fields back to the top level if needed
    updates = {}
    for k, v in profile_data.items():
        updates[f"knowledge_base.{k}"] = v
        # Also sync certain fields to top level for legacy support/indexing
        if k in ["name", "email", "skills", "experience_years", "phone"]:
            updates[k] = v

    updates["updated_at"] = datetime.utcnow().isoformat()
    await candidates_col().update_one(
        {"_id": candidate_id},
        {"$set": updates},
    )


# ─── Application log helpers ─────────────────────────────────────────────────

async def save_application(candidate_id: str, job_url: str, job_title: str,
                            answers: dict, status: str = "logged") -> str:
    """Log an application and return its inserted id."""
    doc = {
        "candidate_id": candidate_id,
        "job_url": job_url,
        "job_title": job_title,
        "answers": answers,
        "status": status,
        "applied_at": datetime.utcnow().isoformat(),
    }
    result = await applications_col().insert_one(doc)
    return str(result.inserted_id)

async def get_applications(candidate_id: str) -> list:
    """Return all application logs for a candidate."""
    cursor = applications_col().find({"candidate_id": candidate_id})
    docs = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        docs.append(doc)
    return docs
