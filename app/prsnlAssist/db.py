import os
from datetime import datetime
from typing import Optional, Dict
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

_client: Optional[AsyncIOMotorClient] = None
_db = None


def get_db():
    return _db


async def connect():
    global _client, _db
    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB", "aria")
    _client = AsyncIOMotorClient(uri)
    _db = _client[db_name]
    # Indexes for fast lookups
    await _db.messages.create_index([("session_id", 1), ("ts", 1)])
    await _db.sessions.create_index([("user_id", 1), ("created_at", -1)])
    await _db.documents.create_index([("user_id", 1)])
    print(f"MongoDB connected → {db_name}")


async def disconnect():
    if _client:
        _client.close()


# ─── Sessions ───────────────────────────────────────────────────────────────

async def create_session(user_id: str) -> str:
    result = await _db.sessions.insert_one({
        "user_id": user_id,
        "created_at": datetime.utcnow(),
        "ended_at": None,
        "resume_prompt": None,
        "active_llm": None,
        "message_count": 0,
    })
    return str(result.inserted_id)


async def get_last_session(user_id: str) -> Optional[Dict]:
    """Get the most recent completed session for this user."""
    doc = await _db.sessions.find_one(
        {"user_id": user_id, "ended_at": {"$ne": None}, "resume_prompt": {"$ne": None}},
        sort=[("ended_at", -1)]
    )
    return doc


async def end_session(session_id: str, summary: str, llm_used: str):
    await _db.sessions.update_one(
        {"_id": ObjectId(session_id)},
        {"$set": {
            "ended_at": datetime.utcnow(),
            "resume_prompt": summary,
            "active_llm": llm_used,
        }}
    )


async def update_session_llm(session_id: str, llm_name: str):
    await _db.sessions.update_one(
        {"_id": ObjectId(session_id)},
        {"$set": {"active_llm": llm_name}}
    )


# ─── Messages ────────────────────────────────────────────────────────────────

async def save_message(session_id: str, role: str, content: str, llm_used: Optional[str] = None):
    await _db.messages.insert_one({
        "session_id": session_id,
        "role": role,
        "content": content,
        "ts": datetime.utcnow(),
        "llm_used": llm_used,
    })
    await _db.sessions.update_one(
        {"_id": ObjectId(session_id)},
        {"$inc": {"message_count": 1}}
    )


async def get_session_messages(session_id: str) -> list[Dict]:
    cursor = _db.messages.find(
        {"session_id": session_id},
        sort=[("ts", 1)]
    )
    return await cursor.to_list(length=200)


# ─── Profile ─────────────────────────────────────────────────────────────────

async def get_profile(user_id: str) -> Optional[str]:
    doc = await _db.profile.find_one({"user_id": user_id})
    return doc["summary"] if doc else None


async def save_profile(user_id: str, summary: str):
    await _db.profile.update_one(
        {"user_id": user_id},
        {"$set": {"summary": summary, "updated_at": datetime.utcnow()}},
        upsert=True
    )


# ─── Documents (RAG chunks) ───────────────────────────────────────────────────

async def save_document(user_id: str, name: str, chunks: list[str]) -> str:
    result = await _db.documents.insert_one({
        "user_id": user_id,
        "name": name,
        "chunks": chunks,
        "created_at": datetime.utcnow(),
    })
    return str(result.inserted_id)


async def get_documents(user_id: str) -> list[Dict]:
    cursor = _db.documents.find({"user_id": user_id}, {"chunks": 0})
    return await cursor.to_list(length=100)


async def delete_document(doc_id: str):
    await _db.documents.delete_one({"_id": ObjectId(doc_id)})


async def retrieve_context(user_id: str, query: str, max_chunks: int = 5) -> list[Dict]:
    """Keyword-based retrieval across all user documents."""
    terms = [t.lower() for t in query.split() if len(t) > 2]
    if not terms:
        return []

    all_docs = await _db.documents.find({"user_id": user_id}).to_list(length=200)
    candidates = []
    for doc in all_docs:
        for chunk in doc.get("chunks", []):
            lower = chunk.lower()
            score = sum(lower.count(t) for t in terms)
            if score > 0:
                candidates.append({"source": doc["name"], "text": chunk, "score": score})

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:max_chunks]
