"""
aria_router.py
All Aria personal-assistant endpoints, mounted onto the existing PR Guardian app.
Nothing in this file touches any PR reviewer code.
"""

import io
import logging
from typing import Optional, List

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from . import db
from .llm_router import call_llm, get_provider_status
from .tools import classify_query, get_realtime_context
from .models import (
    ChatRequest, ChatResponse,
    SessionStartRequest, SessionStartResponse,
    SessionEndRequest, SessionEndResponse,
    DocumentUploadResponse,
)

logger = logging.getLogger(__name__)

aria_router = APIRouter(prefix="/aria", tags=["aria"])


# ─── Lifecycle (called from main.py's startup/shutdown events) ────────────────

async def aria_startup():
    await db.connect()
    logger.info("Aria: MongoDB connected.")


async def aria_shutdown():
    await db.disconnect()
    logger.info("Aria: MongoDB disconnected.")


# ─── System prompt ────────────────────────────────────────────────────────────

def build_system_prompt(
    user_name: str,
    profile: Optional[str],
    resume_context: Optional[str],
    doc_context: Optional[str],
    realtime_context: Optional[str],
    mode: str,
) -> str:
    parts = [
        f"You are Aria, {user_name or 'the user'}'s personal AI assistant — sharp, direct, and deeply attuned to who they are. "
        "You are not a generic assistant. You know this specific person and serve only them.",
    ]
    if profile:
        parts.append(f"What you know about {user_name or 'the user'}:\n{profile}")
    if resume_context:
        parts.append(f"Context from the previous session (pick up naturally from here):\n{resume_context}")
    if doc_context:
        parts.append(f"Relevant material from the user's own documents:\n{doc_context}")
    if realtime_context:
        parts.append(
            f"Real-time data fetched for this query — treat this as ground truth, not an estimate:\n{realtime_context}"
        )
    if mode == "voice":
        parts.append(
            "You are speaking aloud via text-to-speech. "
            "Answer in 1–3 short spoken sentences. Plain language only — no markdown, no lists, no headers, no asterisks. "
            "Say it exactly as you would speak it out loud. "
            "If the answer genuinely needs more depth, give the short version and say 'want me to go deeper?'."
        )
    else:
        parts.append("Be concise and substantive. Use markdown only when it genuinely helps readability.")
    return "\n\n".join(parts)


# ─── Session endpoints ────────────────────────────────────────────────────────

@aria_router.post("/session/start", response_model=SessionStartResponse)
async def start_session(req: SessionStartRequest):
    session_id = await db.create_session(req.user_id)
    last = await db.get_last_session(req.user_id)
    return SessionStartResponse(
        session_id=session_id,
        resume_context=last["resume_prompt"] if last else None,
        last_session_summary=last["resume_prompt"] if last else None,
    )


@aria_router.post("/session/end", response_model=SessionEndResponse)
async def end_session(req: SessionEndRequest):
    msgs = await db.get_session_messages(req.session_id)
    if not msgs:
        raise HTTPException(400, "No messages in this session to summarize.")

    conversation = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in msgs
    )
    summarize_messages = [
        {
            "role": "system",
            "content": (
                "You create compact session-resume prompts. "
                "Write 5–10 bullet points covering: what was discussed, decisions made, "
                "tasks mentioned, any unresolved questions, and the user's apparent mood/energy. "
                "Write it as a briefing for yourself at the start of the next session — "
                "second person ('The user said...', 'You discussed...'). "
                "No fluff. Dense and factual."
            ),
        },
        {
            "role": "user",
            "content": f"Summarize this session into a resume prompt:\n\n{conversation}",
        },
    ]

    summary, llm_used = await call_llm(summarize_messages, max_tokens=600)
    await db.end_session(req.session_id, summary, llm_used)

    return SessionEndResponse(summary=summary, message_count=len(msgs))


# ─── Chat endpoint ────────────────────────────────────────────────────────────

@aria_router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    user_id = "default"  # Extend with auth later

    await db.save_message(req.session_id, "user", req.message)

    profile = await db.get_profile(user_id)
    last_session = await db.get_last_session(user_id)
    resume_context = last_session["resume_prompt"] if last_session else None

    doc_chunks = await db.retrieve_context(user_id, req.message)
    doc_context = (
        "\n\n---\n\n".join(f"[{c['source']}]\n{c['text']}" for c in doc_chunks)
        if doc_chunks else None
    )

    category = classify_query(req.message)
    realtime_context = None
    if category:
        realtime_context = await get_realtime_context(req.message, category)

    system_prompt = build_system_prompt(
        user_name=user_id,
        profile=profile,
        resume_context=resume_context,
        doc_context=doc_context,
        realtime_context=realtime_context,
        mode=req.mode,
    )

    history = await db.get_session_messages(req.session_id)
    llm_messages = [{"role": "system", "content": system_prompt}]
    for m in history[-20:]:
        llm_messages.append({"role": m["role"], "content": m["content"]})

    max_tokens = 300 if req.mode == "voice" else 1200
    reply, llm_used = await call_llm(llm_messages, max_tokens=max_tokens)

    await db.save_message(req.session_id, "assistant", reply, llm_used)
    await db.update_session_llm(req.session_id, llm_used)

    user_msg_count = sum(1 for m in history if m["role"] == "user")
    if user_msg_count > 0 and user_msg_count % 6 == 0:
        await _refresh_profile(user_id, history, profile)

    return ChatResponse(reply=reply, llm_used=llm_used, session_id=req.session_id)


async def _refresh_profile(user_id: str, history: list, current_profile: Optional[str]):
    recent = "\n".join(f"{m['role']}: {m['content']}" for m in history[-24:])
    profile_messages = [
        {
            "role": "system",
            "content": (
                "Maintain a concise factual profile of this user for use in future AI sessions. "
                "Return 5–8 bullet points: goals, working style, recurring topics, preferences, anything practically useful. "
                "No speculation, no flattery. Plain bullets only."
            ),
        },
        {
            "role": "user",
            "content": f"Existing profile:\n{current_profile or '(none)'}\n\nRecent messages:\n{recent}\n\nReturn updated profile bullets only.",
        },
    ]
    try:
        new_profile, _ = await call_llm(profile_messages, max_tokens=400)
        await db.save_profile(user_id, new_profile)
    except Exception:
        pass


# ─── Document endpoints ───────────────────────────────────────────────────────

def chunk_text(text: str, size: int = 900) -> List[str]:
    return [text[i:i+size] for i in range(0, len(text), size)]


@aria_router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form("default"),
):
    content = await file.read()
    text = ""

    if file.filename.endswith(".pdf"):
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            raise HTTPException(400, f"PDF parse error: {e}")
    elif file.filename.endswith(".docx"):
        try:
            import docx
            doc = docx.Document(io.BytesIO(content))
            text = "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            raise HTTPException(400, f"DOCX parse error: {e}")
    else:
        text = content.decode("utf-8", errors="ignore")

    if not text.strip():
        raise HTTPException(400, "No readable text found in the file.")

    chunks = chunk_text(text.strip())
    doc_id = await db.save_document(user_id, file.filename, chunks)

    return DocumentUploadResponse(
        doc_id=doc_id,
        name=file.filename,
        chunk_count=len(chunks),
    )


@aria_router.get("/documents")
async def list_documents(user_id: str = "default"):
    docs = await db.get_documents(user_id)
    return [{"id": str(d["_id"]), "name": d["name"], "created_at": d["created_at"]} for d in docs]


@aria_router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    await db.delete_document(doc_id)
    return {"deleted": doc_id}


# ─── Status ───────────────────────────────────────────────────────────────────

@aria_router.get("/status")
async def status():
    return {
        "status": "ok",
        "providers": get_provider_status(),
    }