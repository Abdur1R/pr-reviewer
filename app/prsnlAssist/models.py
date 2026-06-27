from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class ChatRequest(BaseModel):
    session_id: str
    message: str
    mode: str = "voice"  # "voice" or "text"


class ChatResponse(BaseModel):
    reply: str
    llm_used: str
    session_id: str


class SessionStartRequest(BaseModel):
    user_id: str = "default"


class SessionStartResponse(BaseModel):
    session_id: str
    resume_context: Optional[str] = None
    last_session_summary: Optional[str] = None


class SessionEndRequest(BaseModel):
    session_id: str


class SessionEndResponse(BaseModel):
    summary: str
    message_count: int


class DocumentUploadResponse(BaseModel):
    doc_id: str
    name: str
    chunk_count: int


class Message(BaseModel):
    role: str
    content: str
    ts: datetime
    llm_used: Optional[str] = None
