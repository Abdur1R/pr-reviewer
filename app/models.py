from datetime import datetime
from typing import Literal, Optional, list

from pydantic import BaseModel, Field


class RepoSelection(BaseModel):
    repo_id: Optional[int] = Field(None, alias="repoId")
    full_name: str

    model_config = {"populate_by_name": True}


class ReviewerConfig(BaseModel):
    provider: Literal[
        "openai",
        "claude",
        "huggingface",
        "ollama",
        "groq",
        "openrouter",
        "together",
        "sambanova",
    ]
    model: str

    model_config = {"populate_by_name": True}


class InstallationSettingsRequest(BaseModel):
    user_id: str = Field(..., alias="userId")
    selected_repos: list[RepoSelection] = Field(..., alias="selectedRepos")
    reviewer: ReviewerConfig

    model_config = {"populate_by_name": True}


class InstallationUpdateRequest(BaseModel):
    selected_repos: Optional[list[RepoSelection]] = Field(None, alias="selectedRepos")
    reviewer: Optional[ReviewerConfig] = None

    model_config = {"populate_by_name": True}


class InstallationResponse(BaseModel):
    installation_id: int = Field(..., alias="installationId")
    user_id: str = Field(..., alias="userId")
    selected_repos: list[RepoSelection] = Field(..., alias="selectedRepos")
    reviewer: ReviewerConfig
    created_at: datetime = Field(..., alias="createdAt")
    updated_at: datetime = Field(..., alias="updatedAt")

    model_config = {"populate_by_name": True, "by_alias": True}

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
