from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class RepoSelection(BaseModel):
    repo_id: Optional[int] = Field(None, alias="repoId")
    full_name: str = Field(..., alias="full_name")

    model_config = {"populate_by_name": True}


class ReviewerConfig(BaseModel):
    provider: Literal["openai", "claude", "huggingface", "ollama","groq"]
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