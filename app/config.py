from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    github_app_id: str = Field(..., alias="GITHUB_APP_ID")
    github_private_key_path: str = Field(..., alias="GITHUB_PRIVATE_KEY_PATH")
    github_webhook_secret: str = Field(..., alias="GITHUB_WEBHOOK_SECRET")
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    log_level: str = Field("info", alias="LOG_LEVEL")
    hf_token: str = Field(..., alias="HF_TOKEN")
    hf_api_url: str = Field(..., alias="HF_API_URL")
    mongodb_uri: str = Field("mongodb://localhost:27017", alias="MONGODB_URI")
    mongodb_db: str = Field("pr_reviewer", alias="MONGODB_DB")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore", 
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
