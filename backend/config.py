import os
from typing import Optional

from dotenv import load_dotenv

try:
    from pydantic_settings import BaseSettings
except Exception:
    from pydantic import BaseSettings

from pydantic import Field

load_dotenv()


class Settings(BaseSettings):
    AZURE_OPENAI_API_KEY: Optional[str] = Field(None, env="AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT: Optional[str] = Field(None, env="AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION: str = Field(
        "2024-12-01-preview", env="AZURE_OPENAI_API_VERSION"
    )
    AZURE_OPENAI_DEPLOYMENT: Optional[str] = Field(None, env="AZURE_OPENAI_DEPLOYMENT")

    MAX_PROMPT_TOKENS: int = Field(2000, env="MAX_PROMPT_TOKENS")
    MAX_TOKENS_LIMIT: int = Field(128000, env="MAX_TOKENS_LIMIT")
    PROMPT_MAX_CHARS: int = Field(200000, env="PROMPT_MAX_CHARS")

    REDIS_URL: Optional[str] = Field(None, env="REDIS_URL")
    CACHE_TTL_SECONDS: int = Field(300, env="CACHE_TTL_SECONDS")
    QUOTA_COOLDOWN_SECONDS: int = Field(3600, env="QUOTA_COOLDOWN_SECONDS")

    LLM_MAX_ATTEMPTS: int = Field(3, env="LLM_MAX_ATTEMPTS")
    LLM_PER_ATTEMPT_TIMEOUT: int = Field(30, env="LLM_PER_ATTEMPT_TIMEOUT")
    LLM_BASE_BACKOFF: float = Field(0.5, env="LLM_BASE_BACKOFF")

    LLM_APM_SECRET_KEY: str = Field("dev-secret-change-me", env="LLM_APM_SECRET_KEY")
    METRICS_JWT_SECRET: Optional[str] = Field(None, env="METRICS_JWT_SECRET")
    METRICS_JWT_AUD: str = Field("llm-apm", env="METRICS_JWT_AUD")

    DATABASE_URL: Optional[str] = Field(None, env="DATABASE_URL")

    PORT: int = Field(8000, env="PORT")
    RELOAD: bool = Field(True, env="RELOAD")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"


settings = Settings()


def normalize_env_value(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    return (
        v.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .strip()
        .strip('"')
        .strip("'")
    )


AZURE_OPENAI_API_KEY = normalize_env_value(settings.AZURE_OPENAI_API_KEY)
AZURE_OPENAI_ENDPOINT = normalize_env_value(settings.AZURE_OPENAI_ENDPOINT)
AZURE_OPENAI_API_VERSION = normalize_env_value(settings.AZURE_OPENAI_API_VERSION)
AZURE_OPENAI_DEPLOYMENT = (
    normalize_env_value(settings.AZURE_OPENAI_DEPLOYMENT) or "gpt-4o-mini"
)
if isinstance(AZURE_OPENAI_DEPLOYMENT, str):
    AZURE_OPENAI_DEPLOYMENT = AZURE_OPENAI_DEPLOYMENT.strip()
