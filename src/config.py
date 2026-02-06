"""Configuration management using pydantic-settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OpenAI
    openai_api_key: str

    # Search
    tavily_api_key: str

    # Hugging Face
    HUGGINGFACE_API_KEY: str

    # LangSmith (optional)
    langsmith_api_key: str | None = None
    langchain_tracing_v2: bool = False
    langchain_project: str = "research-report-generator"

    # Database
    database_url: str = "postgresql://postgres:localdev@localhost:5432/research_reports"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # AWS (Phase 4)
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_default_region: str = "eu-north-1"
    aws_s3_bucket: str | None = None

    # Application
    log_level: str = "INFO"
    environment: str = "development"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
