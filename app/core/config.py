# =============================================================================
# Night-Shift — Centralized Configuration
# =============================================================================
# All application settings are loaded from environment variables (or a .env
# file) and validated at startup using Pydantic BaseSettings.  Every tunable
# parameter — LLM provider, thresholds, MCP transport, etc. — lives here so
# the operator never needs to modify source code.
#
# Usage:
#   from app.core.config import get_settings
#   settings = get_settings()
#   print(settings.llm_provider)           # "vllm"
#   print(settings.fine_tuning_threshold)   # 500
# =============================================================================

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized configuration for the Night-Shift Preference Optimizer.

    Values are read from a `.env` file (if present) and can be overridden by
    real environment variables.  Every field has a sensible default so the
    application can start with minimal configuration for local development.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # -------------------------------------------------------------------------
    # Database (PostgreSQL + pgvector)
    # -------------------------------------------------------------------------
    postgres_user: str = "nightshift"
    postgres_password: str = "nightshift"
    postgres_db: str = "nightshift"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    database_url: str | None = None  # Override the auto-composed URL if set

    @computed_field  # type: ignore[prop-decorator]
    @property
    def effective_database_url(self) -> str:
        """
        Return the database connection string.

        If ``database_url`` is explicitly provided, use it verbatim.
        Otherwise, compose it from the individual Postgres fields.
        Uses the ``asyncpg`` driver for async SQLAlchemy support.
        """
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sync_database_url(self) -> str:
        """
        Synchronous database URL used by Alembic migrations.

        Alembic runs synchronously, so we need a ``psycopg2`` (or plain
        ``psycopg``) driver URL rather than the async ``asyncpg`` one.
        """
        if self.database_url:
            return self.database_url.replace(
                "postgresql+asyncpg", "postgresql+psycopg2"
            )
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # -------------------------------------------------------------------------
    # Redis (Celery Broker)
    # -------------------------------------------------------------------------
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_url: str | None = None  # Override the auto-composed URL if set

    @computed_field  # type: ignore[prop-decorator]
    @property
    def effective_redis_url(self) -> str:
        """Return the Redis connection string."""
        if self.redis_url:
            return self.redis_url
        return f"redis://{self.redis_host}:{self.redis_port}/0"

    # -------------------------------------------------------------------------
    # LLM Provider
    # -------------------------------------------------------------------------
    llm_provider: Literal["vllm", "openai", "anthropic"] = "vllm"

    # --- vLLM (local, default) ---
    vllm_base_url: str = "http://192.168.1.180:8000/v1"
    vllm_model_name: str = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
    vllm_api_key: str = "token-placeholder"

    # --- OpenAI ---
    openai_api_key: str = ""
    openai_model_name: str = "gpt-4o"

    # --- Anthropic ---
    anthropic_api_key: str = ""
    anthropic_model_name: str = "claude-sonnet-4-20250514"

    # -------------------------------------------------------------------------
    # Embedding Model
    # -------------------------------------------------------------------------
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_dimension: int = 1024

    # -------------------------------------------------------------------------
    # Night-Shift Worker (Celery)
    # -------------------------------------------------------------------------
    # Cron schedule for periodic nightly processing (hour and minute).
    worker_cron_minute: int = 0
    worker_cron_hour: int = 2

    # Batch-size trigger: fire processing immediately when pending count
    # reaches this value (useful for high-volume enterprise deployments).
    worker_batch_trigger_size: int = 50

    # Maximum logs to process in a single worker invocation.
    worker_batch_process_limit: int = 100

    # -------------------------------------------------------------------------
    # Fine-Tuning (Slow Loop)
    # -------------------------------------------------------------------------
    fine_tuning_threshold: int = 500
    fine_tuning_export_dir: str = "./exports"
    fine_tuning_provider: Literal["local", "openai", "vllm"] = "local"
    fine_tuning_replay_buffer_ratio: float = Field(
        default=0.15, ge=0.0, le=1.0
    )

    # --- OpenAI Fine-Tuning ---
    openai_finetune_model: str = "gpt-4o-mini-2024-07-18"

    # --- vLLM Fine-Tuning ---
    vllm_finetune_base_url: str = "http://localhost:8000"

    # -------------------------------------------------------------------------
    # MCP Server (Fast Loop)
    # -------------------------------------------------------------------------
    mcp_transport: Literal["stdio", "sse"] = "stdio"
    mcp_sse_host: str = "0.0.0.0"
    mcp_sse_port: int = 8080
    mcp_search_top_k: int = 5
    mcp_search_min_score: float = Field(default=0.5, ge=0.0, le=1.0)

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"


@lru_cache()
def get_settings() -> Settings:
    """
    Return a cached singleton of the application settings.

    Using ``lru_cache`` ensures the ``.env`` file is only read once and the
    same ``Settings`` instance is reused throughout the application lifetime.
    """
    return Settings()
