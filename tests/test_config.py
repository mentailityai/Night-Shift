# =============================================================================
# Night-Shift — Configuration Tests
# =============================================================================
# Verifies that the Pydantic settings load correctly, compute derived
# fields, and enforce valid value ranges.
# =============================================================================

from __future__ import annotations

import pytest

from app.core.config import Settings


class TestSettings:
    """Tests for the centralized Settings configuration."""

    def test_default_values(self):
        """Settings should load with sensible defaults."""
        s = Settings()
        assert s.llm_provider == "vllm"
        assert s.postgres_db == "nightshift"
        assert s.fine_tuning_threshold == 500
        assert s.mcp_transport == "stdio"
        assert s.embedding_model_name == "BAAI/bge-m3"

    def test_effective_database_url_composed(self):
        """Database URL should be auto-composed from individual fields."""
        s = Settings(
            postgres_user="user",
            postgres_password="pass",
            postgres_host="db.example.com",
            postgres_port=5433,
            postgres_db="mydb",
        )
        expected = "postgresql+asyncpg://user:pass@db.example.com:5433/mydb"
        assert s.effective_database_url == expected

    def test_effective_database_url_override(self):
        """Explicit database_url should take precedence."""
        explicit_url = "postgresql+asyncpg://custom@host/db"
        s = Settings(database_url=explicit_url)
        assert s.effective_database_url == explicit_url

    def test_sync_database_url(self):
        """Sync URL should use psycopg2 driver for Alembic."""
        s = Settings()
        assert "psycopg2" in s.sync_database_url
        assert "asyncpg" not in s.sync_database_url

    def test_effective_redis_url_composed(self):
        """Redis URL should be auto-composed from host and port."""
        s = Settings(redis_host="redis.local", redis_port=6380)
        assert s.effective_redis_url == "redis://redis.local:6380/0"

    def test_replay_buffer_ratio_valid_range(self):
        """Replay buffer ratio must be between 0.0 and 1.0."""
        s = Settings(fine_tuning_replay_buffer_ratio=0.25)
        assert s.fine_tuning_replay_buffer_ratio == 0.25

    def test_replay_buffer_ratio_invalid(self):
        """Replay buffer ratio outside 0-1 should raise a validation error."""
        with pytest.raises(Exception):
            Settings(fine_tuning_replay_buffer_ratio=1.5)

    def test_llm_provider_validation(self):
        """LLM provider must be one of the allowed values."""
        with pytest.raises(Exception):
            Settings(llm_provider="unsupported_provider")

    def test_mcp_transport_validation(self):
        """MCP transport must be 'stdio' or 'sse'."""
        with pytest.raises(Exception):
            Settings(mcp_transport="grpc")

    def test_fine_tuning_threshold_override(self):
        """The threshold should be configurable."""
        s = Settings(fine_tuning_threshold=1000)
        assert s.fine_tuning_threshold == 1000
