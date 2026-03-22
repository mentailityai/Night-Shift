# =============================================================================
# Night-Shift — Pytest Configuration & Shared Fixtures
# =============================================================================
# Provides reusable fixtures for tests:
#   - Overridden settings (in-memory / test DB)
#   - Async database sessions
#   - FastAPI test client
# =============================================================================

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator
from typing import Generator
from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import Settings, get_settings
from app.db.models import Base


# ---------------------------------------------------------------------------
# Test Settings Override
# ---------------------------------------------------------------------------
def get_test_settings() -> Settings:
    """
    Return a Settings instance configured for testing.

    Uses the same Postgres database but with a test-specific name override
    if desired.  In CI, you'd override ``DATABASE_URL`` via env vars.
    """
    return Settings(
        postgres_db="nightshift_test",
        log_level="DEBUG",
        log_format="console",
        fine_tuning_threshold=5,          # Low threshold for test speed
        worker_batch_trigger_size=2,      # Low trigger for test speed
        worker_batch_process_limit=10,
    )


# ---------------------------------------------------------------------------
# Event Loop Fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create a single event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Database Fixtures
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture(scope="session")
async def test_engine():
    """Create a test database engine and set up all tables."""
    settings = get_test_settings()
    engine = create_async_engine(
        settings.effective_database_url,
        echo=False,
    )

    # Create all tables from the ORM models
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Tear down — drop all tables after the test session
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """
    Provide a transactional database session for each test.

    Each test runs inside a transaction that is rolled back after the
    test completes, ensuring test isolation without the cost of
    recreating tables.
    """
    session_factory = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with session_factory() as session:
        yield session
        await session.rollback()


# ---------------------------------------------------------------------------
# FastAPI Test Client
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """
    Provide an async HTTP test client for the FastAPI application.

    Overrides the database dependency to use the test session, ensuring
    all API calls in the test use the transactional test DB.
    """
    from app.api.dependencies import get_db
    from app.api.main import app

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()
