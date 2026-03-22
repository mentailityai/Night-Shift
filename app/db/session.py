# =============================================================================
# Night-Shift — Database Session Management
# =============================================================================
# Provides the async SQLAlchemy engine and session factory used by every
# module that needs database access.
#
# Usage:
#   from app.db.session import get_async_session
#
#   async for session in get_async_session():
#       result = await session.execute(...)
# =============================================================================

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings

# ---------------------------------------------------------------------------
# Engine — a single connection pool shared across the application.
# ---------------------------------------------------------------------------
# ``echo=False`` in production; set ``LOG_LEVEL=DEBUG`` and ``echo=True``
# during development if you want to see raw SQL.
# ---------------------------------------------------------------------------
_settings = get_settings()

engine = create_async_engine(
    _settings.effective_database_url,
    echo=False,
    pool_pre_ping=True,      # Verify connections before checkout
    pool_size=10,             # Max persistent connections
    max_overflow=20,          # Burst connections beyond pool_size
)

# ---------------------------------------------------------------------------
# Session factory — produces AsyncSession instances bound to the engine.
# ``expire_on_commit=False`` keeps loaded attributes available after commit
# without requiring a refresh (useful for returning objects from API routes).
# ---------------------------------------------------------------------------
async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency-injectable async session generator.

    Yields an ``AsyncSession`` that is automatically closed when the
    caller exits the ``async for`` / ``async with`` block.  FastAPI
    injects this via ``Depends(get_async_session)``.
    """
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()
