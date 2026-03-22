# =============================================================================
# Night-Shift — API Dependencies
# =============================================================================
# Shared FastAPI dependencies injected into route handlers via ``Depends()``.
# Centralising dependencies here keeps route files focused on business logic.
# =============================================================================

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import async_session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Yield an async database session for the duration of a single request.

    The session is automatically committed on success and rolled back on
    exception, then closed when the request completes.

    Usage in route handlers:
        @router.post("/api/logs")
        async def ingest(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
