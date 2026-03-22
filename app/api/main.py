# =============================================================================
# Night-Shift — FastAPI Application Entry Point
# =============================================================================
# Initialises the FastAPI application, configures middleware, registers
# route routers, and sets up the application lifespan (startup/shutdown).
#
# Run locally with:
#   uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
# =============================================================================

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI

from app.api.routes.ingestion import router as ingestion_router
from app.core.logging import setup_logging, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Application Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage startup and shutdown events for the FastAPI application.

    Startup:
      - Initialise structured logging.
      - (Future) Warm up the embedding model if desired.

    Shutdown:
      - (Future) Close database connection pools gracefully.
    """
    # ---- Startup ----
    setup_logging()
    logger.info("nightshift_api_starting")
    yield
    # ---- Shutdown ----
    logger.info("nightshift_api_shutting_down")


# ---------------------------------------------------------------------------
# FastAPI Application Instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Night-Shift Preference Optimizer",
    description=(
        "Automated data pipeline that captures user corrections to AI "
        "outputs and transforms them into usable training data."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Register Routers
# ---------------------------------------------------------------------------
app.include_router(ingestion_router)


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------
@app.get(
    "/health",
    tags=["system"],
    summary="Health check",
    description="Returns a simple status to verify the API is running.",
)
async def health_check() -> dict[str, str]:
    """Return a simple health status."""
    return {"status": "ok", "service": "nightshift-api"}
