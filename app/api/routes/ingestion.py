# =============================================================================
# Night-Shift — Ingestion Route
# =============================================================================
# POST /api/logs
#
# This is the Capture Layer entry point.  The frontend (or any client)
# sends the raw interaction payload here after a user finishes editing
# an AI-generated text block.  The payload is validated, stored in the
# ``Raw_Interaction_Logs`` table with status='pending', and immediately
# returned with a confirmation.
#
# After ingestion, if the number of pending logs reaches the configured
# ``WORKER_BATCH_TRIGGER_SIZE``, a Celery task is dispatched to process
# the batch immediately (in addition to the nightly cron schedule).
# =============================================================================

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db
from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models import RawInteractionLog

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
router = APIRouter(prefix="/api", tags=["ingestion"])


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------
class InteractionPayload(BaseModel):
    """
    Schema for the raw interaction payload sent by the frontend.

    All four text fields are required — they capture the full context
    of the AI generation + human correction cycle.
    """

    system_prompt: str = Field(
        ...,
        min_length=1,
        description="The system instructions the AI was operating under.",
    )
    user_input: str = Field(
        ...,
        min_length=1,
        description="The user request or uploaded source text.",
    )
    ai_output: str = Field(
        ...,
        min_length=1,
        description="The exact AI-generated response (before edits).",
    )
    human_correction: str = Field(
        ...,
        min_length=1,
        description="The final text after the user's manual edits.",
    )


class InteractionResponse(BaseModel):
    """Response returned after a successful ingestion."""

    log_id: uuid.UUID
    status: str
    timestamp: datetime
    message: str


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@router.post(
    "/logs",
    response_model=InteractionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a raw interaction log",
    description=(
        "Receives the raw interaction payload from the frontend and stores "
        "it in the Raw_Interaction_Logs table with status='pending'.  "
        "Optionally triggers a batch processing run if the pending count "
        "exceeds the configured threshold."
    ),
)
async def ingest_interaction(
    payload: InteractionPayload,
    db: AsyncSession = Depends(get_db),
) -> InteractionResponse:
    """
    Capture a single user-correction interaction.

    Steps:
    1. Validate the incoming payload (handled by Pydantic).
    2. Create a new ``RawInteractionLog`` row with status='pending'.
    3. Check if the pending log count has reached the batch trigger size.
    4. If threshold met, dispatch a Celery task for immediate processing.
    5. Return the created log ID and status.
    """

    # --- 1. Build the database record ---
    log_entry = RawInteractionLog(
        system_prompt=payload.system_prompt,
        user_input=payload.user_input,
        ai_output=payload.ai_output,
        human_correction=payload.human_correction,
        status="pending",
    )

    db.add(log_entry)
    await db.flush()  # Populate the auto-generated fields (log_id, timestamp)

    logger.info(
        "interaction_ingested",
        log_id=str(log_entry.log_id),
        status=log_entry.status,
    )

    # --- 2. Check batch trigger threshold ---
    try:
        settings = get_settings()
        count_result = await db.execute(
            select(func.count())
            .select_from(RawInteractionLog)
            .where(RawInteractionLog.status == "pending")
        )
        pending_count = count_result.scalar_one()

        if pending_count >= settings.worker_batch_trigger_size:
            # Import here to avoid circular imports with Celery
            from app.worker.tasks import process_pending_logs

            process_pending_logs.delay()
            logger.info(
                "batch_trigger_fired",
                pending_count=pending_count,
                threshold=settings.worker_batch_trigger_size,
            )
    except Exception as exc:
        # Log but don't fail the ingestion if the batch trigger errors
        logger.warning(
            "batch_trigger_check_failed",
            error=str(exc),
        )

    # --- 3. Return confirmation ---
    return InteractionResponse(
        log_id=log_entry.log_id,
        status=log_entry.status,
        timestamp=log_entry.timestamp,
        message="Interaction log captured successfully.",
    )
