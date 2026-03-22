# =============================================================================
# Night-Shift — Agent Orchestrator
# =============================================================================
# The core orchestration logic for the Night-Shift background worker.
# This module ties together the LLM client, processor, and database
# operations into a single pipeline:
#
#   1. Fetch 'pending' logs (with state locking → 'processing')
#   2. Send each log to the LLM for analysis
#   3. Parse and persist results (rules + training pairs)
#   4. Handle errors gracefully (set status to 'failed' + error_message)
#
# This module is called by the Celery tasks but is designed to be
# testable independently (no Celery dependency in this file).
# =============================================================================

from __future__ import annotations

import traceback

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models import RawInteractionLog
from app.db.session import async_session_factory
from app.worker.llm_client import analyse_interaction
from app.worker.processor import parse_llm_response, persist_processed_result

logger = get_logger(__name__)


async def process_pending_batch() -> dict[str, int]:
    """
    Fetch and process a batch of pending interaction logs.

    This is the main entry point called by the Celery task.  It performs
    the full pipeline for each pending log:

        pending → processing → (LLM call) → processed | failed | discarded

    State Locking
    -------------
    Before processing, logs are atomically moved from ``"pending"`` to
    ``"processing"`` to prevent duplicate work if multiple workers or
    triggers fire concurrently.

    Returns
    -------
    dict[str, int]
        Summary counts: {"processed": N, "discarded": N, "failed": N}
    """
    settings = get_settings()
    limit = settings.worker_batch_process_limit

    # Counters for the run summary
    counts = {"processed": 0, "discarded": 0, "failed": 0}

    async with async_session_factory() as session:
        # -----------------------------------------------------------------
        # Step 1: Claim a batch of pending logs by atomically updating
        # their status to 'processing'.  This uses a SELECT ... FOR UPDATE
        # pattern to prevent race conditions between concurrent workers.
        # -----------------------------------------------------------------
        pending_query = (
            select(RawInteractionLog.log_id)
            .where(RawInteractionLog.status == "pending")
            .order_by(RawInteractionLog.timestamp.asc())
            .limit(limit)
            .with_for_update(skip_locked=True)
        )

        result = await session.execute(pending_query)
        pending_ids = [row[0] for row in result.all()]

        if not pending_ids:
            logger.info("no_pending_logs")
            return counts

        # Atomically mark claimed logs as 'processing'
        await session.execute(
            update(RawInteractionLog)
            .where(RawInteractionLog.log_id.in_(pending_ids))
            .values(status="processing")
        )
        await session.commit()

        logger.info(
            "batch_claimed",
            count=len(pending_ids),
            limit=limit,
        )

        # -----------------------------------------------------------------
        # Step 2: Process each log individually.  We process one at a time
        # so that a failure in one log does not affect others.
        # -----------------------------------------------------------------
        for log_id in pending_ids:
            await _process_single_log(session, log_id, counts)

    return counts


async def _process_single_log(
    session: AsyncSession,
    log_id,
    counts: dict[str, int],
) -> None:
    """
    Process a single interaction log through the LLM pipeline.

    Parameters
    ----------
    session : AsyncSession
        Active database session.
    log_id : UUID
        The ID of the log to process.
    counts : dict[str, int]
        Mutable summary counters (updated in-place).
    """
    try:
        # Reload the log within this session
        log = await session.get(RawInteractionLog, log_id)
        if log is None:
            logger.warning("log_not_found", log_id=str(log_id))
            return

        logger.info("processing_log", log_id=str(log_id))

        # -----------------------------------------------------------------
        # Step 2a: Send the interaction to the configured LLM for analysis
        # -----------------------------------------------------------------
        raw_response = await analyse_interaction(
            system_prompt=log.system_prompt,
            user_input=log.user_input,
            ai_output=log.ai_output,
            human_correction=log.human_correction,
        )

        # -----------------------------------------------------------------
        # Step 2b: Parse the LLM's JSON response
        # -----------------------------------------------------------------
        parsed = parse_llm_response(raw_response)

        # -----------------------------------------------------------------
        # Step 2c: Persist results (rule + training pair) and update status
        # -----------------------------------------------------------------
        await persist_processed_result(session, log, parsed)
        await session.commit()

        # Update counters
        counts[parsed["status"]] = counts.get(parsed["status"], 0) + 1

    except Exception as exc:
        # -----------------------------------------------------------------
        # Error handling: mark the log as 'failed' and record the error
        # message so the operator can review and retry later.
        # -----------------------------------------------------------------
        await session.rollback()

        error_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        logger.error(
            "log_processing_failed",
            log_id=str(log_id),
            error=str(exc),
        )

        try:
            # Re-fetch the log in a clean state to update its status
            log = await session.get(RawInteractionLog, log_id)
            if log is not None:
                log.status = "failed"
                log.error_message = error_msg[:5000]  # Cap error length
                await session.commit()
        except Exception as inner_exc:
            logger.error(
                "failed_to_update_error_status",
                log_id=str(log_id),
                error=str(inner_exc),
            )

        counts["failed"] += 1
