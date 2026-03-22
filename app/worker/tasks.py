# =============================================================================
# Night-Shift — Celery Tasks
# =============================================================================
# Defines the Celery tasks that are scheduled via Beat (cron) or dispatched
# on-demand from the API (batch-size trigger).
#
# Tasks bridge the synchronous Celery world with the async application
# code by running the async orchestrators inside an event loop.
# =============================================================================

from __future__ import annotations

import asyncio

from app.core.logging import get_logger, setup_logging
from app.worker.celery_app import celery_app

logger = get_logger(__name__)


def _run_async(coro):
    """
    Run an async coroutine from synchronous Celery task code.

    Celery workers are synchronous by default.  This helper creates a
    new event loop (or reuses the existing one) to bridge into async.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If there's already a running loop (e.g., in testing),
            # schedule the coroutine on it.
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop exists — create one
        return asyncio.run(coro)


@celery_app.task(
    name="app.worker.tasks.process_pending_logs",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
)
def process_pending_logs(self) -> dict[str, int]:
    """
    Celery task: process all pending interaction logs.

    This task is triggered by:
      1. The nightly Celery Beat cron schedule.
      2. The batch-size trigger in the API ingestion endpoint.

    It delegates to the async ``process_pending_batch()`` orchestrator
    which handles state locking, LLM calls, and database updates.

    Returns
    -------
    dict[str, int]
        Summary counts of processed, discarded, and failed logs.
    """
    setup_logging()
    logger.info("task_started", task="process_pending_logs")

    try:
        from app.worker.agent import process_pending_batch
        counts = _run_async(process_pending_batch())

        logger.info(
            "task_completed",
            task="process_pending_logs",
            **counts,
        )
        return counts

    except Exception as exc:
        logger.error(
            "task_failed",
            task="process_pending_logs",
            error=str(exc),
        )
        # Retry with exponential backoff
        raise self.retry(exc=exc)


@celery_app.task(
    name="app.worker.tasks.check_finetune_threshold",
    bind=True,
    max_retries=2,
    default_retry_delay=120,
    acks_late=True,
)
def check_finetune_threshold(self) -> dict[str, object]:
    """
    Celery task: check if the fine-tuning threshold has been reached.

    This task is triggered by the nightly Celery Beat cron schedule.
    It delegates to the async ``check_and_export()`` function in the
    fine-tuning monitor module.

    Returns
    -------
    dict[str, object]
        Status of the threshold check and any export action taken.
    """
    setup_logging()
    logger.info("task_started", task="check_finetune_threshold")

    try:
        from app.finetune.monitor import check_and_export
        result = _run_async(check_and_export())

        logger.info(
            "task_completed",
            task="check_finetune_threshold",
            **result,
        )
        return result

    except Exception as exc:
        logger.error(
            "task_failed",
            task="check_finetune_threshold",
            error=str(exc),
        )
        raise self.retry(exc=exc)
