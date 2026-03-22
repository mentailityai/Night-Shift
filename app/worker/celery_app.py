# =============================================================================
# Night-Shift — Celery Application
# =============================================================================
# Configures the Celery distributed task queue with Redis as the message
# broker.  Celery Beat is used for the nightly cron schedule, and an
# additional batch-size trigger fires from the API ingestion endpoint.
#
# Start the worker:
#   celery -A app.worker.celery_app worker --loglevel=info
#
# Start the beat scheduler (in a separate terminal):
#   celery -A app.worker.celery_app beat --loglevel=info
# =============================================================================

from __future__ import annotations

from celery import Celery
from celery.schedules import crontab

from app.core.config import get_settings

settings = get_settings()

# ---------------------------------------------------------------------------
# Celery Instance
# ---------------------------------------------------------------------------
# The broker URL points to Redis, which handles message queuing between the
# API (producer) and the worker processes (consumers).
# ---------------------------------------------------------------------------
celery_app = Celery(
    "nightshift",
    broker=settings.effective_redis_url,
    backend=settings.effective_redis_url,
    include=[
        "app.worker.tasks",  # Auto-discover task modules
    ],
)

# ---------------------------------------------------------------------------
# Celery Configuration
# ---------------------------------------------------------------------------
celery_app.conf.update(
    # Serialisation — use JSON for transparency and debuggability
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Timezone — all timestamps in UTC
    timezone="UTC",
    enable_utc=True,

    # Prefetch multiplier — process one task at a time per worker to avoid
    # hogging LLM resources (each task may call an LLM API).
    worker_prefetch_multiplier=1,

    # Task acknowledgement — acknowledge AFTER the task completes so that
    # if a worker crashes mid-task, the message is re-delivered.
    task_acks_late=True,

    # Reject tasks on worker shutdown and re-queue them.
    task_reject_on_worker_lost=True,
)

# ---------------------------------------------------------------------------
# Celery Beat Schedule (Cron Trigger)
# ---------------------------------------------------------------------------
# The nightly processing run is configurable via WORKER_CRON_HOUR and
# WORKER_CRON_MINUTE environment variables (defaults to 02:00 UTC).
# ---------------------------------------------------------------------------
celery_app.conf.beat_schedule = {
    "nightly-process-pending-logs": {
        "task": "app.worker.tasks.process_pending_logs",
        "schedule": crontab(
            minute=settings.worker_cron_minute,
            hour=settings.worker_cron_hour,
        ),
        "options": {"queue": "default"},
    },
    "nightly-check-finetune-threshold": {
        "task": "app.worker.tasks.check_finetune_threshold",
        "schedule": crontab(
            minute=settings.worker_cron_minute,
            hour=settings.worker_cron_hour,
        ),
        "options": {"queue": "default"},
    },
}
