# =============================================================================
# Night-Shift — Fine-Tuning Threshold Monitor
# =============================================================================
# Checks whether the number of 'waiting' rows in Fine_Tuning_Staging has
# reached the configurable threshold (``FINE_TUNING_THRESHOLD`` in .env).
#
# If the threshold is met, it triggers the exporter to compile and write
# the JSONL training file.
#
# This module is called by the Celery ``check_finetune_threshold`` task
# on the nightly cron schedule.
# =============================================================================

from __future__ import annotations

from sqlalchemy import func, select

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models import FineTuningStagingPair
from app.db.session import async_session_factory
from app.finetune.exporter import export_training_data

logger = get_logger(__name__)


async def check_and_export() -> dict[str, object]:
    """
    Check the fine-tuning threshold and trigger export if met.

    Reads the ``FINE_TUNING_THRESHOLD`` from the environment config
    and compares it against the current count of ``"waiting"`` rows
    in the ``Fine_Tuning_Staging`` table.

    Returns
    -------
    dict[str, object]
        Status report:
        - If threshold not met: ``{"status": "below_threshold", ...}``
        - If threshold met: the export result from ``export_training_data()``
    """
    settings = get_settings()
    threshold = settings.fine_tuning_threshold

    async with async_session_factory() as session:
        # Count the number of rows waiting to be exported
        count_query = (
            select(func.count())
            .select_from(FineTuningStagingPair)
            .where(FineTuningStagingPair.batch_status == "waiting")
        )
        result = await session.execute(count_query)
        waiting_count = result.scalar_one()

    logger.info(
        "threshold_check",
        waiting_count=waiting_count,
        threshold=threshold,
    )

    if waiting_count < threshold:
        return {
            "status": "below_threshold",
            "waiting_count": waiting_count,
            "threshold": threshold,
            "message": (
                f"{waiting_count}/{threshold} rows waiting. "
                f"Need {threshold - waiting_count} more to trigger export."
            ),
        }

    # -----------------------------------------------------------------
    # Threshold met — trigger the export pipeline
    # -----------------------------------------------------------------
    logger.info(
        "threshold_met",
        waiting_count=waiting_count,
        threshold=threshold,
    )

    export_result = await export_training_data()

    return export_result
