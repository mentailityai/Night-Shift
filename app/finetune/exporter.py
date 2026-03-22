# =============================================================================
# Night-Shift — Fine-Tuning Exporter
# =============================================================================
# Handles the data export pipeline when the fine-tuning threshold is hit:
#
#   1. Fetches all 'waiting' rows from Fine_Tuning_Staging.
#   2. Mixes in a configurable percentage of older high-quality data
#      (the "replay buffer") to prevent catastrophic forgetting.
#   3. Formats the mixed dataset into a JSONL file.
#   4. Writes the file to the configured export directory.
#   5. Updates batch_status for all included rows.
#
# Future: When ``FINE_TUNING_PROVIDER`` is set to "openai" or "vllm",
# this module will also trigger the respective fine-tuning API.
# =============================================================================

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models import FineTuningStagingPair
from app.db.session import async_session_factory

logger = get_logger(__name__)


async def export_training_data() -> dict[str, object]:
    """
    Export the current batch of training pairs as a JSONL file.

    This function performs the full export pipeline:

    1. Fetch all ``"waiting"`` rows from ``Fine_Tuning_Staging``.
    2. Sample older ``"replay_buffer"`` rows (15% by default) for mixing.
    3. Write the combined dataset to a timestamped ``.jsonl`` file.
    4. Mark exported rows as ``"included_in_run"``.

    Returns
    -------
    dict[str, object]
        Export summary including file path, row counts, and status.
    """
    settings = get_settings()

    async with async_session_factory() as session:
        # -----------------------------------------------------------------
        # Step 1: Fetch all waiting rows
        # -----------------------------------------------------------------
        waiting_query = select(FineTuningStagingPair).where(
            FineTuningStagingPair.batch_status == "waiting"
        )
        result = await session.execute(waiting_query)
        waiting_rows = list(result.scalars().all())

        if not waiting_rows:
            return {"status": "no_data", "message": "No waiting rows to export."}

        new_count = len(waiting_rows)
        logger.info("export_fetched_waiting", count=new_count)

        # -----------------------------------------------------------------
        # Step 2: Mix in replay buffer data
        # -----------------------------------------------------------------
        # Calculate how many older rows to include.
        # E.g., if ratio=0.15 and we have 500 new rows, we add ~88 older rows.
        replay_ratio = settings.fine_tuning_replay_buffer_ratio
        replay_count = math.ceil(new_count * replay_ratio)

        replay_rows = []
        if replay_count > 0:
            # Fetch a random sample of older rows marked as replay_buffer
            # or previously included_in_run (high-quality historical data).
            replay_query = (
                select(FineTuningStagingPair)
                .where(
                    FineTuningStagingPair.batch_status.in_(
                        ["replay_buffer", "included_in_run"]
                    )
                )
                .order_by(func.random())
                .limit(replay_count)
            )
            replay_result = await session.execute(replay_query)
            replay_rows = list(replay_result.scalars().all())

        logger.info(
            "export_replay_buffer",
            requested=replay_count,
            available=len(replay_rows),
        )

        # -----------------------------------------------------------------
        # Step 3: Format and write the JSONL file
        # -----------------------------------------------------------------
        # Combine new and replay data
        all_rows = waiting_rows + replay_rows

        # Build JSONL entries in the OpenAI fine-tuning format
        jsonl_lines = []
        for row in all_rows:
            entry = {
                "messages": [
                    {"role": "user", "content": row.training_prompt},
                    {"role": "assistant", "content": row.ideal_response},
                ],
            }
            jsonl_lines.append(json.dumps(entry, ensure_ascii=False))

        # Create the export directory if it doesn't exist
        export_dir = Path(settings.fine_tuning_export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        # Generate a timestamped filename
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"nightshift_finetune_{timestamp}.jsonl"
        filepath = export_dir / filename

        # Write the JSONL file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(jsonl_lines))
            f.write("\n")  # Trailing newline

        logger.info(
            "export_file_written",
            filepath=str(filepath),
            total_rows=len(all_rows),
            new_rows=new_count,
            replay_rows=len(replay_rows),
        )

        # -----------------------------------------------------------------
        # Step 4: Update batch statuses
        # -----------------------------------------------------------------
        # Mark the new rows as 'included_in_run'
        waiting_ids = [row.pair_id for row in waiting_rows]
        await session.execute(
            update(FineTuningStagingPair)
            .where(FineTuningStagingPair.pair_id.in_(waiting_ids))
            .values(batch_status="included_in_run")
        )

        await session.commit()

        # -----------------------------------------------------------------
        # Step 5 (Future): Trigger fine-tuning API if configured
        # -----------------------------------------------------------------
        if settings.fine_tuning_provider == "openai":
            logger.info(
                "finetune_api_stub",
                provider="openai",
                message="OpenAI fine-tune API integration not yet implemented.",
            )
        elif settings.fine_tuning_provider == "vllm":
            logger.info(
                "finetune_api_stub",
                provider="vllm",
                message="vLLM fine-tune API integration not yet implemented.",
            )

    return {
        "status": "exported",
        "filepath": str(filepath),
        "new_rows": new_count,
        "replay_rows": len(replay_rows),
        "total_rows": len(all_rows),
    }
