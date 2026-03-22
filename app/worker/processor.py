# =============================================================================
# Night-Shift — LLM Output Processor
# =============================================================================
# Receives the raw JSON text response from the LLM client and performs:
#
#   1. JSON parsing with validation
#   2. Embedding generation for the extracted rule
#   3. Database writes to Extracted_Active_Rules and Fine_Tuning_Staging
#   4. Status updates on the source Raw_Interaction_Log
#
# This module is intentionally separated from the LLM client and the
# Celery task orchestration so each concern can be tested independently.
# =============================================================================

from __future__ import annotations

import json
import uuid
from typing import TypedDict

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.embeddings import generate_embedding
from app.core.logging import get_logger
from app.db.models import (
    ExtractedActiveRule,
    FineTuningStagingPair,
    RawInteractionLog,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Type definitions for the expected LLM JSON output
# ---------------------------------------------------------------------------
class ExtractedRuleDict(TypedDict):
    """Shape of the ``extracted_rule`` object in the LLM response."""
    rule_summary: str


class FineTuningPairDict(TypedDict):
    """Shape of the ``fine_tuning_pair`` object in the LLM response."""
    training_prompt: str
    ideal_response: str


class NightShiftOutput(TypedDict):
    """Complete shape of the Night-Shift Agent JSON output."""
    status: str                         # "processed" or "discarded"
    extracted_rule: ExtractedRuleDict
    fine_tuning_pair: FineTuningPairDict


# ---------------------------------------------------------------------------
# JSON Parsing
# ---------------------------------------------------------------------------
def parse_llm_response(raw_text: str) -> NightShiftOutput:
    """
    Parse and validate the raw LLM response text into a typed dict.

    Parameters
    ----------
    raw_text : str
        The raw text returned by the LLM (expected to be valid JSON).

    Returns
    -------
    NightShiftOutput
        Parsed and validated output.

    Raises
    ------
    ValueError
        If the JSON is malformed or required fields are missing.
    """
    # Strip any accidental whitespace or markdown code fences the LLM
    # might include despite the prompt instructions.
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        # Remove markdown code fences (```json ... ```)
        lines = cleaned.split("\n")
        # Drop first and last lines if they are fences
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM response is not valid JSON: {exc}"
        ) from exc

    # --- Validate required top-level fields ---
    if "status" not in data:
        raise ValueError("LLM response missing required field: 'status'")

    if data["status"] not in ("processed", "discarded"):
        raise ValueError(
            f"Unexpected status value: '{data['status']}'. "
            "Expected 'processed' or 'discarded'."
        )

    # For "discarded" status, rule and pair fields may be empty/minimal
    if data["status"] == "processed":
        # Validate nested structures
        if "extracted_rule" not in data:
            raise ValueError("Missing 'extracted_rule' in processed output.")
        if "rule_summary" not in data["extracted_rule"]:
            raise ValueError("Missing 'rule_summary' in extracted_rule.")
        if "fine_tuning_pair" not in data:
            raise ValueError("Missing 'fine_tuning_pair' in processed output.")
        if "training_prompt" not in data["fine_tuning_pair"]:
            raise ValueError("Missing 'training_prompt' in fine_tuning_pair.")
        if "ideal_response" not in data["fine_tuning_pair"]:
            raise ValueError("Missing 'ideal_response' in fine_tuning_pair.")

    return data


# ---------------------------------------------------------------------------
# Database Persistence
# ---------------------------------------------------------------------------
async def persist_processed_result(
    session: AsyncSession,
    log: RawInteractionLog,
    parsed: NightShiftOutput,
) -> None:
    """
    Write the processed results to the database and update the log status.

    For ``status == "processed"``:
      - Creates a new ``ExtractedActiveRule`` with an embedding.
      - Creates a new ``FineTuningStagingPair``.
      - Sets the log status to ``"processed"``.

    For ``status == "discarded"``:
      - Sets the log status to ``"discarded"`` (no rule/pair created).

    Parameters
    ----------
    session : AsyncSession
        Active database session (caller manages commit/rollback).
    log : RawInteractionLog
        The raw log entry being processed.
    parsed : NightShiftOutput
        Validated output from ``parse_llm_response``.
    """
    if parsed["status"] == "discarded":
        # The human edit was trivial (e.g., typo fix) — skip it
        log.status = "discarded"
        logger.info("log_discarded", log_id=str(log.log_id))
        return

    # --- "processed" path ---

    # 1. Generate a vector embedding for the rule summary so it can
    #    be retrieved via semantic search in the MCP fast loop.
    rule_summary = parsed["extracted_rule"]["rule_summary"]

    logger.info(
        "generating_embedding",
        log_id=str(log.log_id),
        rule_summary=rule_summary[:80],
    )
    embedding = generate_embedding(rule_summary)

    # 2. Create the active rule record
    rule = ExtractedActiveRule(
        source_log_id=log.log_id,
        rule_summary=rule_summary,
        embedding=embedding,
        status="active",
    )
    session.add(rule)

    # 3. Create the fine-tuning staging pair
    pair = FineTuningStagingPair(
        source_log_id=log.log_id,
        training_prompt=parsed["fine_tuning_pair"]["training_prompt"],
        ideal_response=parsed["fine_tuning_pair"]["ideal_response"],
        batch_status="waiting",
    )
    session.add(pair)

    # 4. Mark the raw log as successfully processed
    log.status = "processed"

    logger.info(
        "log_processed",
        log_id=str(log.log_id),
        rule_id=str(rule.rule_id),
        pair_id=str(pair.pair_id),
    )
