# =============================================================================
# Night-Shift — Worker Pipeline Tests
# =============================================================================
# Tests the LLM output processor and agent orchestration logic with
# mocked LLM responses.  Verifies:
#   - Valid JSON parsing produces correct rule and training pair records
#   - "discarded" status is handled correctly
#   - Invalid JSON triggers the 'failed' status path
#   - Markdown-fenced JSON is cleaned before parsing
# =============================================================================

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.worker.processor import parse_llm_response


# ---------------------------------------------------------------------------
# Sample LLM Responses
# ---------------------------------------------------------------------------
VALID_PROCESSED_RESPONSE = json.dumps({
    "status": "processed",
    "extracted_rule": {
        "rule_summary": (
            "Always capitalise 'Confidential Information' and ensure "
            "indemnification clauses exclude indirect damages."
        ),
    },
    "fine_tuning_pair": {
        "training_prompt": "Draft an NDA with standard confidentiality provisions.",
        "ideal_response": (
            "This Non-Disclosure Agreement governs the exchange of "
            "Confidential Information between the Parties..."
        ),
    },
})

VALID_DISCARDED_RESPONSE = json.dumps({
    "status": "discarded",
    "extracted_rule": {
        "rule_summary": "",
    },
    "fine_tuning_pair": {
        "training_prompt": "",
        "ideal_response": "",
    },
})

MARKDOWN_FENCED_RESPONSE = (
    "```json\n"
    + VALID_PROCESSED_RESPONSE
    + "\n```"
)


class TestParseResponse:
    """Tests for the LLM response parser."""

    def test_valid_processed_response(self):
        """A valid 'processed' JSON response should parse correctly."""
        result = parse_llm_response(VALID_PROCESSED_RESPONSE)
        assert result["status"] == "processed"
        assert "rule_summary" in result["extracted_rule"]
        assert "training_prompt" in result["fine_tuning_pair"]
        assert "ideal_response" in result["fine_tuning_pair"]

    def test_valid_discarded_response(self):
        """A valid 'discarded' JSON response should parse correctly."""
        result = parse_llm_response(VALID_DISCARDED_RESPONSE)
        assert result["status"] == "discarded"

    def test_markdown_fenced_json(self):
        """The parser should strip markdown code fences."""
        result = parse_llm_response(MARKDOWN_FENCED_RESPONSE)
        assert result["status"] == "processed"

    def test_invalid_json_raises_error(self):
        """Completely invalid JSON should raise ValueError."""
        with pytest.raises(ValueError, match="not valid JSON"):
            parse_llm_response("This is not JSON at all.")

    def test_missing_status_field(self):
        """JSON missing 'status' should raise ValueError."""
        incomplete = json.dumps({"extracted_rule": {"rule_summary": "test"}})
        with pytest.raises(ValueError, match="missing required field"):
            parse_llm_response(incomplete)

    def test_unexpected_status_value(self):
        """An unexpected status value should raise ValueError."""
        bad_status = json.dumps({"status": "unknown"})
        with pytest.raises(ValueError, match="Unexpected status"):
            parse_llm_response(bad_status)

    def test_missing_rule_in_processed(self):
        """A 'processed' response missing 'extracted_rule' should error."""
        missing_rule = json.dumps({
            "status": "processed",
            "fine_tuning_pair": {
                "training_prompt": "test",
                "ideal_response": "test",
            },
        })
        with pytest.raises(ValueError, match="extracted_rule"):
            parse_llm_response(missing_rule)

    def test_missing_training_pair_in_processed(self):
        """A 'processed' response missing 'fine_tuning_pair' should error."""
        missing_pair = json.dumps({
            "status": "processed",
            "extracted_rule": {"rule_summary": "test"},
        })
        with pytest.raises(ValueError, match="fine_tuning_pair"):
            parse_llm_response(missing_pair)
