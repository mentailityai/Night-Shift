# =============================================================================
# Night-Shift — API Ingestion Endpoint Tests
# =============================================================================
# Verifies the POST /api/logs endpoint:
#   - Accepts valid payloads and creates 'pending' logs
#   - Rejects incomplete or invalid payloads
#   - Returns the correct response schema
# =============================================================================

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import AsyncClient


# ---------------------------------------------------------------------------
# Sample Payload
# ---------------------------------------------------------------------------
VALID_PAYLOAD = {
    "system_prompt": "You are a legal drafting assistant.",
    "user_input": "Draft an NDA between Company A and Company B.",
    "ai_output": "This is a Non-Disclosure Agreement between the parties...",
    "human_correction": (
        "This is a Non-Disclosure Agreement between the Parties. "
        "All references to 'Confidential Information' shall be capitalised."
    ),
}


class TestIngestionEndpoint:
    """Tests for POST /api/logs."""

    @pytest.mark.asyncio
    async def test_ingest_valid_payload(self, client: AsyncClient):
        """A valid payload should return 201 with a log_id."""
        response = await client.post("/api/logs", json=VALID_PAYLOAD)

        assert response.status_code == 201
        data = response.json()
        assert "log_id" in data
        assert data["status"] == "pending"
        assert data["message"] == "Interaction log captured successfully."

    @pytest.mark.asyncio
    async def test_ingest_missing_field(self, client: AsyncClient):
        """A payload missing a required field should return 422."""
        incomplete = {
            "system_prompt": "You are a legal assistant.",
            "user_input": "Draft an NDA.",
            # Missing: ai_output, human_correction
        }
        response = await client.post("/api/logs", json=incomplete)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_ingest_empty_field(self, client: AsyncClient):
        """An empty string for a required field should return 422."""
        empty_field = {**VALID_PAYLOAD, "ai_output": ""}
        response = await client.post("/api/logs", json=empty_field)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_health_check(self, client: AsyncClient):
        """The health endpoint should return status ok."""
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
