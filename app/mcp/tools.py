# =============================================================================
# Night-Shift — MCP Server Tools
# =============================================================================
# Defines the MCP tools that primary AI drafting agents can call to
# retrieve relevant user preference rules before generating text.
#
# The main tool is ``search_active_rules``, which performs a cosine
# similarity search against the ``Extracted_Active_Rules`` table using
# pgvector's vector operators.
# =============================================================================

from __future__ import annotations

from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.embeddings import generate_embedding
from app.core.logging import get_logger
from app.db.models import ExtractedActiveRule
from app.db.session import async_session_factory

logger = get_logger(__name__)


async def search_active_rules(query: str) -> list[dict[str, Any]]:
    """
    Search for relevant user preference rules using semantic similarity.

    This function is exposed as an MCP tool.  When a primary drafting
    agent receives a new prompt, it calls this tool with a description
    of the task at hand.  The function returns the most relevant
    preference rules so the agent can incorporate them into its context.

    Parameters
    ----------
    query : str
        A natural-language description of the current drafting task
        (e.g., "drafting an indemnification clause for a biotech license").

    Returns
    -------
    list[dict[str, Any]]
        A list of matching rules, each containing:
        - ``rule_id``: UUID of the rule
        - ``rule_summary``: The actionable preference statement
        - ``score``: Cosine similarity score (higher = more relevant)

    Notes
    -----
    The search uses pgvector's ``<=>`` (cosine distance) operator.
    Results are filtered by the ``MCP_SEARCH_MIN_SCORE`` threshold and
    limited to ``MCP_SEARCH_TOP_K`` results, both configurable in ``.env``.
    """
    settings = get_settings()

    # Generate an embedding for the query text
    query_embedding = generate_embedding(query)

    async with async_session_factory() as session:
        # -----------------------------------------------------------------
        # Perform a cosine similarity search using pgvector.
        #
        # pgvector's ``<=>`` operator returns cosine *distance* (0 = identical,
        # 2 = opposite), so we convert to similarity: 1 - distance.
        #
        # We filter for:
        #   - Only 'active' rules (not archived post-fine-tune)
        #   - Similarity score above the configured minimum threshold
        # -----------------------------------------------------------------
        similarity_expr = (
            1 - ExtractedActiveRule.embedding.cosine_distance(query_embedding)
        )

        query_stmt = (
            select(
                ExtractedActiveRule.rule_id,
                ExtractedActiveRule.rule_summary,
                similarity_expr.label("score"),
            )
            .where(ExtractedActiveRule.status == "active")
            .where(similarity_expr >= settings.mcp_search_min_score)
            .order_by(similarity_expr.desc())
            .limit(settings.mcp_search_top_k)
        )

        result = await session.execute(query_stmt)
        rows = result.all()

    # Format results as a list of dictionaries
    rules = [
        {
            "rule_id": str(row.rule_id),
            "rule_summary": row.rule_summary,
            "score": round(float(row.score), 4),
        }
        for row in rows
    ]

    logger.info(
        "mcp_search_completed",
        query=query[:80],
        results_count=len(rules),
    )

    return rules
