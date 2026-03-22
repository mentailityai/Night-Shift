# =============================================================================
# Night-Shift — SQLAlchemy ORM Models
# =============================================================================
# Defines the three core tables that manage the lifecycle of preference data:
#
#   1. RawInteractionLog   — raw captures from the frontend/application
#   2. ExtractedActiveRule  — preference rules extracted by the Night-Shift Agent
#   3. FineTuningStagingPair — cleaned prompt/response pairs awaiting export
#
# All tables use UUID primary keys for safe distributed generation and
# timestamp columns for auditability.
# =============================================================================

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    ForeignKey,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.core.config import get_settings


# ---------------------------------------------------------------------------
# Base class for all ORM models
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    """Shared declarative base for all Night-Shift ORM models."""

    pass


# ---------------------------------------------------------------------------
# Table 1: Raw_Interaction_Logs
# ---------------------------------------------------------------------------
class RawInteractionLog(Base):
    """
    Stores the raw, unprocessed capture of a user editing session.

    Each row represents a single interaction where the user accepted,
    rejected, or manually corrected an AI-generated output.  The
    ``status`` field tracks the processing lifecycle:

        pending  →  processing  →  processed | failed | discarded

    The ``error_message`` field captures any failure details if the
    Night-Shift Agent could not process the log (e.g., LLM returned
    invalid JSON).
    """

    __tablename__ = "raw_interaction_logs"

    # Primary key — auto-generated UUID
    log_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
        comment="Unique identifier for the interaction log entry.",
    )

    # Timestamp of when the interaction occurred
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        comment="When the user interaction occurred (UTC).",
    )

    # The baseline system instructions the AI was operating under
    system_prompt: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="System prompt that the AI was given during generation.",
    )

    # The specific user request or uploaded source document
    user_input: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="User request or source text provided to the AI.",
    )

    # The exact, unedited AI-generated response
    ai_output: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Original AI-generated output before any human edits.",
    )

    # The final version after the user made manual edits
    human_correction: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Final text after the user manually edited the AI output.",
    )

    # Processing lifecycle status
    status: Mapped[str] = mapped_column(
        String(20),
        default="pending",
        server_default="pending",
        nullable=False,
        index=True,
        comment=(
            "Processing status: pending → processing → "
            "processed | failed | discarded."
        ),
    )

    # Error details if processing failed
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        default=None,
        comment="Error details if the Night-Shift Agent failed to process.",
    )

    # ---- Relationships ----
    extracted_rules: Mapped[list[ExtractedActiveRule]] = relationship(
        back_populates="source_log",
        cascade="all, delete-orphan",
    )
    staging_pairs: Mapped[list[FineTuningStagingPair]] = relationship(
        back_populates="source_log",
        cascade="all, delete-orphan",
    )


# ---------------------------------------------------------------------------
# Table 2: Extracted_Active_Rules
# ---------------------------------------------------------------------------
class ExtractedActiveRule(Base):
    """
    Dynamic memory for the MCP/RAG fast loop.

    Each row is a user preference or formatting rule distilled from a raw
    interaction by the Night-Shift Agent.  The ``embedding`` column stores
    the vector representation for semantic search so that the primary
    drafting agent can retrieve relevant rules before generating text.

    Rules start as ``active`` and are moved to ``archived`` after a
    successful fine-tuning run bakes them permanently into model weights.
    """

    __tablename__ = "extracted_active_rules"

    # Primary key — auto-generated UUID
    rule_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
        comment="Unique identifier for the extracted rule.",
    )

    # Foreign key linking back to the raw interaction that produced this rule
    source_log_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("raw_interaction_logs.log_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Raw interaction log that this rule was extracted from.",
    )

    # Concise, actionable statement of the user preference
    rule_summary: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment=(
            "Concise preference statement, e.g. "
            "'Always bold defined terms in liability sections'."
        ),
    )

    # Vector embedding of the rule_summary for semantic search
    # Dimension is driven by the configured embedding model (default 1024
    # for BGE-M3).
    embedding: Mapped[list[float]] = mapped_column(
        Vector(get_settings().embedding_dimension),
        nullable=False,
        comment="Vector embedding of rule_summary for cosine similarity search.",
    )

    # Lifecycle status for this rule
    status: Mapped[str] = mapped_column(
        String(20),
        default="active",
        server_default="active",
        nullable=False,
        index=True,
        comment="'active' (in use by MCP) or 'archived' (baked into fine-tune).",
    )

    # Timestamp of when the rule was created
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        comment="When this rule was extracted (UTC).",
    )

    # ---- Relationships ----
    source_log: Mapped[RawInteractionLog] = relationship(
        back_populates="extracted_rules",
    )


# ---------------------------------------------------------------------------
# Table 3: Fine_Tuning_Staging
# ---------------------------------------------------------------------------
class FineTuningStagingPair(Base):
    """
    Holds pristine prompt-and-response pairs awaiting fine-tuning export.

    The Night-Shift Agent rewrites messy human interactions into clean,
    generalised training data suitable for model fine-tuning.

    The ``batch_status`` tracks export lifecycle:

        waiting  →  included_in_run
        (older rows may be flagged as ``replay_buffer`` for mixing)
    """

    __tablename__ = "fine_tuning_staging"

    # Primary key — auto-generated UUID
    pair_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
        comment="Unique identifier for the training pair.",
    )

    # Foreign key linking back to the raw interaction
    source_log_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("raw_interaction_logs.log_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Raw interaction log this training pair was derived from.",
    )

    # The sanitised, generalised user request
    training_prompt: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Cleaned and generalised user request for fine-tuning.",
    )

    # The perfect AI response incorporating the human's exact preference
    ideal_response: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Ideal AI response that demonstrates the learned preference.",
    )

    # Export lifecycle status
    batch_status: Mapped[str] = mapped_column(
        String(20),
        default="waiting",
        server_default="waiting",
        nullable=False,
        index=True,
        comment="'waiting', 'included_in_run', or 'replay_buffer'.",
    )

    # Timestamp of when the pair was created
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        comment="When this pair was generated (UTC).",
    )

    # ---- Relationships ----
    source_log: Mapped[RawInteractionLog] = relationship(
        back_populates="staging_pairs",
    )
