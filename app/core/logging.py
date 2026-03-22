# =============================================================================
# Night-Shift — Structured Logging
# =============================================================================
# Configures ``structlog`` for consistent, machine-parseable (JSON) or
# human-readable (console) log output across all modules.
#
# Usage:
#   from app.core.logging import get_logger
#   logger = get_logger(__name__)
#   logger.info("processing_started", log_id="abc-123", status="pending")
# =============================================================================

from __future__ import annotations

import logging
import sys

import structlog

from app.core.config import get_settings


def setup_logging() -> None:
    """
    Initialize the application-wide logging configuration.

    Call this once at application startup (e.g., in FastAPI lifespan or
    the Celery worker ``on_after_configure`` signal).

    The output format (JSON vs. pretty console) and level are driven by
    the ``LOG_FORMAT`` and ``LOG_LEVEL`` environment variables.
    """
    settings = get_settings()

    # -------------------------------------------------------------------------
    # Choose processors based on the configured output format.
    # "json"    → structured JSON lines (ideal for log aggregators)
    # "console" → coloured, human-friendly output (ideal for development)
    # -------------------------------------------------------------------------
    if settings.log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    # Common processor chain applied to every log event
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,        # thread-local context
        structlog.stdlib.add_logger_name,               # attach logger name
        structlog.stdlib.add_log_level,                 # attach level string
        structlog.processors.TimeStamper(fmt="iso"),    # ISO-8601 timestamp
        structlog.processors.StackInfoRenderer(),       # stack traces
        structlog.processors.UnicodeDecoder(),          # safe unicode
    ]

    # Configure structlog
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure the standard-library root logger so that third-party libraries
    # (SQLAlchemy, uvicorn, celery, …) also go through structlog's formatting.
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(settings.log_level.upper())


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Return a bound structlog logger for the given module name.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.

    Returns
    -------
    structlog.stdlib.BoundLogger
        A logger instance with structured binding support.
    """
    return structlog.get_logger(name)
