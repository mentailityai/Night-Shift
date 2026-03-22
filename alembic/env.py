# =============================================================================
# Night-Shift — Alembic Environment Configuration
# =============================================================================
# This file is executed by Alembic to configure the migration environment.
# It reads the database URL from the application's centralized config
# (``app.core.config``) so there is a single source of truth.
# =============================================================================

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Import the ORM Base so Alembic can detect all registered models.
from app.db.models import Base  # noqa: F401 — registers models with metadata
from app.core.config import get_settings

# ---------------------------------------------------------------------------
# Alembic Config object — provides access to values in alembic.ini.
# ---------------------------------------------------------------------------
config = context.config

# Standard Python logging from the ini file
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ---------------------------------------------------------------------------
# Override the sqlalchemy.url with the application's sync database URL.
# This ensures migrations always target the same database as the app.
# ---------------------------------------------------------------------------
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.sync_database_url)

# ---------------------------------------------------------------------------
# Target metadata — Alembic uses this to compare the current DB state
# against the ORM models and generate migration diffs.
# ---------------------------------------------------------------------------
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    Generates SQL scripts without requiring a live database connection.
    Useful for code review or applying migrations in air-gapped environments.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    Connects to the database and applies migrations directly.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


# ---------------------------------------------------------------------------
# Entry point — detect whether we are running online or offline.
# ---------------------------------------------------------------------------
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
