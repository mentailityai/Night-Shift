-- =============================================================================
-- Night-Shift — PostgreSQL Initialization Script
-- =============================================================================
-- This script runs automatically on first container startup via the
-- docker-entrypoint-initdb.d mechanism.  It enables the pgvector extension
-- which is required for storing and querying embedding vectors.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS vector;
