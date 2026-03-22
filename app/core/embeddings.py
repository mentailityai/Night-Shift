# =============================================================================
# Night-Shift — Embedding Generation Utility
# =============================================================================
# Loads the configured embedding model (default: BAAI/bge-m3, 1024-dim)
# using ``sentence-transformers`` and exposes a simple function to convert
# text into a vector.
#
# The model is loaded lazily (on first call) and cached for the process
# lifetime to avoid repeated disk I/O.
#
# Usage:
#   from app.core.embeddings import generate_embedding
#   vector = generate_embedding("Always bold defined terms")
#   # vector is a list[float] with len == EMBEDDING_DIMENSION
# =============================================================================

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from app.core.config import get_settings
from app.core.logging import get_logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """
    Lazily load the sentence-transformer model on first use.

    The model is cached in-process so subsequent calls are instant.
    BGE-M3 is approximately 2 GB and takes a few seconds to load on
    first invocation (subsequent calls return the cached instance).
    """
    from sentence_transformers import SentenceTransformer

    settings = get_settings()
    model_name = settings.embedding_model_name

    logger.info("loading_embedding_model", model=model_name)
    model = SentenceTransformer(model_name)
    logger.info("embedding_model_loaded", model=model_name)

    return model


def generate_embedding(text: str) -> list[float]:
    """
    Convert a text string into a dense vector embedding.

    Parameters
    ----------
    text : str
        The input text to embed (e.g., a rule summary).

    Returns
    -------
    list[float]
        A list of floats with length equal to ``EMBEDDING_DIMENSION``
        from the configuration (default 1024 for BGE-M3).

    Notes
    -----
    The function normalizes the output vector to unit length, which is
    required for correct cosine similarity comparisons in pgvector.
    """
    model = _load_model()

    # ``normalize_embeddings=True`` produces unit vectors so that dot
    # product == cosine similarity (pgvector's ``<=>`` operator).
    embedding = model.encode(
        text,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # sentence-transformers returns a numpy array; convert to plain list
    # for storage in PostgreSQL via pgvector.
    return embedding.tolist()
