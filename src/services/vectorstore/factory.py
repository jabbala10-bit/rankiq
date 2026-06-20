"""
VectorStore factory — translates `Settings.vector_backend` into a
concrete VectorStore implementation. This is the single place that
knows about all three backend classes; everywhere else in the codebase
depends only on the abstract `VectorStore` interface (ADR-001).
"""
from __future__ import annotations

from typing import Optional

from src.config.settings import Settings, get_settings
from src.domain.exceptions import ConfigurationError
from src.domain.schemas import VectorBackend
from src.services.vectorstore.base import VectorStore
from src.services.vectorstore.faiss_store import FAISSVectorStore
from src.services.vectorstore.pgvector_store import PgVectorStore
from src.services.vectorstore.qdrant_store import QdrantVectorStore


def create_vector_store(settings: Optional[Settings] = None) -> VectorStore:
    """
    Constructs the VectorStore implementation selected by
    `settings.vector_backend`. This is the only function in the codebase
    that should ever import all three concrete backend classes together —
    every other module should depend on `VectorStore` (the interface) and
    receive a constructed instance via dependency injection.
    """
    settings = settings or get_settings()
    backend = VectorBackend(settings.vector_backend)

    if backend == VectorBackend.FAISS:
        return FAISSVectorStore(settings)
    if backend == VectorBackend.QDRANT:
        return QdrantVectorStore(settings)
    if backend == VectorBackend.PGVECTOR:
        return PgVectorStore(settings)

    raise ConfigurationError(f"Unhandled vector backend: {backend}")
