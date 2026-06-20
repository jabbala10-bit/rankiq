"""Unit tests for src/services/vectorstore/factory.py — the strategy-selector."""
from __future__ import annotations

import pytest

from src.services.vectorstore.factory import create_vector_store
from src.services.vectorstore.faiss_store import FAISSVectorStore
from src.services.vectorstore.pgvector_store import PgVectorStore
from src.services.vectorstore.qdrant_store import QdrantVectorStore


class TestBackendSelection:
    def test_faiss_backend_selected(self, test_settings):
        settings = test_settings.model_copy(update={"vector_backend": "faiss"})
        store = create_vector_store(settings)
        assert isinstance(store, FAISSVectorStore)

    def test_qdrant_backend_selected(self, test_settings):
        settings = test_settings.model_copy(update={"vector_backend": "qdrant"})
        store = create_vector_store(settings)
        assert isinstance(store, QdrantVectorStore)

    def test_pgvector_backend_selected(self, test_settings):
        settings = test_settings.model_copy(update={"vector_backend": "pgvector"})
        store = create_vector_store(settings)
        assert isinstance(store, PgVectorStore)

    def test_invalid_backend_rejected_at_settings_construction(self):
        from src.config.settings import Settings

        with pytest.raises(ValueError):
            Settings(vector_backend="not_a_real_backend")
