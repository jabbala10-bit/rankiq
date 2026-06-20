"""Unit tests for src/services/vectorstore/qdrant_store.py."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.domain.exceptions import VectorStoreError
from src.domain.schemas import ProductEmbedding
from src.services.vectorstore.qdrant_store import QdrantVectorStore


def _embedding(product_id: str, dims: int = 8) -> ProductEmbedding:
    return ProductEmbedding(product_id=product_id, vector=[0.1] * dims, model_name="test", dimensions=dims)


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[])
    return client


class TestUpsert:
    def test_upsert_calls_client_upsert(self, test_settings, mock_client):
        store = QdrantVectorStore(test_settings, client=mock_client)
        store.upsert([_embedding("p1")])
        mock_client.upsert.assert_called_once()

    def test_upsert_empty_list_is_noop(self, test_settings, mock_client):
        store = QdrantVectorStore(test_settings, client=mock_client)
        store.upsert([])
        mock_client.upsert.assert_not_called()

    def test_upsert_error_raises_vector_store_error(self, test_settings, mock_client):
        mock_client.upsert.side_effect = RuntimeError("connection reset")
        store = QdrantVectorStore(test_settings, client=mock_client)
        with pytest.raises(VectorStoreError):
            store.upsert([_embedding("p1")])


class TestSearch:
    def test_search_returns_hits_with_product_id_and_score(self, test_settings, mock_client):
        fake_hit = MagicMock()
        fake_hit.payload = {"product_id": "p1"}
        fake_hit.score = 0.95
        mock_client.search.return_value = [fake_hit]

        store = QdrantVectorStore(test_settings, client=mock_client)
        results = store.search([0.1] * 8, top_k=5)

        assert len(results) == 1
        assert results[0].product_id == "p1"
        assert results[0].score == 0.95

    def test_search_error_raises_vector_store_error(self, test_settings, mock_client):
        mock_client.search.side_effect = RuntimeError("timeout")
        store = QdrantVectorStore(test_settings, client=mock_client)
        with pytest.raises(VectorStoreError):
            store.search([0.1] * 8, top_k=5)


class TestDelete:
    def test_delete_calls_client_delete(self, test_settings, mock_client):
        store = QdrantVectorStore(test_settings, client=mock_client)
        store.delete(["p1", "p2"])
        mock_client.delete.assert_called_once()

    def test_delete_empty_list_is_noop(self, test_settings, mock_client):
        store = QdrantVectorStore(test_settings, client=mock_client)
        store.delete([])
        mock_client.delete.assert_not_called()


class TestCount:
    def test_count_returns_points_count(self, test_settings, mock_client):
        mock_client.get_collection.return_value = MagicMock(points_count=42)
        store = QdrantVectorStore(test_settings, client=mock_client)
        assert store.count() == 42

    def test_count_handles_none_points_count(self, test_settings, mock_client):
        mock_client.get_collection.return_value = MagicMock(points_count=None)
        store = QdrantVectorStore(test_settings, client=mock_client)
        assert store.count() == 0
