"""
Integration tests for the FastAPI application using TestClient with
dependency overrides, backed by the same InMemoryVectorStore +
FakeEmbeddingModel used in the service-level integration tests, so the
full HTTP request path (routing, validation, middleware) is exercised
without any native ML/vector-DB dependency.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import dependencies as deps
from src.api.main import app
from src.services.indexing.pipeline import IndexingPipeline
from src.services.retrieval.bm25_index import BM25KeywordIndex
from src.services.retrieval.retrieval_service import RetrievalService
from src.services.storage.sqlite_service import SQLiteStorageService
from tests.integration.test_indexing_retrieval_integration import FakeEmbeddingModel, InMemoryVectorStore


@pytest.fixture
def client(test_settings):
    vector_store = InMemoryVectorStore()
    bm25 = BM25KeywordIndex()
    storage = SQLiteStorageService(test_settings)

    from src.services.embedding.embedding_service import EmbeddingService

    embedding = EmbeddingService(test_settings)
    embedding._model = FakeEmbeddingModel()

    pipeline = IndexingPipeline(
        vector_store=vector_store, embedding_service=embedding, bm25_index=bm25, storage=storage,
    )
    retrieval = RetrievalService(
        vector_store=vector_store, bm25_index=bm25, embedding_service=embedding, storage=storage,
    )

    app.dependency_overrides[deps.get_settings] = lambda: test_settings
    app.dependency_overrides[deps.get_storage_service] = lambda: storage
    app.dependency_overrides[deps.get_vector_store] = lambda: vector_store
    app.dependency_overrides[deps.get_bm25_index] = lambda: bm25
    app.dependency_overrides[deps.get_embedding_service] = lambda: embedding
    app.dependency_overrides[deps.get_indexing_pipeline] = lambda: pipeline
    app.dependency_overrides[deps.get_retrieval_service] = lambda: retrieval

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


class TestHealthRoutes:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["vector_backend"] == "faiss"

    def test_readiness_reflects_empty_catalog(self, client):
        resp = client.get("/health/ready")
        assert resp.status_code == 200
        assert resp.json()["vector_count"] == 0


class TestCatalogRoutes:
    def test_index_catalog(self, client):
        products = [
            {"sku": "A1", "title": "Waterproof Hiking Jacket", "description": "waterproof hiking jacket"},
            {"sku": "A2", "title": "Leather Dress Shoes", "description": "formal leather shoes"},
        ]
        resp = client.post("/catalog/index", json=products)
        assert resp.status_code == 201
        body = resp.json()
        assert body["status"] == "completed"
        assert body["products_indexed"] == 2

    def test_get_product_after_index(self, client):
        products = [{"sku": "A1", "title": "Waterproof Hiking Jacket", "description": "test"}]
        index_resp = client.post("/catalog/index", json=products)
        assert index_resp.status_code == 201

        stats_resp = client.get("/catalog/stats")
        assert stats_resp.json()["product_count"] == 1

    def test_get_nonexistent_product_404(self, client):
        resp = client.get("/catalog/products/does-not-exist")
        assert resp.status_code == 404

    def test_upsert_single_product(self, client):
        product = {"sku": "B1", "title": "Snowboard Goggles", "description": "anti-fog goggles"}
        resp = client.post("/catalog/products", json=product)
        assert resp.status_code == 201

        get_resp = client.get(f"/catalog/products/{resp.json()['product_id']}")
        assert get_resp.status_code == 200

    def test_delete_product(self, client):
        product = {"sku": "C1", "title": "Test Product"}
        upsert_resp = client.post("/catalog/products", json=product)
        product_id = upsert_resp.json()["product_id"]

        delete_resp = client.delete(f"/catalog/products/{product_id}")
        assert delete_resp.status_code == 204

        get_resp = client.get(f"/catalog/products/{product_id}")
        assert get_resp.status_code == 404


class TestSearchRoutes:
    def test_search_after_indexing(self, client):
        products = [
            {"sku": "A1", "title": "Waterproof Hiking Jacket", "description": "waterproof jacket for hiking"},
            {"sku": "A2", "title": "Leather Dress Shoes", "description": "formal leather dress shoes"},
        ]
        client.post("/catalog/index", json=products)

        resp = client.post("/search", json={"query_text": "waterproof hiking", "fusion_strategy": "rrf", "top_k": 5})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["results"]) > 0
        assert body["results"][0]["product"]["title"] == "Waterproof Hiking Jacket"

    def test_search_empty_catalog_returns_no_results(self, client):
        resp = client.post("/search", json={"query_text": "anything", "fusion_strategy": "rrf"})
        assert resp.status_code == 200
        assert resp.json()["results"] == []

    def test_search_invalid_query_returns_422(self, client):
        resp = client.post("/search", json={"query_text": "", "fusion_strategy": "rrf"})
        assert resp.status_code == 422

    def test_search_with_filters(self, client):
        products = [
            {"sku": "A1", "title": "Hiking Jacket", "category": "Outerwear", "description": "waterproof"},
            {"sku": "A2", "title": "Hiking Shoes", "category": "Footwear", "description": "waterproof"},
        ]
        client.post("/catalog/index", json=products)

        resp = client.post(
            "/search",
            json={"query_text": "waterproof", "fusion_strategy": "rrf", "filters": {"category": "Footwear"}},
        )
        body = resp.json()
        assert all(r["product"]["category"] == "Footwear" for r in body["results"])


class TestMetricsEndpoint:
    def test_metrics_responds(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
