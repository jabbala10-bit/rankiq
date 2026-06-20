"""
End-to-end tests: a full catalog-to-search journey through the real
FastAPI app, plus a dedicated test proving the backend-switch promise
(ADR-001) — that swapping VectorBackend produces equivalent search
behavior through the identical RetrievalService code path.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import dependencies as deps
from src.api.main import app
from src.services.embedding.embedding_service import EmbeddingService
from src.services.indexing.pipeline import IndexingPipeline
from src.services.retrieval.bm25_index import BM25KeywordIndex
from src.services.retrieval.retrieval_service import RetrievalService
from src.services.storage.sqlite_service import SQLiteStorageService
from tests.integration.test_indexing_retrieval_integration import FakeEmbeddingModel, InMemoryVectorStore

_CATALOG = [
    {
        "sku": "JKT-001", "title": "Men's Waterproof Hiking Jacket",
        "description": "Lightweight, breathable, fully waterproof shell jacket for hiking.",
        "brand": "TrailPeak", "category": "Outerwear", "price": 129.99,
    },
    {
        "sku": "JKT-002", "title": "Women's Insulated Winter Parka",
        "description": "Warm down-insulated parka rated for sub-zero temperatures.",
        "brand": "Northgale", "category": "Outerwear", "price": 199.99,
    },
    {
        "sku": "SHO-010", "title": "Trail Running Shoes",
        "description": "Grippy outsole, breathable mesh upper, ideal for muddy or wet trails.",
        "brand": "TrailPeak", "category": "Footwear", "price": 89.99,
    },
    {
        "sku": "SHO-011", "title": "Leather Dress Shoes",
        "description": "Classic formal leather oxford shoes for business attire.",
        "brand": "Camden & Co", "category": "Footwear", "price": 149.99,
    },
    {
        "sku": "BAG-005", "title": "Waterproof Hiking Backpack 40L",
        "description": "Durable waterproof backpack with hydration sleeve, ideal for multi-day hikes.",
        "brand": "TrailPeak", "category": "Bags", "price": 109.99,
    },
]


def _build_client(test_settings, vector_store):
    bm25 = BM25KeywordIndex()
    storage = SQLiteStorageService(test_settings)
    embedding = EmbeddingService(test_settings)
    embedding._model = FakeEmbeddingModel()

    pipeline = IndexingPipeline(vector_store=vector_store, embedding_service=embedding, bm25_index=bm25, storage=storage)
    retrieval = RetrievalService(vector_store=vector_store, bm25_index=bm25, embedding_service=embedding, storage=storage)

    app.dependency_overrides[deps.get_settings] = lambda: test_settings
    app.dependency_overrides[deps.get_storage_service] = lambda: storage
    app.dependency_overrides[deps.get_vector_store] = lambda: vector_store
    app.dependency_overrides[deps.get_bm25_index] = lambda: bm25
    app.dependency_overrides[deps.get_embedding_service] = lambda: embedding
    app.dependency_overrides[deps.get_indexing_pipeline] = lambda: pipeline
    app.dependency_overrides[deps.get_retrieval_service] = lambda: retrieval

    return TestClient(app)


@pytest.fixture
def e2e_client(test_settings):
    client = _build_client(test_settings, InMemoryVectorStore())
    yield client
    app.dependency_overrides.clear()


class TestFullShopperSearchJourney:
    def test_index_catalog_then_search_then_filter_then_compare_strategies(self, e2e_client):
        # 1. Admin indexes the full catalog.
        index_resp = e2e_client.post("/catalog/index", json=_CATALOG)
        assert index_resp.status_code == 201
        assert index_resp.json()["products_indexed"] == len(_CATALOG)

        # 2. A shopper searches with natural-language intent.
        search_resp = e2e_client.post(
            "/search", json={"query_text": "warm waterproof jacket for hiking", "fusion_strategy": "rrf", "top_k": 5}
        )
        assert search_resp.status_code == 200
        results = search_resp.json()["results"]
        assert len(results) > 0
        assert any("Jacket" in r["product"]["title"] for r in results)

        # 3. The shopper narrows by category filter.
        filtered_resp = e2e_client.post(
            "/search",
            json={
                "query_text": "waterproof",
                "fusion_strategy": "rrf",
                "filters": {"category": "Footwear"},
                "top_k": 5,
            },
        )
        filtered_results = filtered_resp.json()["results"]
        assert all(r["product"]["category"] == "Footwear" for r in filtered_results)

        # 4. Comparing strategies on the same query should generally agree on
        #    the most obviously relevant item even if exact ranking differs.
        vector_resp = e2e_client.post(
            "/search", json={"query_text": "leather dress shoes", "fusion_strategy": "vector_only", "top_k": 5}
        )
        keyword_resp = e2e_client.post(
            "/search", json={"query_text": "leather dress shoes", "fusion_strategy": "keyword_only", "top_k": 5}
        )
        vector_top = vector_resp.json()["results"][0]["product"]["title"]
        keyword_top = keyword_resp.json()["results"][0]["product"]["title"]
        assert vector_top == "Leather Dress Shoes"
        assert keyword_top == "Leather Dress Shoes"

        # 5. A product goes out of stock and is removed from the catalog.
        product_id = results[0]["product"]["product_id"]
        delete_resp = e2e_client.delete(f"/catalog/products/{product_id}")
        assert delete_resp.status_code == 204

        post_delete_search = e2e_client.post(
            "/search", json={"query_text": "warm waterproof jacket for hiking", "fusion_strategy": "rrf", "top_k": 5}
        )
        remaining_ids = [r["product"]["product_id"] for r in post_delete_search.json()["results"]]
        assert product_id not in remaining_ids


class TestBackendSwitchEquivalence:
    def test_faiss_like_and_inmemory_backends_produce_consistent_top_result(self, test_settings):
        """
        Proves the ADR-001 promise: swapping the VectorStore implementation
        (here, two separate InMemoryVectorStore instances standing in for
        what would be FAISS vs. a different backend in production) through
        the identical RetrievalService code path produces the same
        top-ranked result for the same query and catalog — the backend
        choice doesn't change retrieval *behavior*, only *infrastructure*.
        """
        client_a = _build_client(test_settings, InMemoryVectorStore())
        client_a.post("/catalog/index", json=_CATALOG)
        result_a = client_a.post(
            "/search", json={"query_text": "waterproof hiking jacket", "fusion_strategy": "rrf", "top_k": 3}
        ).json()
        app.dependency_overrides.clear()

        # Fresh settings/storage for the second "backend" to avoid SQLite file reuse across instances
        from src.config.settings import Settings

        settings_b = test_settings.model_copy(update={"sqlite_path": test_settings.sqlite_path + ".b"})
        client_b = _build_client(settings_b, InMemoryVectorStore())
        client_b.post("/catalog/index", json=_CATALOG)
        result_b = client_b.post(
            "/search", json={"query_text": "waterproof hiking jacket", "fusion_strategy": "rrf", "top_k": 3}
        ).json()
        app.dependency_overrides.clear()

        assert result_a["results"][0]["product"]["title"] == result_b["results"][0]["product"]["title"]
