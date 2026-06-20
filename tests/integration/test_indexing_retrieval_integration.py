"""
Integration tests for IndexingPipeline + RetrievalService.

Uses a hand-rolled InMemoryVectorStore (a real, correct implementation
of the VectorStore interface — just backed by a Python dict instead of
FAISS/Qdrant/pgvector) and a fake embedding model that produces
deterministic, semantically-irrelevant-but-stable vectors. This proves
out the orchestration logic (indexing -> dual-write -> hybrid search ->
fusion) end to end without any native ML/vector-DB dependency, while
still exercising the real BM25KeywordIndex, real SQLiteStorageService,
and real fusion functions.
"""
from __future__ import annotations

import math

import pytest

from src.domain.schemas import FusionStrategy, Product, SearchQuery, VectorBackend
from src.services.embedding.embedding_service import EmbeddingService
from src.services.indexing.pipeline import IndexingPipeline
from src.services.retrieval.bm25_index import BM25KeywordIndex
from src.services.retrieval.retrieval_service import RetrievalService
from src.services.storage.sqlite_service import SQLiteStorageService
from src.services.vectorstore.base import VectorSearchHit, VectorStore


class InMemoryVectorStore(VectorStore):
    """A real, correct VectorStore implementation backed by a plain dict — used only in tests."""

    def __init__(self):
        self._vectors: dict[str, list[float]] = {}

    def upsert(self, embeddings) -> None:
        for emb in embeddings:
            self._vectors[emb.product_id] = emb.vector

    def search(self, query_vector, top_k):
        def cosine_sim(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

        scored = [(pid, cosine_sim(query_vector, vec)) for pid, vec in self._vectors.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [VectorSearchHit(product_id=pid, score=score) for pid, score in scored[:top_k]]

    def delete(self, product_ids) -> None:
        for pid in product_ids:
            self._vectors.pop(pid, None)

    def count(self) -> int:
        return len(self._vectors)

    def clear(self) -> None:
        self._vectors.clear()


class FakeEmbeddingModel:
    """
    Produces a deterministic vector per text by hashing keywords into
    fixed dimensions — not semantically meaningful in a deep-learning
    sense, but stable and designed so that texts sharing words produce
    similar vectors, which is enough to test that "semantically closer"
    inputs really do retrieve as closer in the fake vector space.
    """

    def encode(self, texts, normalize_embeddings=True, batch_size=None):
        import numpy as np

        single = isinstance(texts, str)
        text_list = [texts] if single else list(texts)

        vocab = [
            "waterproof", "hiking", "jacket", "parka", "winter", "insulated",
            "shoes", "running", "trail", "leather", "dress", "backpack", "bag",
        ]
        vectors = []
        for text in text_list:
            lower = text.lower()
            vec = np.array([1.0 if word in lower else 0.0 for word in vocab])
            norm = np.linalg.norm(vec)
            vectors.append(vec / norm if norm > 0 else vec)

        result = np.array(vectors)
        return result[0] if single else result


@pytest.fixture
def embedding_service(test_settings) -> EmbeddingService:
    service = EmbeddingService(test_settings)
    service._model = FakeEmbeddingModel()
    return service


@pytest.fixture
def vector_store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def bm25_index() -> BM25KeywordIndex:
    return BM25KeywordIndex()


@pytest.fixture
def storage(test_settings) -> SQLiteStorageService:
    return SQLiteStorageService(test_settings)


@pytest.fixture
def indexing_pipeline(vector_store, embedding_service, bm25_index, storage) -> IndexingPipeline:
    return IndexingPipeline(
        vector_store=vector_store,
        embedding_service=embedding_service,
        bm25_index=bm25_index,
        storage=storage,
        vector_backend=VectorBackend.FAISS,
    )


@pytest.fixture
def retrieval_service(vector_store, bm25_index, embedding_service, storage) -> RetrievalService:
    return RetrievalService(
        vector_store=vector_store,
        bm25_index=bm25_index,
        embedding_service=embedding_service,
        storage=storage,
    )


class TestIndexingPipeline:
    def test_index_catalog_persists_to_all_three_stores(
        self, indexing_pipeline, sample_catalog, vector_store, bm25_index, storage
    ):
        job = indexing_pipeline.index_catalog(sample_catalog)

        assert job.status.value == "completed"
        assert job.products_indexed == len(sample_catalog)
        assert vector_store.count() == len(sample_catalog)
        assert bm25_index.count() == len(sample_catalog)
        assert storage.count_products() == len(sample_catalog)

    def test_upsert_single_product_updates_all_stores(self, indexing_pipeline, vector_store, bm25_index, storage):
        product = Product(sku="NEW-1", title="Snowboard Goggles", description="anti-fog ski goggles")
        indexing_pipeline.upsert_product(product)

        assert vector_store.count() == 1
        assert bm25_index.count() == 1
        assert storage.get_product(product.product_id) is not None

    def test_delete_product_removes_from_all_stores(self, indexing_pipeline, sample_catalog, vector_store, bm25_index, storage):
        indexing_pipeline.index_catalog(sample_catalog)
        target = sample_catalog[0]

        indexing_pipeline.delete_product(target.product_id)

        assert storage.get_product(target.product_id) is None
        assert vector_store.count() == len(sample_catalog) - 1
        assert bm25_index.count() == len(sample_catalog) - 1


class TestHybridRetrieval:
    def test_rrf_search_returns_relevant_products(self, indexing_pipeline, retrieval_service, sample_catalog):
        indexing_pipeline.index_catalog(sample_catalog)

        query = SearchQuery(query_text="waterproof hiking jacket", fusion_strategy=FusionStrategy.RRF, top_k=5)
        result = retrieval_service.search(query)

        assert len(result.results) > 0
        top_titles = [r.product.title for r in result.results]
        assert any("Jacket" in t or "Backpack" in t for t in top_titles)

    def test_vector_only_search(self, indexing_pipeline, retrieval_service, sample_catalog):
        indexing_pipeline.index_catalog(sample_catalog)
        query = SearchQuery(query_text="waterproof hiking", fusion_strategy=FusionStrategy.VECTOR_ONLY, top_k=5)
        result = retrieval_service.search(query)
        assert result.keyword_candidate_count == 0
        assert result.vector_candidate_count > 0

    def test_keyword_only_search(self, indexing_pipeline, retrieval_service, sample_catalog):
        indexing_pipeline.index_catalog(sample_catalog)
        query = SearchQuery(query_text="leather dress shoes", fusion_strategy=FusionStrategy.KEYWORD_ONLY, top_k=5)
        result = retrieval_service.search(query)
        assert result.vector_candidate_count == 0
        assert result.keyword_candidate_count > 0
        assert any("Dress Shoes" in r.product.title for r in result.results)

    def test_weighted_sum_search(self, indexing_pipeline, retrieval_service, sample_catalog):
        indexing_pipeline.index_catalog(sample_catalog)
        query = SearchQuery(query_text="trail shoes", fusion_strategy=FusionStrategy.WEIGHTED_SUM, top_k=5)
        result = retrieval_service.search(query)
        assert len(result.results) > 0

    def test_filters_exclude_non_matching_category(self, indexing_pipeline, retrieval_service, sample_catalog):
        indexing_pipeline.index_catalog(sample_catalog)
        query = SearchQuery(
            query_text="waterproof", fusion_strategy=FusionStrategy.RRF, top_k=10,
            filters={"category": "Footwear"},
        )
        result = retrieval_service.search(query)
        assert all(r.product.category == "Footwear" for r in result.results)

    def test_top_k_limits_result_count(self, indexing_pipeline, retrieval_service, sample_catalog):
        indexing_pipeline.index_catalog(sample_catalog)
        query = SearchQuery(query_text="shoes jacket backpack", fusion_strategy=FusionStrategy.RRF, top_k=2)
        result = retrieval_service.search(query)
        assert len(result.results) <= 2

    def test_search_on_empty_catalog_returns_no_results(self, retrieval_service):
        query = SearchQuery(query_text="anything", fusion_strategy=FusionStrategy.RRF, top_k=5)
        result = retrieval_service.search(query)
        assert result.results == []

    def test_search_result_includes_rank_provenance(self, indexing_pipeline, retrieval_service, sample_catalog):
        indexing_pipeline.index_catalog(sample_catalog)
        query = SearchQuery(query_text="waterproof hiking jacket", fusion_strategy=FusionStrategy.RRF, top_k=5)
        result = retrieval_service.search(query)

        # At least some results should show provenance from one or both retrieval methods
        assert any(r.vector_rank is not None or r.keyword_rank is not None for r in result.results)
