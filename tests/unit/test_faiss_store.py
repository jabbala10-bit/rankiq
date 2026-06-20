"""
Unit tests for src/services/vectorstore/faiss_store.py.

Uses a hand-rolled fake FAISS index (matching the small subset of the
real faiss API this implementation depends on: IndexFlatIP-like add/
search/remove semantics) injected via the constructor, so these tests
run without the real faiss package installed.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.domain.exceptions import DimensionMismatchError
from src.domain.schemas import ProductEmbedding
from src.services.vectorstore.faiss_store import FAISSVectorStore


class FakeFaissIndex:
    """Minimal fake reproducing IndexIDMap2(IndexFlatIP(...)) behavior for tests."""

    def __init__(self):
        self._vectors: dict[int, np.ndarray] = {}

    @property
    def ntotal(self) -> int:
        return len(self._vectors)

    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        for vec, id_ in zip(vectors, ids):
            self._vectors[int(id_)] = vec

    def remove_ids(self, ids: np.ndarray) -> None:
        for id_ in ids:
            self._vectors.pop(int(id_), None)

    def search(self, query: np.ndarray, k: int):
        if not self._vectors:
            return np.array([[]]), np.array([[]], dtype="int64")
        ids = list(self._vectors.keys())
        vectors = np.array([self._vectors[i] for i in ids])
        scores = vectors @ query[0]  # inner product
        order = np.argsort(scores)[::-1][:k]
        top_scores = scores[order]
        top_ids = np.array([ids[i] for i in order], dtype="int64")
        pad = k - len(top_ids)
        if pad > 0:
            top_scores = np.concatenate([top_scores, np.zeros(pad)])
            top_ids = np.concatenate([top_ids, -np.ones(pad, dtype="int64")])
        return np.array([top_scores]), np.array([top_ids])


@pytest.fixture
def store(test_settings) -> FAISSVectorStore:
    return FAISSVectorStore(test_settings, index=FakeFaissIndex())


def _embedding(product_id: str, vector: list[float]) -> ProductEmbedding:
    return ProductEmbedding(product_id=product_id, vector=vector, model_name="test", dimensions=len(vector))


class TestUpsertAndSearch:
    def test_upsert_then_search_returns_closest_match(self, store):
        store.upsert(
            [
                _embedding("p1", [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                _embedding("p2", [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ]
        )
        results = store.search([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], top_k=2)
        assert results[0].product_id == "p1"

    def test_search_on_empty_index_returns_empty_list(self, store):
        results = store.search([1.0] * 8, top_k=5)
        assert results == []

    def test_upsert_empty_list_is_noop(self, store):
        store.upsert([])
        assert store.count() == 0

    def test_dimension_mismatch_raises(self, store):
        with pytest.raises(DimensionMismatchError):
            store.upsert([_embedding("p1", [1.0, 2.0])])  # 2 dims, index expects 8

    def test_upsert_is_idempotent(self, store):
        emb = _embedding("p1", [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        store.upsert([emb])
        store.upsert([emb])  # re-upsert same product
        assert store.count() == 1


class TestDelete:
    def test_delete_removes_from_index(self, store):
        store.upsert([_embedding("p1", [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])])
        assert store.count() == 1
        store.delete(["p1"])
        assert store.count() == 0

    def test_delete_nonexistent_id_is_noop(self, store):
        store.delete(["does-not-exist"])  # should not raise

    def test_delete_empty_list_is_noop(self, store):
        store.delete([])


class TestCount:
    def test_count_reflects_upserts(self, store):
        store.upsert(
            [
                _embedding("p1", [1.0, 0, 0, 0, 0, 0, 0, 0]),
                _embedding("p2", [0, 1.0, 0, 0, 0, 0, 0, 0]),
                _embedding("p3", [0, 0, 1.0, 0, 0, 0, 0, 0]),
            ]
        )
        assert store.count() == 3


class TestClear:
    def test_clear_resets_index(self, test_settings, monkeypatch):
        store = FAISSVectorStore(test_settings, index=FakeFaissIndex())
        store.upsert([_embedding("p1", [1.0, 0, 0, 0, 0, 0, 0, 0])])
        assert store.count() == 1

        import sys
        import types

        fake_faiss = types.ModuleType("faiss")
        fake_faiss.IndexFlatIP = lambda dims: FakeFaissIndex()
        fake_faiss.IndexIDMap2 = lambda inner: inner
        monkeypatch.setitem(sys.modules, "faiss", fake_faiss)

        store.clear()
        assert store.count() == 0
