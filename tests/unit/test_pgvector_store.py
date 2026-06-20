"""Unit tests for src/services/vectorstore/pgvector_store.py."""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from src.domain.exceptions import VectorStoreError
from src.domain.schemas import ProductEmbedding
from src.services.vectorstore.pgvector_store import PgVectorStore


def _embedding(product_id: str, dims: int = 8) -> ProductEmbedding:
    return ProductEmbedding(product_id=product_id, vector=[0.1] * dims, model_name="test", dimensions=dims)


class FakeCursor:
    def __init__(self, fetch_result=None):
        self._fetch_result = fetch_result or []
        self.executed = []

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def fetchall(self):
        return self._fetch_result

    def fetchone(self):
        return self._fetch_result[0] if self._fetch_result else (0,)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class FakeConnection:
    def __init__(self, fetch_result=None):
        self._fetch_result = fetch_result
        self.cursor_calls = []

    def cursor(self):
        cur = FakeCursor(self._fetch_result)
        self.cursor_calls.append(cur)
        return cur


@pytest.fixture
def fake_conn() -> FakeConnection:
    return FakeConnection()


class TestUpsert:
    def test_upsert_executes_insert_for_each_embedding(self, test_settings, fake_conn):
        store = PgVectorStore(test_settings, connection=fake_conn)
        store.upsert([_embedding("p1"), _embedding("p2")])
        # ensure_table + 2 inserts
        all_queries = [q for cur in fake_conn.cursor_calls for q, _ in cur.executed]
        insert_queries = [q for q in all_queries if "INSERT INTO" in q]
        assert len(insert_queries) == 2

    def test_upsert_empty_list_is_noop(self, test_settings, fake_conn):
        store = PgVectorStore(test_settings, connection=fake_conn)
        store.upsert([])
        assert fake_conn.cursor_calls == []

    def test_upsert_error_raises_vector_store_error(self, test_settings):
        bad_conn = MagicMock()
        bad_conn.cursor.side_effect = RuntimeError("connection lost")
        store = PgVectorStore(test_settings, connection=bad_conn)
        with pytest.raises(VectorStoreError):
            store.upsert([_embedding("p1")])


class TestSearch:
    def test_search_returns_hits(self, test_settings):
        conn = FakeConnection(fetch_result=[("p1", 0.95), ("p2", 0.80)])
        store = PgVectorStore(test_settings, connection=conn)
        results = store.search([0.1] * 8, top_k=5)
        assert len(results) == 2
        assert results[0].product_id == "p1"
        assert results[0].score == 0.95

    def test_search_empty_table_returns_empty_list(self, test_settings):
        conn = FakeConnection(fetch_result=[])
        store = PgVectorStore(test_settings, connection=conn)
        results = store.search([0.1] * 8, top_k=5)
        assert results == []


class TestDeleteAndCount:
    def test_delete_executes_delete_query(self, test_settings, fake_conn):
        store = PgVectorStore(test_settings, connection=fake_conn)
        store.delete(["p1"])
        all_queries = [q for cur in fake_conn.cursor_calls for q, _ in cur.executed]
        assert any("DELETE FROM" in q for q in all_queries)

    def test_delete_empty_list_is_noop(self, test_settings, fake_conn):
        store = PgVectorStore(test_settings, connection=fake_conn)
        store.delete([])
        assert fake_conn.cursor_calls == []

    def test_count_returns_row_count(self, test_settings):
        conn = FakeConnection(fetch_result=[(7,)])
        store = PgVectorStore(test_settings, connection=conn)
        assert store.count() == 7


class TestClear:
    def test_clear_executes_truncate(self, test_settings, fake_conn):
        store = PgVectorStore(test_settings, connection=fake_conn)
        store.clear()
        all_queries = [q for cur in fake_conn.cursor_calls for q, _ in cur.executed]
        assert any("TRUNCATE" in q for q in all_queries)
