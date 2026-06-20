"""
pgvector VectorStore implementation.

pgvector is a Postgres extension — the simplest ops story of the three
backends if Postgres is already part of the stack (no new service to
run at all), at the cost of needing raw SQL for vector operations since
there's no high-level client library equivalent to qdrant_client. Uses
psycopg's connection directly; deferred import so unit tests don't need
psycopg/a live Postgres instance.
"""
from __future__ import annotations

from typing import Optional

from src.config.settings import Settings, get_settings
from src.domain.exceptions import VectorStoreError, VectorStoreUnavailableError
from src.domain.schemas import ProductEmbedding
from src.observability.logging import get_logger
from src.services.vectorstore.base import VectorSearchHit, VectorStore

logger = get_logger(__name__)


class PgVectorStore(VectorStore):
    """pgvector-backed implementation using cosine distance (`<=>` operator)."""

    def __init__(self, settings: Optional[Settings] = None, connection: Optional[object] = None):
        self._settings = settings or get_settings()
        self._conn = connection  # injected in tests
        self._table = self._settings.pgvector_table
        self._ensured = connection is not None

    def _get_connection(self):
        if self._conn is not None:
            return self._conn
        try:
            import psycopg  # deferred import

            self._conn = psycopg.connect(self._settings.pgvector_dsn, autocommit=True)
            return self._conn
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreUnavailableError(f"Could not connect to Postgres: {exc}") from exc

    def _ensure_table(self) -> None:
        if self._ensured:
            return
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._table} (
                        product_id TEXT PRIMARY KEY,
                        embedding vector({self._settings.embedding_dimensions})
                    );
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {self._table}_embedding_idx
                    ON {self._table} USING hnsw (embedding vector_cosine_ops);
                    """
                )
            self._ensured = True
            logger.info("pgvector_table_ensured", table=self._table)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to ensure pgvector table: {exc}") from exc

    def upsert(self, embeddings: list[ProductEmbedding]) -> None:
        if not embeddings:
            return
        self._ensure_table()
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                for emb in embeddings:
                    vector_literal = "[" + ",".join(str(x) for x in emb.vector) + "]"
                    cur.execute(
                        f"""
                        INSERT INTO {self._table} (product_id, embedding)
                        VALUES (%s, %s::vector)
                        ON CONFLICT (product_id) DO UPDATE SET embedding = EXCLUDED.embedding
                        """,
                        (emb.product_id, vector_literal),
                    )
            logger.info("pgvector_upsert_complete", count=len(embeddings))
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"pgvector upsert failed: {exc}") from exc

    def search(self, query_vector: list[float], top_k: int) -> list[VectorSearchHit]:
        self._ensure_table()
        conn = self._get_connection()
        try:
            vector_literal = "[" + ",".join(str(x) for x in query_vector) + "]"
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT product_id, 1 - (embedding <=> %s::vector) AS score
                    FROM {self._table}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (vector_literal, vector_literal, top_k),
                )
                rows = cur.fetchall()
            return [VectorSearchHit(product_id=row[0], score=float(row[1])) for row in rows]
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"pgvector search failed: {exc}") from exc

    def delete(self, product_ids: list[str]) -> None:
        if not product_ids:
            return
        self._ensure_table()
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {self._table} WHERE product_id = ANY(%s)", (product_ids,))
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"pgvector delete failed: {exc}") from exc

    def count(self) -> int:
        self._ensure_table()
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self._table}")
                row = cur.fetchone()
            return int(row[0])
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"pgvector count failed: {exc}") from exc

    def clear(self) -> None:
        self._ensure_table()
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {self._table}")
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"pgvector clear failed: {exc}") from exc
