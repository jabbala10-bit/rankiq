"""
SQLite storage service: the system of record for the product catalog
itself (vector stores and the BM25 index hold derived representations,
but the canonical Product data — title, price, stock status — lives
here) and indexing job tracking.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from src.config.settings import Settings, get_settings
from src.domain.exceptions import StorageError
from src.domain.schemas import IndexingJob, IndexingJobStatus, Product, VectorBackend
from src.observability.logging import get_logger

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    product_id TEXT PRIMARY KEY,
    sku TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    brand TEXT,
    category TEXT,
    price REAL,
    attributes_json TEXT NOT NULL DEFAULT '{}',
    in_stock INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS indexing_jobs (
    job_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    products_total INTEGER NOT NULL DEFAULT 0,
    products_indexed INTEGER NOT NULL DEFAULT 0,
    backend TEXT NOT NULL,
    error TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_products_category ON products (category);
CREATE INDEX IF NOT EXISTS idx_products_brand ON products (brand);
"""


class SQLiteStorageService:
    """Connection-per-call SQLite storage with WAL mode."""

    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()
        self._db_path = self._settings.sqlite_path
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ----------------------------------------------------------------
    # Products
    # ----------------------------------------------------------------

    def save_product(self, product: Product) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO products
                        (product_id, sku, title, description, brand, category, price,
                         attributes_json, in_stock, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(product_id) DO UPDATE SET
                        sku=excluded.sku, title=excluded.title, description=excluded.description,
                        brand=excluded.brand, category=excluded.category, price=excluded.price,
                        attributes_json=excluded.attributes_json, in_stock=excluded.in_stock,
                        updated_at=excluded.updated_at
                    """,
                    (
                        product.product_id,
                        product.sku,
                        product.title,
                        product.description,
                        product.brand,
                        product.category,
                        product.price,
                        json.dumps(product.attributes),
                        int(product.in_stock),
                        product.created_at.isoformat(),
                        product.updated_at.isoformat(),
                    ),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to save product {product.product_id}: {exc}") from exc

    def save_products_batch(self, products: list[Product]) -> None:
        for product in products:
            self.save_product(product)

    def get_product(self, product_id: str) -> Optional[Product]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM products WHERE product_id = ?", (product_id,)).fetchone()
        return self._row_to_product(row) if row else None

    def get_products_by_ids(self, product_ids: list[str]) -> list[Product]:
        if not product_ids:
            return []
        placeholders = ",".join("?" for _ in product_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM products WHERE product_id IN ({placeholders})", product_ids
            ).fetchall()
        by_id = {row["product_id"]: self._row_to_product(row) for row in rows}
        # Preserve the order of the requested product_ids (important — callers pass
        # in rank order and expect that order preserved through hydration).
        return [by_id[pid] for pid in product_ids if pid in by_id]

    def list_all_products(self, limit: int = 100_000) -> list[Product]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM products LIMIT ?", (limit,)).fetchall()
        return [self._row_to_product(r) for r in rows]

    def delete_product(self, product_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM products WHERE product_id = ?", (product_id,))

    def count_products(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM products").fetchone()
        return row["c"]

    @staticmethod
    def _row_to_product(row: sqlite3.Row) -> Product:
        return Product(
            product_id=row["product_id"],
            sku=row["sku"],
            title=row["title"],
            description=row["description"],
            brand=row["brand"],
            category=row["category"],
            price=row["price"],
            attributes=json.loads(row["attributes_json"]),
            in_stock=bool(row["in_stock"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # ----------------------------------------------------------------
    # Indexing jobs
    # ----------------------------------------------------------------

    def save_indexing_job(self, job: IndexingJob) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO indexing_jobs
                        (job_id, status, products_total, products_indexed, backend, error, started_at, completed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(job_id) DO UPDATE SET
                        status=excluded.status, products_total=excluded.products_total,
                        products_indexed=excluded.products_indexed, error=excluded.error,
                        completed_at=excluded.completed_at
                    """,
                    (
                        job.job_id,
                        job.status.value,
                        job.products_total,
                        job.products_indexed,
                        job.backend.value,
                        job.error,
                        job.started_at.isoformat(),
                        job.completed_at.isoformat() if job.completed_at else None,
                    ),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"Failed to save indexing job {job.job_id}: {exc}") from exc

    def get_indexing_job(self, job_id: str) -> Optional[IndexingJob]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM indexing_jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return IndexingJob(
            job_id=row["job_id"],
            status=IndexingJobStatus(row["status"]),
            products_total=row["products_total"],
            products_indexed=row["products_indexed"],
            backend=VectorBackend(row["backend"]),
            error=row["error"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
        )
