"""
Catalog indexing pipeline: ingests products, generates embeddings, and
dual-writes to both the BM25 keyword index and the configured
VectorStore backend. This is the write path that the retrieval service's
read path (hybrid search) depends on.
"""
from __future__ import annotations

from typing import Optional

from src.domain.constants import INDEXING_BATCH_SIZE
from src.domain.exceptions import IndexingError, RankIQError
from src.domain.schemas import IndexingJob, IndexingJobStatus, Product, VectorBackend
from src.observability.logging import get_logger
from src.observability.metrics import CATALOG_SIZE, INDEXING_JOBS_TOTAL, PRODUCTS_INDEXED_TOTAL
from src.services.embedding.embedding_service import EmbeddingService
from src.services.retrieval.bm25_index import BM25KeywordIndex
from src.services.storage.sqlite_service import SQLiteStorageService
from src.services.vectorstore.base import VectorStore

logger = get_logger(__name__)


class IndexingPipeline:
    """Orchestrates: persist products -> embed -> upsert vector store -> rebuild BM25 index."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: Optional[EmbeddingService] = None,
        bm25_index: Optional[BM25KeywordIndex] = None,
        storage: Optional[SQLiteStorageService] = None,
        vector_backend: VectorBackend = VectorBackend.FAISS,
    ):
        self._vector_store = vector_store
        self._embedding = embedding_service or EmbeddingService()
        self._bm25 = bm25_index or BM25KeywordIndex()
        self._storage = storage or SQLiteStorageService()
        self._vector_backend = vector_backend

    def index_catalog(self, products: list[Product]) -> IndexingJob:
        """
        Full (re)indexing of a product list: persists each product,
        generates embeddings in batches, upserts the vector store, and
        rebuilds the BM25 index over the complete set.
        """
        job = IndexingJob(products_total=len(products), backend=self._vector_backend)
        self._storage.save_indexing_job(job)
        job.status = IndexingJobStatus.RUNNING
        self._storage.save_indexing_job(job)

        try:
            indexed_count = 0
            for batch_start in range(0, len(products), INDEXING_BATCH_SIZE):
                batch = products[batch_start : batch_start + INDEXING_BATCH_SIZE]
                self._storage.save_products_batch(batch)

                embeddings = self._embedding.embed_batch(batch)
                self._vector_store.upsert(embeddings)

                indexed_count += len(batch)
                job.products_indexed = indexed_count
                self._storage.save_indexing_job(job)
                PRODUCTS_INDEXED_TOTAL.inc(len(batch))

            # BM25 needs the full corpus for accurate IDF statistics, so
            # it's rebuilt once at the end rather than incrementally per batch.
            all_products = self._storage.list_all_products()
            self._bm25.build(all_products)

            job.status = IndexingJobStatus.COMPLETED
            from datetime import datetime, timezone

            job.completed_at = datetime.now(timezone.utc)
            self._storage.save_indexing_job(job)

            INDEXING_JOBS_TOTAL.labels(status="success").inc()
            CATALOG_SIZE.set(self._storage.count_products())
            logger.info("indexing_job_complete", job_id=job.job_id, products_indexed=indexed_count)
            return job

        except RankIQError as exc:
            job.status = IndexingJobStatus.FAILED
            job.error = str(exc)
            self._storage.save_indexing_job(job)
            INDEXING_JOBS_TOTAL.labels(status="error").inc()
            logger.error("indexing_job_failed", job_id=job.job_id, error=str(exc))
            raise IndexingError(f"Indexing job {job.job_id} failed: {exc}") from exc

    def upsert_product(self, product: Product) -> None:
        """Incremental single-product upsert — vector store update is cheap; BM25 requires a rebuild."""
        self._storage.save_product(product)
        embedding = self._embedding.embed_product(product)
        self._vector_store.upsert([embedding])
        self._bm25.upsert([product])
        CATALOG_SIZE.set(self._storage.count_products())

    def delete_product(self, product_id: str) -> None:
        self._storage.delete_product(product_id)
        self._vector_store.delete([product_id])
        self._bm25.delete([product_id])
        CATALOG_SIZE.set(self._storage.count_products())
