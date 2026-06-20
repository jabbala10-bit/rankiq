"""Unit tests for src/services/storage/sqlite_service.py."""
from __future__ import annotations

import pytest

from src.domain.schemas import IndexingJob, IndexingJobStatus, VectorBackend
from src.services.storage.sqlite_service import SQLiteStorageService


@pytest.fixture
def storage(test_settings) -> SQLiteStorageService:
    return SQLiteStorageService(test_settings)


class TestProductPersistence:
    def test_save_and_get_roundtrip(self, storage, sample_product):
        storage.save_product(sample_product)
        fetched = storage.get_product(sample_product.product_id)
        assert fetched is not None
        assert fetched.title == sample_product.title
        assert fetched.attributes == sample_product.attributes

    def test_get_nonexistent_returns_none(self, storage):
        assert storage.get_product("missing") is None

    def test_upsert_updates_existing_product(self, storage, sample_product):
        storage.save_product(sample_product)
        updated = sample_product.model_copy(update={"price": 99.99})
        storage.save_product(updated)
        fetched = storage.get_product(sample_product.product_id)
        assert fetched.price == 99.99

    def test_save_products_batch(self, storage, sample_catalog):
        storage.save_products_batch(sample_catalog)
        assert storage.count_products() == len(sample_catalog)

    def test_get_products_by_ids_preserves_order(self, storage, sample_catalog):
        storage.save_products_batch(sample_catalog)
        ids = [p.product_id for p in sample_catalog]
        reversed_ids = list(reversed(ids))
        fetched = storage.get_products_by_ids(reversed_ids)
        assert [p.product_id for p in fetched] == reversed_ids

    def test_get_products_by_ids_skips_missing(self, storage, sample_catalog):
        storage.save_products_batch(sample_catalog)
        ids = [sample_catalog[0].product_id, "does-not-exist", sample_catalog[1].product_id]
        fetched = storage.get_products_by_ids(ids)
        assert len(fetched) == 2

    def test_delete_product(self, storage, sample_product):
        storage.save_product(sample_product)
        storage.delete_product(sample_product.product_id)
        assert storage.get_product(sample_product.product_id) is None

    def test_list_all_products(self, storage, sample_catalog):
        storage.save_products_batch(sample_catalog)
        all_products = storage.list_all_products()
        assert len(all_products) == len(sample_catalog)


class TestIndexingJobPersistence:
    def test_save_and_get_roundtrip(self, storage):
        job = IndexingJob(backend=VectorBackend.FAISS, products_total=10)
        storage.save_indexing_job(job)
        fetched = storage.get_indexing_job(job.job_id)
        assert fetched is not None
        assert fetched.products_total == 10
        assert fetched.status == IndexingJobStatus.PENDING

    def test_update_job_progress(self, storage):
        job = IndexingJob(backend=VectorBackend.QDRANT, products_total=100)
        storage.save_indexing_job(job)

        job.status = IndexingJobStatus.RUNNING
        job.products_indexed = 50
        storage.save_indexing_job(job)

        fetched = storage.get_indexing_job(job.job_id)
        assert fetched.status == IndexingJobStatus.RUNNING
        assert fetched.products_indexed == 50

    def test_get_nonexistent_job_returns_none(self, storage):
        assert storage.get_indexing_job("missing") is None
