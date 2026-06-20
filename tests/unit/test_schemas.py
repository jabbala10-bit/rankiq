"""Unit tests for src/domain/schemas.py."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.domain.schemas import IndexingJob, IndexingJobStatus, Product, ProductEmbedding, VectorBackend


class TestProduct:
    def test_valid_product_is_created(self):
        p = Product(sku="SKU1", title="Test Product")
        assert p.product_id
        assert p.in_stock is True

    def test_title_required_and_bounded(self):
        with pytest.raises(ValidationError):
            Product(sku="SKU1", title="")
        with pytest.raises(ValidationError):
            Product(sku="SKU1", title="x" * 301)

    def test_price_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            Product(sku="SKU1", title="Test", price=-1.0)

    def test_searchable_text_includes_all_relevant_fields(self):
        p = Product(
            sku="SKU1", title="Hiking Jacket", brand="TrailPeak", category="Outerwear",
            description="Waterproof shell", attributes={"color": "green"},
        )
        text = p.searchable_text
        assert "Hiking Jacket" in text
        assert "TrailPeak" in text
        assert "Outerwear" in text
        assert "Waterproof shell" in text
        assert "green" in text

    def test_searchable_text_handles_missing_optional_fields(self):
        p = Product(sku="SKU1", title="Plain Product")
        assert p.searchable_text == "Plain Product"


class TestProductEmbedding:
    def test_empty_vector_rejected(self):
        with pytest.raises(ValidationError):
            ProductEmbedding(product_id="p1", vector=[], model_name="test", dimensions=0)

    def test_valid_embedding(self):
        emb = ProductEmbedding(product_id="p1", vector=[0.1, 0.2, 0.3], model_name="test", dimensions=3)
        assert len(emb.vector) == 3


class TestIndexingJob:
    def test_progress_pct_zero_when_no_products(self):
        job = IndexingJob(backend=VectorBackend.FAISS)
        assert job.progress_pct == 0.0

    def test_progress_pct_computed_correctly(self):
        job = IndexingJob(backend=VectorBackend.FAISS, products_total=200, products_indexed=50)
        assert job.progress_pct == 25.0

    def test_default_status_is_pending(self):
        job = IndexingJob(backend=VectorBackend.QDRANT)
        assert job.status == IndexingJobStatus.PENDING
