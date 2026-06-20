"""Shared pytest fixtures for RankIQ tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.config.settings import Settings
from src.domain.schemas import Product


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> str:
    return str(tmp_path / "test_rankiq.db")


@pytest.fixture
def test_settings(tmp_path: Path, tmp_db_path: str) -> Settings:
    return Settings(
        environment="development",
        sqlite_path=tmp_db_path,
        faiss_index_dir=str(tmp_path / "faiss_index"),
        catalog_dir=str(tmp_path / "catalog"),
        embedding_dimensions=8,  # small dims for fast fake-embedding tests
        vector_backend="faiss",
    )


@pytest.fixture
def sample_product() -> Product:
    return Product(
        sku="JKT-001",
        title="Men's Waterproof Hiking Jacket",
        description="Lightweight, breathable, fully waterproof shell jacket for hiking.",
        brand="TrailPeak",
        category="Outerwear",
        price=129.99,
        attributes={"color": "forest green"},
    )


@pytest.fixture
def sample_catalog() -> list[Product]:
    return [
        Product(
            sku="JKT-001", title="Men's Waterproof Hiking Jacket",
            description="Lightweight, breathable, fully waterproof shell jacket for hiking.",
            brand="TrailPeak", category="Outerwear", price=129.99,
        ),
        Product(
            sku="JKT-002", title="Women's Insulated Winter Parka",
            description="Warm down-insulated parka rated for sub-zero temperatures.",
            brand="Northgale", category="Outerwear", price=199.99,
        ),
        Product(
            sku="SHO-010", title="Trail Running Shoes",
            description="Grippy outsole, breathable mesh upper, ideal for muddy or wet trails.",
            brand="TrailPeak", category="Footwear", price=89.99,
        ),
        Product(
            sku="SHO-011", title="Leather Dress Shoes",
            description="Classic formal leather oxford shoes for business attire.",
            brand="Camden & Co", category="Footwear", price=149.99,
        ),
        Product(
            sku="BAG-005", title="Waterproof Hiking Backpack 40L",
            description="Durable waterproof backpack with hydration sleeve, ideal for multi-day hikes.",
            brand="TrailPeak", category="Bags", price=109.99,
        ),
    ]


def make_fake_embedding(text: str, dimensions: int = 8) -> list[float]:
    """
    Deterministic fake embedding for tests: hashes the text into a fixed-
    length float vector. Not semantically meaningful, but deterministic
    and stable — enough for testing plumbing (upsert/search/delete
    roundtrips) without needing a real sentence-transformers model.
    """
    import hashlib

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = [b / 255.0 for b in digest[:dimensions]]
    norm = sum(x**2 for x in raw) ** 0.5
    return [x / norm for x in raw] if norm > 0 else raw
