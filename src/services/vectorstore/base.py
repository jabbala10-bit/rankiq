"""
VectorStore interface — the strategy-pattern abstraction at the heart of
ADR-001. Every backend (FAISS, Qdrant, pgvector) implements this exact
interface, so `RetrievalService` and the indexing pipeline never know or
care which backend is active. Swapping backends is a one-line config
change (`VECTOR_BACKEND` env var), not a code change anywhere else.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from src.domain.schemas import Product, ProductEmbedding


class VectorSearchHit(dict):
    """
    Lightweight result type returned by `search()`: a dict-like object
    with `product_id` and `score` keys. Kept as a plain dict subclass
    (rather than a full Pydantic model) since this is an internal,
    backend-facing type that never crosses the API boundary directly —
    `RetrievalService` translates these into domain `ScoredProduct`
    objects before anything external sees them.
    """

    @property
    def product_id(self) -> str:
        return self["product_id"]

    @property
    def score(self) -> float:
        return self["score"]


class VectorStore(ABC):
    """Abstract interface every vector backend implementation must satisfy."""

    @abstractmethod
    def upsert(self, embeddings: list[ProductEmbedding]) -> None:
        """Inserts or updates vectors for the given product embeddings."""
        raise NotImplementedError

    @abstractmethod
    def search(self, query_vector: list[float], top_k: int) -> list[VectorSearchHit]:
        """
        Returns the top_k nearest neighbors to query_vector, each as a
        VectorSearchHit with `product_id` and a similarity `score`
        (higher = more similar, normalized to a comparable scale across
        backends — see each implementation's docstring for specifics).
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, product_ids: list[str]) -> None:
        """Removes vectors for the given product IDs, if present."""
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """Returns the current number of vectors stored."""
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Removes all vectors — used by tests and full-reindex operations."""
        raise NotImplementedError
