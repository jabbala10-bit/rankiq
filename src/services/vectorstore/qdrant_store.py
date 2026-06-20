"""
Qdrant VectorStore implementation.

Qdrant is a purpose-built vector database server — unlike FAISS, it
handles persistence, filtering, and concurrent access natively, at the
cost of running a separate service. Deferred import of `qdrant_client`
so unit tests can inject a mock client without the real package
installed.
"""
from __future__ import annotations

from typing import Optional

from src.config.settings import Settings, get_settings
from src.domain.exceptions import VectorStoreError, VectorStoreUnavailableError
from src.domain.schemas import ProductEmbedding
from src.observability.logging import get_logger
from src.services.vectorstore.base import VectorSearchHit, VectorStore

logger = get_logger(__name__)


class QdrantVectorStore(VectorStore):
    """Qdrant-backed implementation using cosine distance."""

    def __init__(self, settings: Optional[Settings] = None, client: Optional[object] = None):
        self._settings = settings or get_settings()
        self._client = client  # injected in tests
        self._collection = self._settings.qdrant_collection
        self._ensured = client is not None  # if a client is injected, assume the test sets up the collection

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from qdrant_client import QdrantClient  # deferred import

            self._client = QdrantClient(url=self._settings.qdrant_url)
            return self._client
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreUnavailableError(f"Could not connect to Qdrant: {exc}") from exc

    def _ensure_collection(self) -> None:
        if self._ensured:
            return
        from qdrant_client.models import Distance, VectorParams

        client = self._get_client()
        existing = [c.name for c in client.get_collections().collections]
        if self._collection not in existing:
            client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._settings.embedding_dimensions, distance=Distance.COSINE),
            )
            logger.info("qdrant_collection_created", collection=self._collection)
        self._ensured = True

    @staticmethod
    def _point_id(product_id: str) -> str:
        """
        Qdrant accepts string or unsigned-int IDs natively, so unlike
        FAISS we can use the product_id directly without a hash — but we
        still funnel it through this helper for symmetry with the other
        backends and to keep the ID-derivation logic discoverable in one
        place if that ever needs to change.
        """
        return product_id

    def upsert(self, embeddings: list[ProductEmbedding]) -> None:
        if not embeddings:
            return
        self._ensure_collection()
        from qdrant_client.models import PointStruct

        client = self._get_client()
        try:
            points = [
                PointStruct(id=self._point_id(e.product_id), vector=e.vector, payload={"product_id": e.product_id})
                for e in embeddings
            ]
            client.upsert(collection_name=self._collection, points=points)
            logger.info("qdrant_upsert_complete", count=len(embeddings))
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Qdrant upsert failed: {exc}") from exc

    def search(self, query_vector: list[float], top_k: int) -> list[VectorSearchHit]:
        self._ensure_collection()
        client = self._get_client()
        try:
            results = client.search(collection_name=self._collection, query_vector=query_vector, limit=top_k)
            return [
                VectorSearchHit(product_id=hit.payload["product_id"], score=float(hit.score)) for hit in results
            ]
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Qdrant search failed: {exc}") from exc

    def delete(self, product_ids: list[str]) -> None:
        if not product_ids:
            return
        self._ensure_collection()
        client = self._get_client()
        try:
            client.delete(collection_name=self._collection, points_selector=[self._point_id(p) for p in product_ids])
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Qdrant delete failed: {exc}") from exc

    def count(self) -> int:
        self._ensure_collection()
        client = self._get_client()
        try:
            info = client.get_collection(collection_name=self._collection)
            return int(info.points_count or 0)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Qdrant count failed: {exc}") from exc

    def clear(self) -> None:
        client = self._get_client()
        try:
            client.delete_collection(collection_name=self._collection)
        except Exception:  # noqa: BLE001
            pass  # collection may not exist yet; that's fine for clear()
        self._ensured = False
        self._ensure_collection()
