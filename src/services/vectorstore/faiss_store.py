"""
FAISS VectorStore implementation.

FAISS is an in-process library, not a server — there's no daemon to
connect to, which makes it the simplest backend to run locally (ADR-001).
The tradeoff: FAISS itself has no native metadata/filtering or
persistence story, so this implementation maintains its own
product_id<->internal-index-position mapping and handles save/load to
disk explicitly.

Deferred import of `faiss` (same pattern as every other heavy native
dependency in this portfolio) so unit tests can run without it installed,
using a fake in-memory index injected via the constructor.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.config.settings import Settings, get_settings
from src.domain.exceptions import DimensionMismatchError, VectorStoreError
from src.domain.schemas import ProductEmbedding
from src.observability.logging import get_logger
from src.services.vectorstore.base import VectorSearchHit, VectorStore

logger = get_logger(__name__)


class FAISSVectorStore(VectorStore):
    """
    FAISS-backed implementation using a flat inner-product index wrapped
    in IndexIDMap2 (so FAISS's internal integer IDs can be set explicitly,
    letting us use a stable hash of product_id as the FAISS ID rather
    than relying on insertion order).
    """

    def __init__(self, settings: Optional[Settings] = None, index: Optional[object] = None):
        self._settings = settings or get_settings()
        self._dimensions = self._settings.embedding_dimensions
        self._index = index  # injected in tests; lazily created otherwise
        self._id_to_product: dict[int, str] = {}
        self._product_to_id: dict[str, int] = {}

    def _get_index(self):
        if self._index is not None:
            return self._index
        import faiss  # deferred import

        flat_index = faiss.IndexFlatIP(self._dimensions)  # inner product on normalized vectors = cosine sim
        self._index = faiss.IndexIDMap2(flat_index)
        return self._index

    @staticmethod
    def _stable_id(product_id: str) -> int:
        """FAISS IDs must be int64 — derive a stable one from the product_id string."""
        import hashlib

        digest = hashlib.sha256(product_id.encode("utf-8")).hexdigest()
        return int(digest[:15], 16)  # fits comfortably within int64 range

    def upsert(self, embeddings: list[ProductEmbedding]) -> None:
        if not embeddings:
            return
        import numpy as np

        index = self._get_index()
        for emb in embeddings:
            if emb.dimensions != self._dimensions:
                raise DimensionMismatchError(
                    f"Embedding for {emb.product_id} has {emb.dimensions} dims, "
                    f"index expects {self._dimensions}."
                )

        try:
            # FAISS has no in-place update — remove any existing IDs for these
            # products first, then add fresh, so upsert is idempotent.
            existing_ids = [
                self._product_to_id[e.product_id] for e in embeddings if e.product_id in self._product_to_id
            ]
            if existing_ids:
                index.remove_ids(np.array(existing_ids, dtype="int64"))

            vectors = np.array([e.vector for e in embeddings], dtype="float32")
            ids = np.array([self._stable_id(e.product_id) for e in embeddings], dtype="int64")
            index.add_with_ids(vectors, ids)

            for emb, faiss_id in zip(embeddings, ids):
                self._id_to_product[int(faiss_id)] = emb.product_id
                self._product_to_id[emb.product_id] = int(faiss_id)

            logger.info("faiss_upsert_complete", count=len(embeddings))
        except DimensionMismatchError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"FAISS upsert failed: {exc}") from exc

    def search(self, query_vector: list[float], top_k: int) -> list[VectorSearchHit]:
        import numpy as np

        index = self._get_index()
        if index.ntotal == 0:
            return []

        try:
            query = np.array([query_vector], dtype="float32")
            scores, ids = index.search(query, min(top_k, index.ntotal))

            hits: list[VectorSearchHit] = []
            for score, faiss_id in zip(scores[0], ids[0]):
                if faiss_id == -1:
                    continue
                product_id = self._id_to_product.get(int(faiss_id))
                if product_id is None:
                    continue
                hits.append(VectorSearchHit(product_id=product_id, score=float(score)))
            return hits
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"FAISS search failed: {exc}") from exc

    def delete(self, product_ids: list[str]) -> None:
        if not product_ids:
            return
        import numpy as np

        index = self._get_index()
        ids_to_remove = [self._product_to_id[pid] for pid in product_ids if pid in self._product_to_id]
        if not ids_to_remove:
            return
        try:
            index.remove_ids(np.array(ids_to_remove, dtype="int64"))
            for pid in product_ids:
                faiss_id = self._product_to_id.pop(pid, None)
                if faiss_id is not None:
                    self._id_to_product.pop(faiss_id, None)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"FAISS delete failed: {exc}") from exc

    def count(self) -> int:
        index = self._get_index()
        return int(index.ntotal)

    def clear(self) -> None:
        import faiss

        flat_index = faiss.IndexFlatIP(self._dimensions)
        self._index = faiss.IndexIDMap2(flat_index)
        self._id_to_product.clear()
        self._product_to_id.clear()

    # ----------------------------------------------------------------
    # Persistence (FAISS-specific; not part of the abstract interface
    # since not every backend needs explicit save/load — Qdrant and
    # pgvector persist automatically via their own servers)
    # ----------------------------------------------------------------

    def save(self, directory: Optional[str] = None) -> None:
        import faiss

        dir_path = Path(directory or self._settings.faiss_index_dir)
        dir_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._get_index(), str(dir_path / "index.faiss"))
        with open(dir_path / "id_map.json", "w") as f:
            json.dump(self._id_to_product, f)
        logger.info("faiss_index_saved", path=str(dir_path))

    def load(self, directory: Optional[str] = None) -> None:
        import faiss

        dir_path = Path(directory or self._settings.faiss_index_dir)
        index_path = dir_path / "id_map.json"
        if not index_path.exists():
            raise VectorStoreError(f"No saved FAISS index found at {dir_path}")

        self._index = faiss.read_index(str(dir_path / "index.faiss"))
        with open(index_path) as f:
            raw_map = json.load(f)
        self._id_to_product = {int(k): v for k, v in raw_map.items()}
        self._product_to_id = {v: k for k, v in self._id_to_product.items()}
        logger.info("faiss_index_loaded", path=str(dir_path), count=self.count())
