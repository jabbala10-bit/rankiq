"""
Embedding service: wraps sentence-transformers to convert product text
into dense vectors.

Same deferred-import pattern used throughout this portfolio (faster-whisper
in FieldOpsIQ, confluent-kafka in StreamGuardIQ): the real
SentenceTransformer import happens inside `load_model()`, so unit tests
can construct this class and inject a mock encoder without requiring the
(sizeable) sentence-transformers + torch dependency stack to be installed.
"""
from __future__ import annotations

import time
from typing import Optional

from src.config.settings import Settings, get_settings
from src.domain.exceptions import EmbeddingError, ModelNotLoadedError
from src.domain.schemas import Product, ProductEmbedding
from src.observability.logging import get_logger
from src.observability.metrics import EMBEDDING_DURATION_SECONDS, EMBEDDING_REQUESTS_TOTAL

logger = get_logger(__name__)


class EmbeddingService:
    """Generates dense vector embeddings for product text."""

    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()
        self._model = None  # lazily loaded; type is sentence_transformers.SentenceTransformer

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_model(self) -> None:
        """
        Loads the sentence-transformer model into memory. Call once at
        startup (analogous to FieldOpsIQ's eager Whisper-model load).

        Raises:
            EmbeddingError: if the model fails to load.
        """
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # deferred import

            logger.info("loading_embedding_model", model=self._settings.embedding_model)
            self._model = SentenceTransformer(self._settings.embedding_model)
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"Failed to load embedding model: {exc}") from exc

    def embed_product(self, product: Product) -> ProductEmbedding:
        """Embeds a single product's searchable_text."""
        return self.embed_text(product.product_id, product.searchable_text)

    def embed_text(self, identifier: str, text: str) -> ProductEmbedding:
        """
        Embeds arbitrary text (used for both product indexing and query
        embedding — identical code path guarantees the query vector and
        the indexed vectors live in the same embedding space).

        Raises:
            ModelNotLoadedError: if load_model() was never called.
            EmbeddingError: if encoding fails.
        """
        if self._model is None:
            raise ModelNotLoadedError("Embedding model not loaded — call load_model() first.")

        start = time.monotonic()
        try:
            vector = self._model.encode(text, normalize_embeddings=True).tolist()
            elapsed = time.monotonic() - start
            EMBEDDING_DURATION_SECONDS.observe(elapsed)
            EMBEDDING_REQUESTS_TOTAL.labels(status="success").inc()
            return ProductEmbedding(
                product_id=identifier,
                vector=vector,
                model_name=self._settings.embedding_model,
                dimensions=len(vector),
            )
        except Exception as exc:  # noqa: BLE001
            EMBEDDING_REQUESTS_TOTAL.labels(status="error").inc()
            raise EmbeddingError(f"Failed to embed text for '{identifier}': {exc}") from exc

    def embed_batch(self, products: list[Product]) -> list[ProductEmbedding]:
        """
        Embeds a batch of products in one model call — substantially
        faster than embedding one at a time due to GPU/CPU batching in
        the underlying sentence-transformers encode() call.
        """
        if self._model is None:
            raise ModelNotLoadedError("Embedding model not loaded — call load_model() first.")
        if not products:
            return []

        start = time.monotonic()
        try:
            texts = [p.searchable_text for p in products]
            vectors = self._model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=min(len(texts), 64),
            )
            elapsed = time.monotonic() - start
            EMBEDDING_DURATION_SECONDS.observe(elapsed)
            EMBEDDING_REQUESTS_TOTAL.labels(status="success").inc()

            return [
                ProductEmbedding(
                    product_id=product.product_id,
                    vector=vector.tolist(),
                    model_name=self._settings.embedding_model,
                    dimensions=len(vector),
                )
                for product, vector in zip(products, vectors)
            ]
        except Exception as exc:  # noqa: BLE001
            EMBEDDING_REQUESTS_TOTAL.labels(status="error").inc()
            raise EmbeddingError(f"Failed to embed batch of {len(products)} products: {exc}") from exc
