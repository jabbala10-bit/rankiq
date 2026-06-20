"""
Cross-encoder reranking service — the optional final stage that re-scores
the top fused candidates by jointly encoding (query, product_text) pairs,
rather than comparing independently-computed embeddings (ADR-004).

Deferred import of sentence_transformers' CrossEncoder, same pattern as
every other heavy ML dependency in this portfolio.
"""
from __future__ import annotations

import time
from typing import Optional

from src.config.settings import Settings, get_settings
from src.domain.exceptions import ModelNotLoadedError, RerankingError
from src.domain.schemas import Product
from src.observability.logging import get_logger
from src.observability.metrics import RERANK_REQUESTS_TOTAL

logger = get_logger(__name__)


class CrossEncoderReranker:
    """Wraps a sentence-transformers CrossEncoder for query-product relevance scoring."""

    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()
        self._model = None  # lazily loaded; type is sentence_transformers.CrossEncoder

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder  # deferred import

            logger.info("loading_reranker_model", model=self._settings.reranker_model)
            self._model = CrossEncoder(self._settings.reranker_model)
        except Exception as exc:  # noqa: BLE001
            raise RerankingError(f"Failed to load reranker model: {exc}") from exc

    def rerank(self, query_text: str, products: list[Product]) -> list[tuple[Product, float]]:
        """
        Scores each (query, product) pair jointly and returns
        (product, rerank_score) tuples sorted descending by score.

        Raises:
            ModelNotLoadedError: if load_model() was never called.
            RerankingError: if scoring fails.
        """
        if self._model is None:
            raise ModelNotLoadedError("Reranker model not loaded — call load_model() first.")
        if not products:
            return []

        start = time.monotonic()
        try:
            pairs = [(query_text, p.searchable_text) for p in products]
            scores = self._model.predict(pairs)

            ranked = sorted(zip(products, scores), key=lambda item: item[1], reverse=True)
            RERANK_REQUESTS_TOTAL.labels(status="success").inc()
            logger.info(
                "rerank_complete",
                candidate_count=len(products),
                duration_seconds=round(time.monotonic() - start, 3),
            )
            return [(product, float(score)) for product, score in ranked]
        except Exception as exc:  # noqa: BLE001
            RERANK_REQUESTS_TOTAL.labels(status="error").inc()
            raise RerankingError(f"Reranking failed: {exc}") from exc
