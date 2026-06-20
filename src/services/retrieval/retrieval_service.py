"""
Retrieval service: the read-path orchestrator for hybrid search.

Sequence: embed query -> vector search (top N candidates) + BM25 search
(top N candidates) -> fuse -> hydrate Product objects -> optional
cross-encoder rerank -> return ranked ScoredProduct list.

Like FieldOpsIQ's and StreamGuardIQ's pipeline orchestrators, this is a
plain class with a linear sequence — no branching agent behavior, so no
LangGraph-style state machine is warranted here either.
"""
from __future__ import annotations

import time
from typing import Optional

from src.domain.constants import DEFAULT_KEYWORD_CANDIDATES, DEFAULT_RERANK_CANDIDATES, DEFAULT_VECTOR_CANDIDATES
from src.domain.exceptions import RetrievalError
from src.domain.schemas import (
    FusionStrategy,
    RetrievalSource,
    ScoredProduct,
    SearchQuery,
    SearchResult,
)
from src.observability.logging import get_logger
from src.observability.metrics import SEARCH_LATENCY_SECONDS, SEARCH_REQUESTS_TOTAL
from src.services.embedding.embedding_service import EmbeddingService
from src.services.retrieval.bm25_index import BM25KeywordIndex
from src.services.retrieval.fusion import fuse_rrf, fuse_weighted_sum
from src.services.retrieval.reranker import CrossEncoderReranker
from src.services.storage.sqlite_service import SQLiteStorageService
from src.services.vectorstore.base import VectorStore

logger = get_logger(__name__)


class RetrievalService:
    """Orchestrates hybrid vector + keyword retrieval, fusion, and optional reranking."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25KeywordIndex,
        embedding_service: Optional[EmbeddingService] = None,
        storage: Optional[SQLiteStorageService] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4,
    ):
        self._vector_store = vector_store
        self._bm25 = bm25_index
        self._embedding = embedding_service or EmbeddingService()
        self._storage = storage or SQLiteStorageService()
        self._reranker = reranker
        self._vector_weight = vector_weight
        self._keyword_weight = keyword_weight

    def search(self, query: SearchQuery) -> SearchResult:
        """
        Executes a hybrid search and returns a ranked SearchResult.

        Raises:
            RetrievalError: on unexpected failures anywhere in the pipeline.
        """
        total_start = time.monotonic()
        try:
            vector_hits, keyword_hits = self._retrieve_candidates(query)

            fused_ids = self._fuse(query.fusion_strategy, vector_hits, keyword_hits)
            vector_rank_by_id = {hit.product_id: i + 1 for i, hit in enumerate(vector_hits)}
            keyword_rank_by_id = {pid: i + 1 for i, (pid, _) in enumerate(keyword_hits)}

            candidate_ids = [pid for pid, _ in fused_ids][: max(query.top_k, DEFAULT_RERANK_CANDIDATES)]
            products_by_id = {p.product_id: p for p in self._storage.get_products_by_ids(candidate_ids)}

            scored: list[ScoredProduct] = []
            for pid, fused_score in fused_ids:
                product = products_by_id.get(pid)
                if product is None:
                    continue  # vector/BM25 index references a product no longer in the catalog
                if query.filters and not self._matches_filters(product, query.filters):
                    continue
                scored.append(
                    ScoredProduct(
                        product=product,
                        score=fused_score,
                        source=RetrievalSource.FUSED,
                        vector_rank=vector_rank_by_id.get(pid),
                        keyword_rank=keyword_rank_by_id.get(pid),
                    )
                )
                if len(scored) >= max(query.top_k, DEFAULT_RERANK_CANDIDATES):
                    break

            if query.rerank and self._reranker is not None and scored:
                scored = self._apply_rerank(query.query_text, scored)

            final_results = scored[: query.top_k]
            total_elapsed_ms = (time.monotonic() - total_start) * 1000
            SEARCH_LATENCY_SECONDS.labels(stage="total").observe(total_elapsed_ms / 1000)
            SEARCH_REQUESTS_TOTAL.labels(fusion_strategy=query.fusion_strategy.value, status="success").inc()

            return SearchResult(
                query=query,
                results=final_results,
                vector_candidate_count=len(vector_hits),
                keyword_candidate_count=len(keyword_hits),
                total_latency_ms=round(total_elapsed_ms, 2),
            )

        except Exception as exc:  # noqa: BLE001
            SEARCH_REQUESTS_TOTAL.labels(fusion_strategy=query.fusion_strategy.value, status="error").inc()
            raise RetrievalError(f"Search failed for query '{query.query_text}': {exc}") from exc

    def _retrieve_candidates(self, query: SearchQuery):
        vector_hits = []
        keyword_hits = []

        if query.fusion_strategy != FusionStrategy.KEYWORD_ONLY:
            start = time.monotonic()
            query_embedding = self._embedding.embed_text("query", query.query_text)
            vector_hits = self._vector_store.search(query_embedding.vector, DEFAULT_VECTOR_CANDIDATES)
            SEARCH_LATENCY_SECONDS.labels(stage="vector").observe(time.monotonic() - start)

        if query.fusion_strategy != FusionStrategy.VECTOR_ONLY:
            start = time.monotonic()
            keyword_hits = self._bm25.search(query.query_text, DEFAULT_KEYWORD_CANDIDATES)
            SEARCH_LATENCY_SECONDS.labels(stage="keyword").observe(time.monotonic() - start)

        return vector_hits, keyword_hits

    def _fuse(self, strategy: FusionStrategy, vector_hits, keyword_hits) -> list[tuple[str, float]]:
        start = time.monotonic()
        try:
            if strategy == FusionStrategy.VECTOR_ONLY:
                result = [(hit.product_id, hit.score) for hit in vector_hits]
            elif strategy == FusionStrategy.KEYWORD_ONLY:
                result = keyword_hits
            elif strategy == FusionStrategy.RRF:
                vector_ids = [hit.product_id for hit in vector_hits]
                keyword_ids = [pid for pid, _ in keyword_hits]
                result = fuse_rrf(vector_ids, keyword_ids)
            else:  # WEIGHTED_SUM
                vector_scores = {hit.product_id: hit.score for hit in vector_hits}
                keyword_scores = dict(keyword_hits)
                result = fuse_weighted_sum(vector_scores, keyword_scores, self._vector_weight, self._keyword_weight)
            return result
        finally:
            SEARCH_LATENCY_SECONDS.labels(stage="fusion").observe(time.monotonic() - start)

    def _apply_rerank(self, query_text: str, scored: list[ScoredProduct]) -> list[ScoredProduct]:
        start = time.monotonic()
        candidates = scored[:DEFAULT_RERANK_CANDIDATES]
        products = [sp.product for sp in candidates]
        reranked = self._reranker.rerank(query_text, products)

        rerank_score_by_id = {p.product_id: score for p, score in reranked}
        updated = [
            sp.model_copy(
                update={
                    "rerank_score": rerank_score_by_id.get(sp.product.product_id),
                    "source": RetrievalSource.RERANKED,
                }
            )
            for sp in candidates
        ]
        updated.sort(key=lambda sp: sp.rerank_score if sp.rerank_score is not None else float("-inf"), reverse=True)
        SEARCH_LATENCY_SECONDS.labels(stage="rerank").observe(time.monotonic() - start)
        return updated + scored[DEFAULT_RERANK_CANDIDATES:]

    @staticmethod
    def _matches_filters(product, filters: dict[str, str]) -> bool:
        for key, value in filters.items():
            if key == "category" and product.category != value:
                return False
            if key == "brand" and product.brand != value:
                return False
            if key == "in_stock" and str(product.in_stock).lower() != value.lower():
                return False
        return True
