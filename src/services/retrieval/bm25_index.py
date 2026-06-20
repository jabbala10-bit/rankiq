"""
BM25 keyword index — the "exact-match-aware" half of the hybrid retrieval
design (ADR-002). Pure-Python BM25 implementation (via the `rank_bm25`
package) rather than a separate search-engine server (Elasticsearch/
OpenSearch), since at the scale this case study demonstrates, an
in-process BM25 index is simpler to run and reason about — see ADR-002
for the full tradeoff discussion and when Elasticsearch would be the
better choice.
"""
from __future__ import annotations

from typing import Optional

from src.domain.exceptions import KeywordIndexError
from src.domain.schemas import Product
from src.observability.logging import get_logger

logger = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple lowercase whitespace/punctuation tokenizer, shared by index build and query time."""
    import re

    return re.findall(r"[a-z0-9]+", text.lower())


class BM25KeywordIndex:
    """In-process BM25 index over the product catalog's searchable_text."""

    def __init__(self):
        self._bm25 = None  # lazily built; type is rank_bm25.BM25Okapi
        self._product_ids: list[str] = []
        self._products_by_id: dict[str, Product] = {}

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None

    def build(self, products: list[Product]) -> None:
        """
        Builds (or rebuilds) the BM25 index from scratch over the given
        products. BM25's IDF statistics depend on the full corpus, so
        unlike the vector store there's no meaningful per-document
        incremental upsert — adding/removing documents requires a full
        rebuild, which `upsert()`/`delete()` below handle transparently.
        """
        try:
            from rank_bm25 import BM25Okapi  # deferred import

            self._products_by_id = {p.product_id: p for p in products}
            self._product_ids = list(self._products_by_id.keys())
            tokenized_corpus = [_tokenize(p.searchable_text) for p in products]

            if tokenized_corpus:
                self._bm25 = BM25Okapi(tokenized_corpus)
            else:
                self._bm25 = None
            logger.info("bm25_index_built", product_count=len(products))
        except Exception as exc:  # noqa: BLE001
            raise KeywordIndexError(f"Failed to build BM25 index: {exc}") from exc

    def upsert(self, products: list[Product]) -> None:
        """Adds/updates products and rebuilds the index (see build() docstring on why)."""
        merged = dict(self._products_by_id)
        for p in products:
            merged[p.product_id] = p
        self.build(list(merged.values()))

    def delete(self, product_ids: list[str]) -> None:
        remaining = [p for pid, p in self._products_by_id.items() if pid not in product_ids]
        self.build(remaining)

    def search(self, query_text: str, top_k: int) -> list[tuple[str, float]]:
        """
        Returns up to top_k (product_id, bm25_score) tuples, sorted
        descending by score. Returns an empty list if the index hasn't
        been built yet or the corpus is empty, rather than raising — an
        empty keyword index is a valid (if degenerate) state, not an
        error condition.
        """
        if self._bm25 is None:
            return []

        try:
            import numpy as np

            tokenized_query = _tokenize(query_text)
            scores = self._bm25.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[::-1][:top_k]
            return [
                (self._product_ids[i], float(scores[i]))
                for i in top_indices
                if scores[i] > 0  # exclude zero-score (no term overlap) results
            ]
        except Exception as exc:  # noqa: BLE001
            raise KeywordIndexError(f"BM25 search failed: {exc}") from exc

    def count(self) -> int:
        return len(self._product_ids)

    def clear(self) -> None:
        self._bm25 = None
        self._product_ids = []
        self._products_by_id = {}
