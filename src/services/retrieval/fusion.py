"""
Fusion strategies: combine vector-search and BM25-keyword-search
candidate lists into one ranked list (ADR-002).

Both RRF and weighted-sum are implemented as pure functions operating on
plain (product_id, score, rank) data — no dependency on the VectorStore
or BM25 index classes themselves, which makes fusion logic trivially
unit-testable in isolation from any backend.
"""
from __future__ import annotations

from src.domain.constants import RRF_K
from src.domain.exceptions import FusionError


def fuse_rrf(
    vector_ranked_ids: list[str],
    keyword_ranked_ids: list[str],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion (Cormack, Clarke & Buettcher 2009): each
    product's fused score is the sum of 1/(k + rank) across every list
    it appears in (rank is 1-indexed). RRF is chosen as the default
    fusion strategy because it requires no score normalization between
    vector similarity and BM25 scores — which live on entirely different,
    incomparable scales — since it only ever looks at *rank position*,
    not the raw scores themselves.

    Returns a list of (product_id, fused_score) tuples, sorted descending
    by fused_score.
    """
    try:
        scores: dict[str, float] = {}
        for rank, product_id in enumerate(vector_ranked_ids, start=1):
            scores[product_id] = scores.get(product_id, 0.0) + 1.0 / (k + rank)
        for rank, product_id in enumerate(keyword_ranked_ids, start=1):
            scores[product_id] = scores.get(product_id, 0.0) + 1.0 / (k + rank)

        return sorted(scores.items(), key=lambda item: item[1], reverse=True)
    except Exception as exc:  # noqa: BLE001
        raise FusionError(f"RRF fusion failed: {exc}") from exc


def fuse_weighted_sum(
    vector_scores: dict[str, float],
    keyword_scores: dict[str, float],
    vector_weight: float,
    keyword_weight: float,
) -> list[tuple[str, float]]:
    """
    Weighted-sum fusion: normalizes each score list to [0, 1] via min-max
    scaling (since vector similarity and BM25 scores are on incomparable
    raw scales), then combines as `vector_weight * norm_vector_score +
    keyword_weight * norm_keyword_score`.

    This is offered as an alternative to RRF for deployments that want
    explicit, tunable control over how much each signal contributes,
    at the cost of needing to choose weights and accept that min-max
    normalization is sensitive to outlier scores in either list — see
    ADR-002 for the full tradeoff discussion.
    """
    try:
        norm_vector = _min_max_normalize(vector_scores)
        norm_keyword = _min_max_normalize(keyword_scores)

        all_ids = set(norm_vector) | set(norm_keyword)
        fused = {
            pid: vector_weight * norm_vector.get(pid, 0.0) + keyword_weight * norm_keyword.get(pid, 0.0)
            for pid in all_ids
        }
        return sorted(fused.items(), key=lambda item: item[1], reverse=True)
    except Exception as exc:  # noqa: BLE001
        raise FusionError(f"Weighted-sum fusion failed: {exc}") from exc


def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi == lo:
        return {pid: 1.0 for pid in scores}  # all equal — avoid divide-by-zero, treat as max relevance
    return {pid: (score - lo) / (hi - lo) for pid, score in scores.items()}
