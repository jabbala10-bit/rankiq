"""Unit tests for src/services/retrieval/reranker.py."""
from __future__ import annotations

import pytest

from src.domain.exceptions import ModelNotLoadedError
from src.services.retrieval.reranker import CrossEncoderReranker


class FakeCrossEncoderModel:
    """Fake CrossEncoder: scores pairs by counting shared tokens between query and text."""

    def predict(self, pairs):
        scores = []
        for query, text in pairs:
            query_tokens = set(query.lower().split())
            text_tokens = set(text.lower().split())
            scores.append(float(len(query_tokens & text_tokens)))
        return scores


@pytest.fixture
def reranker(test_settings) -> CrossEncoderReranker:
    r = CrossEncoderReranker(test_settings)
    r._model = FakeCrossEncoderModel()  # bypass load_model() for testing
    return r


class TestRerank:
    def test_raises_if_model_not_loaded(self, test_settings):
        reranker = CrossEncoderReranker(test_settings)
        with pytest.raises(ModelNotLoadedError):
            reranker.rerank("query", [])

    def test_empty_products_returns_empty_list(self, reranker):
        assert reranker.rerank("waterproof jacket", []) == []

    def test_reranks_by_relevance_to_query(self, reranker, sample_catalog):
        results = reranker.rerank("waterproof hiking jacket", sample_catalog)
        assert len(results) == len(sample_catalog)
        # Results should be sorted descending by score
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_most_relevant_product_ranks_first(self, reranker, sample_catalog):
        results = reranker.rerank("waterproof hiking jacket trailpeak outerwear", sample_catalog)
        top_product, top_score = results[0]
        # The hiking jacket shares the most tokens with this query
        assert "Jacket" in top_product.title
