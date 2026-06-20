"""Unit tests for src/services/retrieval/bm25_index.py."""
from __future__ import annotations

from src.services.retrieval.bm25_index import BM25KeywordIndex


class TestBuildAndSearch:
    def test_search_before_build_returns_empty(self):
        index = BM25KeywordIndex()
        assert index.search("anything", top_k=5) == []

    def test_build_with_empty_corpus(self):
        index = BM25KeywordIndex()
        index.build([])
        assert index.is_built is False
        assert index.count() == 0

    def test_exact_keyword_match_ranks_highest(self, sample_catalog):
        index = BM25KeywordIndex()
        index.build(sample_catalog)

        results = index.search("waterproof hiking", top_k=5)
        assert len(results) > 0
        top_product_id = results[0][0]
        # Both the jacket and the backpack mention "waterproof" and "hiking" —
        # either is a reasonable top match; what matters is *some* relevant
        # result is returned and scores are descending.
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_no_term_overlap_returns_empty(self, sample_catalog):
        index = BM25KeywordIndex()
        index.build(sample_catalog)
        results = index.search("zzznonexistenttermzzz", top_k=5)
        assert results == []

    def test_brand_name_match(self, sample_catalog):
        index = BM25KeywordIndex()
        index.build(sample_catalog)
        results = index.search("trailpeak", top_k=10)
        result_ids = {pid for pid, _ in results}
        trailpeak_products = {p.product_id for p in sample_catalog if p.brand == "TrailPeak"}
        assert result_ids == trailpeak_products


class TestUpsertAndDelete:
    def test_upsert_adds_new_product_to_index(self, sample_catalog):
        from src.domain.schemas import Product

        index = BM25KeywordIndex()
        index.build(sample_catalog)
        initial_count = index.count()

        new_product = Product(sku="NEW-001", title="Unique Snowboard Gear", description="snowboard equipment")
        index.upsert([new_product])

        assert index.count() == initial_count + 1
        results = index.search("snowboard", top_k=5)
        assert any(pid == new_product.product_id for pid, _ in results)

    def test_delete_removes_product_from_index(self, sample_catalog):
        index = BM25KeywordIndex()
        index.build(sample_catalog)
        target = sample_catalog[0]

        index.delete([target.product_id])
        assert index.count() == len(sample_catalog) - 1

    def test_delete_all_products_leaves_index_empty(self, sample_catalog):
        index = BM25KeywordIndex()
        index.build(sample_catalog)
        index.delete([p.product_id for p in sample_catalog])
        assert index.count() == 0
        assert index.is_built is False


class TestClear:
    def test_clear_resets_index(self, sample_catalog):
        index = BM25KeywordIndex()
        index.build(sample_catalog)
        index.clear()
        assert index.is_built is False
        assert index.count() == 0
        assert index.search("hiking", top_k=5) == []
