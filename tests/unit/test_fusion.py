"""Unit tests for src/services/retrieval/fusion.py."""
from __future__ import annotations

from src.services.retrieval.fusion import fuse_rrf, fuse_weighted_sum


class TestRRF:
    def test_item_in_both_lists_ranks_higher_than_single_list_items(self):
        vector_ids = ["p1", "p2", "p3"]
        keyword_ids = ["p1", "p4", "p5"]
        fused = fuse_rrf(vector_ids, keyword_ids)
        fused_ids = [pid for pid, _ in fused]
        assert fused_ids[0] == "p1"  # appears in both lists at rank 1 -> highest combined score

    def test_top_ranked_item_scores_higher_than_lower_ranked(self):
        vector_ids = ["p1", "p2", "p3"]
        keyword_ids = []
        fused = fuse_rrf(vector_ids, keyword_ids)
        scores = dict(fused)
        assert scores["p1"] > scores["p2"] > scores["p3"]

    def test_empty_lists_produce_empty_result(self):
        assert fuse_rrf([], []) == []

    def test_disjoint_lists_includes_all_items(self):
        fused = fuse_rrf(["p1", "p2"], ["p3", "p4"])
        fused_ids = {pid for pid, _ in fused}
        assert fused_ids == {"p1", "p2", "p3", "p4"}

    def test_k_parameter_changes_relative_weighting(self):
        # A smaller k makes rank position matter more (steeper falloff)
        vector_ids = ["p1", "p2"]
        keyword_ids = []
        fused_small_k = dict(fuse_rrf(vector_ids, keyword_ids, k=1))
        fused_large_k = dict(fuse_rrf(vector_ids, keyword_ids, k=1000))

        ratio_small_k = fused_small_k["p1"] / fused_small_k["p2"]
        ratio_large_k = fused_large_k["p1"] / fused_large_k["p2"]
        assert ratio_small_k > ratio_large_k  # smaller k -> bigger gap between rank 1 and rank 2


class TestWeightedSum:
    def test_higher_vector_score_wins_with_vector_weight_dominant(self):
        vector_scores = {"p1": 0.9, "p2": 0.1}
        keyword_scores = {"p1": 0.1, "p2": 0.9}
        fused = fuse_weighted_sum(vector_scores, keyword_scores, vector_weight=0.9, keyword_weight=0.1)
        assert fused[0][0] == "p1"

    def test_higher_keyword_score_wins_with_keyword_weight_dominant(self):
        vector_scores = {"p1": 0.9, "p2": 0.1}
        keyword_scores = {"p1": 0.1, "p2": 0.9}
        fused = fuse_weighted_sum(vector_scores, keyword_scores, vector_weight=0.1, keyword_weight=0.9)
        assert fused[0][0] == "p2"

    def test_item_only_in_one_list_still_included(self):
        vector_scores = {"p1": 0.5}
        keyword_scores = {"p2": 0.5}
        fused = fuse_weighted_sum(vector_scores, keyword_scores, vector_weight=0.5, keyword_weight=0.5)
        fused_ids = {pid for pid, _ in fused}
        assert fused_ids == {"p1", "p2"}

    def test_equal_scores_normalize_to_max_relevance(self):
        # min == max in a score dict -> normalization should not divide by zero
        vector_scores = {"p1": 0.5, "p2": 0.5}
        keyword_scores = {}
        fused = fuse_weighted_sum(vector_scores, keyword_scores, vector_weight=1.0, keyword_weight=0.0)
        assert all(score == 1.0 for _, score in fused)

    def test_empty_inputs_produce_empty_result(self):
        assert fuse_weighted_sum({}, {}, 0.5, 0.5) == []
