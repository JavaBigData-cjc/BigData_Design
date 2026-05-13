"""Tests for retrieval module."""
import numpy as np
import pytest


class TestBM25Index:
    """Test BM25 sparse retrieval."""

    def test_build_and_search(self):
        from src.retrieval.bm25_index import BM25Index
        docs = [
            "the cat sits on the mat",
            "a dog runs in the park",
            "cats and dogs are friends",
            "the bird flies high in the sky",
        ]

        bm25 = BM25Index()
        bm25.build(docs)

        ids, scores = bm25.search("cat on mat", k=2)
        assert len(ids) == 2
        # First doc should match best
        assert ids[0] == 0


class TestQueryRouter:
    """Test adaptive query router."""

    def test_classify_semantic(self):
        from src.retrieval.query_router import QueryRouter, QueryType
        router = QueryRouter()
        q = "a peaceful sunset over the ocean"
        assert router.classify(q) == QueryType.SEMANTIC

    def test_classify_keyword(self):
        from src.retrieval.query_router import QueryRouter, QueryType
        router = QueryRouter()
        q = "red car blue sky automobile vehicle"
        assert router.classify(q) in (QueryType.KEYWORD, QueryType.HYBRID)

    def test_classify_metadata(self):
        from src.retrieval.query_router import QueryRouter, QueryType
        router = QueryRouter()
        q = "outdoor photos from 2023"
        assert router.classify(q) in (QueryType.METADATA, QueryType.HYBRID)

    def test_get_weights(self):
        from src.retrieval.query_router import QueryRouter
        router = QueryRouter()
        w_d, w_s, w_m = router.get_weights("outdoor photos of golden retrievers from 2023")
        # Weights should sum to 1.0
        assert abs(w_d + w_s + w_m - 1.0) < 0.01
        # Metadata weight should be elevated for this query
        assert w_m > router.default_meta


class TestScoreFusion:
    """Test score fusion strategies."""

    def test_weighted_sum(self):
        from src.retrieval.fusion import ScoreFusion
        dense = np.array([0.8, 0.3, 0.1, 0.9])
        sparse = np.array([0.2, 0.9, 0.1, 0.3])

        fused = ScoreFusion.weighted_sum(dense, sparse, weights=(0.7, 0.3, 0.0))
        assert len(fused) == 4
        # Best item with 0.7-0.3 should be index 0 or 3
        assert np.argmax(fused) in (0, 3)

    def test_reciprocal_rank_fusion(self):
        from src.retrieval.fusion import ScoreFusion
        ranking_1 = [(1, 0.9), (2, 0.7), (3, 0.5)]
        ranking_2 = [(2, 0.8), (3, 0.6), (1, 0.4)]

        fused = ScoreFusion.reciprocal_rank_fusion([ranking_1, ranking_2])
        # Top result should have good ranks in both lists
        top_id, top_score = fused[0]
        assert top_id in (1, 2)
