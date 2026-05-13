"""Tests for indexing module."""
import numpy as np
import os
import tempfile


class TestFAISSIndex:
    """Test FAISS index wrapper."""

    def test_build_and_search(self):
        from src.indexing.faiss_index import FAISSIndex
        dim = 64
        n = 1000
        vectors = np.random.randn(n, dim).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        index = FAISSIndex(dim=dim, index_type="flat")
        index.build(vectors)

        # Self-retrieval: query vector should find itself first
        query = vectors[0:1]
        ids, dists = index.search(query, k=3)
        assert ids[0, 0] == 0  # Self is closest
        assert dists[0, 0] < 0.01  # Near-zero distance

    def test_save_and_load(self):
        from src.indexing.faiss_index import FAISSIndex
        dim = 64
        vectors = np.random.randn(100, dim).astype(np.float32)

        index = FAISSIndex(dim=dim, index_type="flat")
        index.build(vectors)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_faiss")
            index.save(path)

            loaded = FAISSIndex.load(path)
            assert loaded.is_built
            assert loaded.dim == dim

            # Verify same search results
            query = vectors[5:6]
            ids1, _ = index.search(query, k=3)
            ids2, _ = loaded.search(query, k=3)
            assert np.array_equal(ids1, ids2)


class TestManualHNSW:
    """Test manual HNSW implementation."""

    def test_build_and_search(self):
        from src.indexing.hnsw_manual import ManualHNSW
        dim = 32
        n = 200
        np.random.seed(42)
        vectors = np.random.randn(n, dim).astype(np.float32)

        hnsw = ManualHNSW(dim=dim, M=8, ef_construction=50)
        hnsw.build(vectors)

        # Self-retrieval
        query = vectors[10:11]
        ids, dists = hnsw.search(query, k=3)
        assert 10 in ids[0]  # Self should be in top results


class TestCosineLSH:
    """Test LSH implementation."""

    def test_build_and_search(self):
        from src.indexing.lsh import CosineLSH
        dim = 32
        n = 300
        np.random.seed(42)
        vectors = np.random.randn(n, dim).astype(np.float32)

        lsh = CosineLSH(dim=dim, n_tables=5, n_hashes=8)
        lsh.build(vectors)

        # Search
        query = vectors[0:1]
        ids, dists = lsh.search(query, k=5)
        assert len(ids[0]) == 5
