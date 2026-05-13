"""
Locality-Sensitive Hashing (LSH) for cosine similarity.
Uses random projection (SimHash / hyperplane LSH).
Reference: Charikar, STOC 2002.
"""

import pickle
from typing import Optional

import numpy as np

from .base_index import BaseIndex


class CosineLSH(BaseIndex):
    """Locality-Sensitive Hashing for cosine similarity using random hyperplanes."""

    def __init__(self, dim: int, n_tables: int = 10, n_hashes: int = 16,
                 metric: str = "cosine"):
        super().__init__(dim, metric)
        self.n_tables = n_tables
        self.n_hashes = n_hashes
        self.random_vectors = None  # (n_tables, n_hashes, dim)
        self.hash_tables = None     # list of dicts: hash -> list of ids
        self.vectors = None
        self._id_map = None

    def _random_plane(self, n_tables: int, n_hashes: int):
        """Generate random projection vectors (hyperplanes)."""
        return np.random.randn(n_tables, n_hashes, self.dim).astype(np.float32)

    def _hash_vector(self, vec: np.ndarray) -> list[str]:
        """Compute hash signatures across all tables."""
        # vec: (dim,) -> sign(vec @ random_vectors.T) for each table
        projections = vec @ self.random_vectors.transpose(0, 2, 1)  # (n_tables, n_hashes)
        bits = (projections > 0).astype(np.int32)
        return ["".join(str(b) for b in table_bits) for table_bits in bits]

    def build(self, vectors: np.ndarray,
              ids: Optional[np.ndarray] = None):
        vectors = self._validate_vectors(vectors).astype(np.float32)
        self._n_vectors = len(vectors)

        # L2 normalize for cosine
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norms + 1e-10)
        self.vectors = vectors

        if ids is None:
            ids = np.arange(len(vectors))
        self._id_map = ids

        self.random_vectors = self._random_plane(self.n_tables, self.n_hashes)
        self.hash_tables = [{} for _ in range(self.n_tables)]

        for i, vec in enumerate(vectors):
            signatures = self._hash_vector(vec)
            for table_idx, sig in enumerate(signatures):
                self.hash_tables[table_idx].setdefault(sig, []).append(i)

        self._built = True

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        query = self._validate_vectors(query).astype(np.float32)
        # Normalize query
        query = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-10)

        all_ids = []
        all_distances = []

        for q in query:
            signatures = self._hash_vector(q)
            candidates = set()
            for table_idx, sig in enumerate(signatures):
                if sig in self.hash_tables[table_idx]:
                    candidates.update(self.hash_tables[table_idx][sig])

            if len(candidates) == 0:
                # Fallback: random sample when no hash match
                candidates = set(np.random.choice(self._n_vectors, min(k * 10, self._n_vectors), replace=False))

            candidate_vecs = self.vectors[list(candidates)]
            # Cosine similarity
            similarities = candidate_vecs @ q  # Already normalized
            top_k_idx = np.argsort(-similarities)[:k]

            mapped_ids = [self._id_map[list(candidates)[i]] for i in top_k_idx]
            distances = 1.0 - similarities[top_k_idx]

            all_ids.append(mapped_ids)
            all_distances.append(distances)

        if len(all_ids) == 1:
            return np.array(all_ids[0]), np.array(all_distances[0])
        return np.array(all_ids), np.array(all_distances)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "dim": self.dim, "n_tables": self.n_tables,
                "n_hashes": self.n_hashes, "metric": self.metric,
                "random_vectors": self.random_vectors,
                "hash_tables": self.hash_tables,
                "vectors": self.vectors,
                "id_map": self._id_map,
                "n_vectors": self._n_vectors,
            }, f)

    @classmethod
    def load(cls, path: str) -> "CosineLSH":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.dim = data["dim"]
        obj.n_tables = data["n_tables"]
        obj.n_hashes = data["n_hashes"]
        obj.metric = data["metric"]
        obj.random_vectors = data["random_vectors"]
        obj.hash_tables = data["hash_tables"]
        obj.vectors = data["vectors"]
        obj._id_map = data["id_map"]
        obj._n_vectors = data["n_vectors"]
        obj._built = True
        return obj
