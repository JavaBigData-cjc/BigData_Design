"""
HNSW index via hnswlib (optimized C++ library).
Provides production-quality baseline for benchmarking against manual HNSW.
"""

import os
import pickle
from typing import Optional

import numpy as np
import hnswlib

from .base_index import BaseIndex


class HNSWLib(BaseIndex):
    """HNSW index backed by the optimized hnswlib C++ library."""

    SPACE_MAP = {
        "cosine": "cosine",
        "euclidean": "l2",
        "dot": "ip",
    }

    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200,
                 ef_search: int = 50, metric: str = "cosine"):
        super().__init__(dim, metric)
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.index = None

    def build(self, vectors: np.ndarray,
              ids: Optional[np.ndarray] = None):
        vectors = self._validate_vectors(vectors).astype(np.float32)
        self._n_vectors = len(vectors)

        space = self.SPACE_MAP.get(self.metric, "cosine")

        self.index = hnswlib.Index(space=space, dim=self.dim)
        self.index.init_index(
            max_elements=max(len(vectors), 1),
            ef_construction=self.ef_construction,
            M=self.M,
        )

        if ids is None:
            ids = np.arange(len(vectors)).astype(np.int64)

        self.index.add_items(vectors, ids.astype(np.int64))
        self.index.set_ef(self.ef_search)
        self._built = True

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        query = self._validate_vectors(query).astype(np.float32)
        labels, distances = self.index.knn_query(query, k=k)
        return labels, distances

    def save(self, path: str):
        self.index.save_index(f"{path}.hnsw")
        meta = {
            "dim": self.dim, "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search, "metric": self.metric,
        }
        with open(f"{path}.meta.pkl", "wb") as f:
            pickle.dump(meta, f)

    @classmethod
    def load(cls, path: str) -> "HNSWLib":
        with open(f"{path}.meta.pkl", "rb") as f:
            meta = pickle.load(f)

        obj = cls(dim=meta["dim"], M=meta["M"],
                  ef_construction=meta["ef_construction"],
                  ef_search=meta["ef_search"], metric=meta["metric"])
        obj.index = hnswlib.Index(
            space=cls.SPACE_MAP.get(meta["metric"], "cosine"),
            dim=meta["dim"]
        )
        obj.index.load_index(f"{path}.hnsw")
        obj.index.set_ef(meta["ef_search"])
        obj._built = True
        return obj
