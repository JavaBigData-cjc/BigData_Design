"""
FAISS IVF-PQ index wrapper.
Supports: Flat (brute-force), IVF+PQ, HNSW via FAISS.
"""

import os
import pickle
from typing import Optional

import numpy as np
import faiss

from .base_index import BaseIndex


class FAISSIndex(BaseIndex):
    """FAISS-based ANN index supporting multiple index types."""

    INDEX_TYPES = ["flat", "ivfpq", "hnsw"]

    def __init__(self, dim: int, index_type: str = "ivfpq",
                 metric: str = "cosine",
                 nlist: int = 100, m: int = 8, nbits: int = 8):
        super().__init__(dim, metric)
        self.index_type = index_type
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.index = None
        self._id_map = None

    def build(self, vectors: np.ndarray,
              ids: Optional[np.ndarray] = None):
        vectors = self._validate_vectors(vectors).astype(np.float32)
        self._n_vectors = len(vectors)

        # Normalize vectors for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(vectors)

        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(self.dim)  # inner product ~ cosine

        elif self.index_type == "ivfpq":
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFPQ(
                quantizer, self.dim, self.nlist, self.m, self.nbits
            )
            self.index.train(vectors)

        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.dim, self.nlist)
            # nlist is repurposed as M (connections) in HNSW mode

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        self.index.add(vectors)
        self._id_map = ids if ids is not None else np.arange(len(vectors))
        self._built = True

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        query = self._validate_vectors(query).astype(np.float32)

        if self.metric == "cosine":
            faiss.normalize_L2(query)

        distances, indices = self.index.search(query, k)

        # Map FAISS internal indices to external IDs
        mapped_indices = self._id_map[indices] if self._id_map is not None else indices

        # For cosine (inner product), convert to distance: dist = 1 - similarity
        if self.metric == "cosine":
            distances = 1.0 - distances

        return mapped_indices, distances

    def save(self, path: str):
        faiss.write_index(self.index, f"{path}.faiss")
        meta = {
            "dim": self.dim, "index_type": self.index_type,
            "metric": self.metric, "nlist": self.nlist,
            "m": self.m, "nbits": self.nbits,
            "id_map": self._id_map,
        }
        with open(f"{path}.meta.pkl", "wb") as f:
            pickle.dump(meta, f)

    @classmethod
    def load(cls, path: str) -> "FAISSIndex":
        with open(f"{path}.meta.pkl", "rb") as f:
            meta = pickle.load(f)
        obj = cls(dim=meta["dim"], index_type=meta["index_type"],
                  metric=meta["metric"], nlist=meta["nlist"],
                  m=meta["m"], nbits=meta["nbits"])
        obj.index = faiss.read_index(f"{path}.faiss")
        obj._id_map = meta["id_map"]
        obj._built = True
        return obj

    @property
    def memory_usage_mb(self) -> float:
        """Estimate index memory in MB."""
        if self.index is None:
            return 0.0
        # Rough estimate
        return self.index.ntotal * self.dim * 4 / (1024 * 1024)
