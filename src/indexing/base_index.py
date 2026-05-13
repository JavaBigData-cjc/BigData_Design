"""
Abstract base class for all vector index implementations.
Defines the common interface: build, search, save, load.
"""

from abc import ABC, abstractmethod
from typing import Optional
import time
import numpy as np


class BaseIndex(ABC):
    """Abstract base for ANN index implementations."""

    def __init__(self, dim: int, metric: str = "cosine"):
        self.dim = dim
        self.metric = metric
        self._built = False
        self._build_time = 0.0
        self._n_vectors = 0

    @abstractmethod
    def build(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None):
        """Build the index from a set of vectors."""

    @abstractmethod
    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors. Returns (ids, distances)."""

    def timed_build(self, vectors: np.ndarray,
                    ids: Optional[np.ndarray] = None) -> float:
        """Build with timing. Returns build time in seconds."""
        t0 = time.perf_counter()
        self.build(vectors, ids)
        self._build_time = time.perf_counter() - t0
        return self._build_time

    def timed_search(self, query: np.ndarray, k: int = 10) -> dict:
        """Search with timing. Returns dict with results + latency."""
        t0 = time.perf_counter()
        ids, distances = self.search(query, k)
        latency = (time.perf_counter() - t0) * 1000  # ms
        return {"ids": ids, "distances": distances, "latency_ms": latency}

    @property
    def is_built(self) -> bool:
        return self._built

    @property
    def build_time(self) -> float:
        return self._build_time

    @property
    def n_vectors(self) -> int:
        return self._n_vectors

    @abstractmethod
    def save(self, path: str):
        """Save index to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseIndex":
        """Load index from disk."""

    def _validate_vectors(self, vectors: np.ndarray):
        """Check vector shape matches index dimension."""
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Expected vectors of dim {self.dim}, got {vectors.shape[1]}"
            )
        return vectors
