"""
Annoy index wrapper (Approximate Nearest Neighbors Oh Yeah).
Spotify's tree-based ANN algorithm.
"""

import pickle
from typing import Optional

import numpy as np
from annoy import AnnoyIndex

from .base_index import BaseIndex


class AnnoyIndexWrapper(BaseIndex):
    """Wrapper around Spotify's Annoy library for tree-based ANN search."""

    METRIC_MAP = {
        "cosine": "angular",  # angular distance ~ cosine on normalized vectors
        "euclidean": "euclidean",
        "manhattan": "manhattan",
        "dot": "dot",
    }

    def __init__(self, dim: int, n_trees: int = 10,
                 search_k: int = -1, metric: str = "cosine"):
        super().__init__(dim, metric)
        self.n_trees = n_trees
        self.search_k = search_k
        self.index = None
        self._id_map = None

    def build(self, vectors: np.ndarray,
              ids: Optional[np.ndarray] = None):
        vectors = self._validate_vectors(vectors).astype(np.float32)
        self._n_vectors = len(vectors)

        metric_str = self.METRIC_MAP.get(self.metric, "angular")
        self.index = AnnoyIndex(self.dim, metric_str)

        if ids is None:
            ids = np.arange(len(vectors))

        for i, (vec, nid) in enumerate(zip(vectors, ids)):
            self.index.add_item(int(nid), vec)

        self.index.build(self.n_trees)
        self._id_map = ids
        self._built = True

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        query = self._validate_vectors(query)

        all_ids = []
        all_distances = []

        for q in query:
            if self.search_k > 0:
                ids, distances = self.index.get_nns_by_vector(
                    q, k, search_k=self.search_k, include_distances=True
                )
            else:
                ids, distances = self.index.get_nns_by_vector(
                    q, k, include_distances=True
                )
            all_ids.append(ids)
            all_distances.append(distances)

        if len(all_ids) == 1:
            return np.array(all_ids[0]), np.array(all_distances[0])
        return np.array(all_ids), np.array(all_distances)

    def save(self, path: str):
        self.index.save(f"{path}.ann")
        meta = {
            "dim": self.dim, "n_trees": self.n_trees,
            "search_k": self.search_k, "metric": self.metric,
        }
        with open(f"{path}.meta.pkl", "wb") as f:
            pickle.dump(meta, f)

    @classmethod
    def load(cls, path: str) -> "AnnoyIndexWrapper":
        with open(f"{path}.meta.pkl", "rb") as f:
            meta = pickle.load(f)

        metric_str = cls.METRIC_MAP.get(meta["metric"], "angular")
        obj = cls(dim=meta["dim"], n_trees=meta["n_trees"],
                  search_k=meta["search_k"], metric=meta["metric"])
        obj.index = AnnoyIndex(meta["dim"], metric_str)
        obj.index.load(f"{path}.ann")
        obj._built = True
        return obj
