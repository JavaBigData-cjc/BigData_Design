"""
Manual HNSW (Hierarchical Navigable Small World) implementation.
Reference: Malkov & Yashunin, IEEE TPAMI 2018.

This is a pedagogical implementation demonstrating deep understanding
of the HNSW graph construction and search algorithm.
"""

import heapq
import pickle
import random
from typing import Optional

import numpy as np

from .base_index import BaseIndex


class HNSWNode:
    """A node in the HNSW multi-layer graph."""

    __slots__ = ("id", "vector", "level", "neighbors")

    def __init__(self, node_id: int, vector: np.ndarray, level: int):
        self.id = node_id
        self.vector = vector.astype(np.float32)
        self.level = level
        self.neighbors = {l: [] for l in range(level + 1)}  # level -> list of neighbor ids


class ManualHNSW(BaseIndex):
    """Manual HNSW implementation for educational and comparison purposes."""

    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200,
                 ef_search: int = 50, ml: float = 0.5, metric: str = "cosine"):
        super().__init__(dim, metric)
        self.M_max = M
        self.M_max0 = M * 2  # more connections at bottom layer
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.ml = ml  # level generation factor: 1/ln(2) for skip-list-like
        self.nodes = {}  # id -> HNSWNode
        self.entry_point = None

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute distance between two vectors."""
        if self.metric == "cosine":
            # cosine distance = 1 - (a·b)/(||a||*||b||)
            sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
            return 1.0 - sim
        else:
            return np.linalg.norm(a - b)

    def _random_level(self) -> int:
        """Generate random level with exponentially decaying probability."""
        r = -np.log(random.random()) * self.ml
        return int(r)

    def _select_neighbors(self, query: np.ndarray, candidates: list[tuple[float, int]],
                          M: int, use_heuristic: bool = True) -> list[tuple[float, int]]:
        """Select M nearest neighbors, with optional diversity heuristic."""
        candidates = sorted(candidates)

        if not use_heuristic or len(candidates) <= M:
            return candidates[:M]

        selected = [candidates[0]]  # Always accept the closest

        for dist, node_id in candidates[1:]:
            if len(selected) >= M:
                break
            # Heuristic: accept if closer to query than to any selected neighbor
            accept = True
            for _, sel_id in selected:
                if self._distance(self.nodes[node_id].vector,
                                  self.nodes[sel_id].vector) < dist:
                    accept = False
                    break
            if accept:
                selected.append((dist, node_id))

        # If heuristic didn't fill M, pad with closest remaining
        if len(selected) < M:
            for dist, node_id in candidates:
                if (dist, node_id) not in selected:
                    selected.append((dist, node_id))
                    if len(selected) >= M:
                        break

        return selected[:M]

    def _search_layer(self, query: np.ndarray, entry_id: int,
                      level: int, ef: int) -> list[tuple[float, int]]:
        """Greedy search within a single layer."""
        visited = {entry_id}
        dist_entry = self._distance(query, self.nodes[entry_id].vector)
        # Max-heap: candidates (negative distance for min-heap behavior)
        candidates = [(-dist_entry, entry_id)]
        # Min-heap: results (distance, id)
        results = [(dist_entry, entry_id)]

        while candidates:
            neg_dist, current_id = heapq.heappop(candidates)

            # Prune if current is farther than the ef-th best result
            farthest = max(d for d, _ in results) if len(results) >= ef else float("inf")
            if -neg_dist > farthest:
                break

            for neighbor_id in self.nodes[current_id].neighbors[level]:
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)

                dist = self._distance(query, self.nodes[neighbor_id].vector)
                farthest = max(d for d, _ in results) if len(results) >= ef else float("inf")

                if dist < farthest or len(results) < ef:
                    heapq.heappush(candidates, (-dist, neighbor_id))
                    heapq.heappush(results, (dist, neighbor_id))
                    if len(results) > ef:
                        results = sorted(results)[:ef]
                        heapq.heapify(results)

        return sorted(results)[:ef]

    def build(self, vectors: np.ndarray,
              ids: Optional[np.ndarray] = None):
        vectors = self._validate_vectors(vectors)
        self._n_vectors = len(vectors)

        if ids is None:
            ids = np.arange(len(vectors))

        for i, (vec, nid) in enumerate(zip(vectors, ids)):
            self._insert(int(nid), vec)

        self._built = True

    def _insert(self, node_id: int, vector: np.ndarray):
        """Insert a single node into the HNSW graph."""
        level = self._random_level()
        node = HNSWNode(node_id, vector, level)

        if self.entry_point is None:
            self.nodes[node_id] = node
            self.entry_point = node_id
            return

        # Start from entry point at top level
        current_ep = self.entry_point
        top_level = self.nodes[self.entry_point].level

        # Navigate down from top level to just above new node's level
        for l in range(top_level, level, -1):
            results = self._search_layer(vector, current_ep, l, 1)
            current_ep = results[0][1]

        # Insert from min(level, top_level) down to 0
        start_level = min(level, top_level)
        for l in range(start_level, -1, -1):
            results = self._search_layer(vector, current_ep, l, self.ef_construction)
            M = self.M_max if l > 0 else self.M_max0
            neighbors = self._select_neighbors(vector, results, M)

            # Add bidirectional connections
            node.neighbors[l] = [nid for _, nid in neighbors]

            for dist, neighbor_id in neighbors:
                nb = self.nodes[neighbor_id]
                if l not in nb.neighbors:
                    nb.neighbors[l] = []

                nb.neighbors[l].append(node_id)

                # Prune neighbor's connections if too many
                if len(nb.neighbors[l]) > M:
                    nb_vec = nb.vector
                    all_nb = [(self._distance(nb_vec, self.nodes[nid].vector), nid)
                              for nid in nb.neighbors[l] if nid != node_id]
                    nb.neighbors[l] = [nid for _, nid in
                                       self._select_neighbors(nb_vec, all_nb, M)]

            current_ep = results[0][1]

        self.nodes[node_id] = node

        # Update entry point if new node has higher level
        if level > self.nodes[self.entry_point].level:
            self.entry_point = node_id

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        query = self._validate_vectors(query)
        all_ids = []
        all_distances = []

        for q in query:
            # Navigate down from top
            current_ep = self.entry_point
            top_level = self.nodes[self.entry_point].level

            for l in range(top_level, 1, -1):
                results = self._search_layer(q, current_ep, l, 1)
                current_ep = results[0][1]

            # Search bottom layer with ef = k
            results = self._search_layer(q, current_ep, 0, self.ef_search)
            ids = [node_id for _, node_id in results[:k]]
            dists = [dist for dist, _ in results[:k]]

            all_ids.append(ids)
            all_distances.append(dists)

        if len(all_ids) == 1:
            return np.array(all_ids[0]), np.array(all_distances[0])
        return np.array(all_ids), np.array(all_distances)

    def save(self, path: str):
        """Save HNSW index via pickle."""
        with open(path, "wb") as f:
            pickle.dump({
                "dim": self.dim, "M_max": self.M_max,
                "M_max0": self.M_max0, "ef_construction": self.ef_construction,
                "ef_search": self.ef_search,
                "ml": self.ml, "metric": self.metric,
                "nodes": self.nodes, "entry_point": self.entry_point,
                "n_vectors": self._n_vectors,
            }, f)

    @classmethod
    def load(cls, path: str) -> "ManualHNSW":
        """Load HNSW index from pickle."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        obj = cls.__new__(cls)
        obj.dim = data["dim"]
        obj.M_max = data["M_max"]
        obj.M_max0 = data["M_max0"]
        obj.ef_construction = data["ef_construction"]
        obj.ef_search = data.get("ef_search", 50)
        obj.ml = data["ml"]
        obj.metric = data["metric"]
        obj.nodes = data["nodes"]
        obj.entry_point = data["entry_point"]
        obj._n_vectors = data["n_vectors"]
        obj._built = True
        return obj
