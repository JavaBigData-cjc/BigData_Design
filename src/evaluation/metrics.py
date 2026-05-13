"""
Evaluation metrics for cross-modal retrieval.
Recall@K, Precision@K, mAP, MRR, latency statistics.
"""

from typing import Optional

import numpy as np


class RetrievalMetrics:
    """Compute standard information retrieval evaluation metrics."""

    @staticmethod
    def recall_at_k(predictions: np.ndarray, ground_truth: np.ndarray,
                    k: int) -> float:
        """Recall@K: fraction of queries where relevant item is in top-K.

        Args:
            predictions: (num_queries, K) predicted item IDs
            ground_truth: (num_queries,) true relevant item IDs
            k: cutoff

        Returns:
            Recall@k value in [0, 1]
        """
        num_queries = len(predictions)
        hits = 0
        for i in range(num_queries):
            if ground_truth[i] in predictions[i, :k]:
                hits += 1
        return hits / num_queries

    @staticmethod
    def recall_at_k_batch(predictions: np.ndarray, ground_truth: np.ndarray,
                          k_values: list[int] = (1, 5, 10, 50)) -> dict[int, float]:
        """Compute Recall@K for multiple K values."""
        return {k: RetrievalMetrics.recall_at_k(predictions, ground_truth, k)
                for k in k_values}

    @staticmethod
    def precision_at_k(predictions: np.ndarray, ground_truth: np.ndarray,
                       k: int) -> float:
        """Precision@K: fraction of top-K results that are relevant."""
        num_queries = len(predictions)
        precisions = []
        for i in range(num_queries):
            relevant = np.sum(predictions[i, :k] == ground_truth[i])
            precisions.append(relevant / k)
        return np.mean(precisions)

    @staticmethod
    def mean_average_precision(predictions: np.ndarray,
                               ground_truth: np.ndarray,
                               k: Optional[int] = None) -> float:
        """Mean Average Precision (mAP)."""
        if k is not None:
            predictions = predictions[:, :k]

        num_queries = len(predictions)
        aps = []

        for i in range(num_queries):
            relevant_mask = predictions[i] == ground_truth[i]
            cumsum = np.cumsum(relevant_mask)
            positions = np.arange(1, len(relevant_mask) + 1)
            precision_at_k = cumsum / positions
            ap = np.sum(precision_at_k * relevant_mask) / max(np.sum(relevant_mask), 1)
            aps.append(ap)

        return np.mean(aps)

    @staticmethod
    def mean_reciprocal_rank(predictions: np.ndarray,
                             ground_truth: np.ndarray) -> float:
        """Mean Reciprocal Rank (MRR)."""
        num_queries = len(predictions)
        rr_sum = 0.0
        for i in range(num_queries):
            matches = np.where(predictions[i] == ground_truth[i])[0]
            if len(matches) > 0:
                rr_sum += 1.0 / (matches[0] + 1)
        return rr_sum / num_queries

    @staticmethod
    def latency_stats(latencies_ms: list[float]) -> dict:
        """Compute latency statistics (P50, P95, P99)."""
        arr = np.array(latencies_ms)
        return {
            "mean_ms": float(np.mean(arr)),
            "std_ms": float(np.std(arr)),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
        }
