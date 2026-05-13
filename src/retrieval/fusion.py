"""
Score fusion strategies for hybrid retrieval.
Supports: Weighted Sum, Reciprocal Rank Fusion (RRF), and Score Calibration.
Reference: Cormack, Clarke, Büttcher, SIGIR 2009.
"""

from collections import defaultdict
from typing import Optional

import numpy as np


class ScoreFusion:
    """Collection of fusion strategies combining multiple retrieval signals."""

    @staticmethod
    def weighted_sum(dense_scores: np.ndarray, sparse_scores: np.ndarray,
                     meta_scores: Optional[np.ndarray] = None,
                     weights: tuple[float, float, float] = (0.6, 0.25, 0.15)) -> np.ndarray:
        """Linear weighted combination of normalized scores.

        Args:
            dense_scores: (N,) dense retrieval scores
            sparse_scores: (N,) sparse (BM25) scores
            meta_scores: (N,) metadata match scores (optional)
            weights: (w_dense, w_sparse, w_meta)

        Returns:
            (N,) fused scores
        """
        # Min-max normalize each score distribution
        def normalize(scores):
            s_min, s_max = scores.min(), scores.max()
            if s_max - s_min < 1e-10:
                return np.zeros_like(scores)
            return (scores - s_min) / (s_max - s_min)

        fused = weights[0] * normalize(dense_scores) + weights[1] * normalize(sparse_scores)

        if meta_scores is not None:
            fused += weights[2] * normalize(meta_scores)

        return fused

    @staticmethod
    def reciprocal_rank_fusion(rankings_list: list[list[tuple[int, float]]],
                               k: int = 60) -> list[tuple[int, float]]:
        """Reciprocal Rank Fusion (Cormack et al., SIGIR 2009).

        Args:
            rankings_list: list of rankings, each is [(doc_id, score), ...]
            k: RRF constant (default 60)

        Returns:
            Sorted list of (doc_id, rrf_score)
        """
        scores = defaultdict(float)

        for ranking in rankings_list:
            for rank, (doc_id, _) in enumerate(ranking, 1):
                scores[doc_id] += 1.0 / (k + rank)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    @staticmethod
    def score_calibration(scores_list: list[np.ndarray],
                          method: str = "minmax") -> list[np.ndarray]:
        """Calibrate score distributions from different retrievers.

        Args:
            scores_list: list of score arrays from different retrievers
            method: "minmax" or "zscore"

        Returns:
            list of calibrated score arrays
        """
        calibrated = []
        for scores in scores_list:
            if method == "minmax":
                s_min, s_max = scores.min(), scores.max()
                if s_max - s_min < 1e-10:
                    calibrated.append(np.zeros_like(scores))
                else:
                    calibrated.append((scores - s_min) / (s_max - s_min))
            elif method == "zscore":
                s_mean, s_std = scores.mean(), scores.std()
                if s_std < 1e-10:
                    calibrated.append(np.zeros_like(scores))
                else:
                    calibrated.append((scores - s_mean) / s_std)
        return calibrated
