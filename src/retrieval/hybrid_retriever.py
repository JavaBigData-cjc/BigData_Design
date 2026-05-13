"""
Hybrid Retriever: orchestrates dense (vector), sparse (BM25), and metadata retrieval.

This is the core innovation module combining:
1. Adaptive query routing (DAT-inspired, Hsu & Tzeng 2025)
2. Reciprocal Rank Fusion (Cormack et al., SIGIR 2009)
3. Multi-stage coarse-to-fine retrieval (ACE + CMC inspired)
4. Weighted score fusion with calibration
"""

from typing import Optional

import numpy as np
import pandas as pd

from .query_router import QueryRouter
from .bm25_index import BM25Index
from .fusion import ScoreFusion
from .metadata_filter import MetadataFilter


class HybridRetriever:
    """Orchestrates multi-strategy hybrid retrieval for cross-modal search."""

    def __init__(self, dense_index, bm25_index: BM25Index,
                 metadata_df: Optional[pd.DataFrame] = None,
                 query_router: Optional[QueryRouter] = None,
                 clip_encoder=None):
        self.dense_index = dense_index
        self.bm25 = bm25_index
        self.metadata_df = metadata_df
        self.meta_filter = MetadataFilter(metadata_df)

        self.query_router = query_router or QueryRouter(
            bm25_index=bm25_index,
            default_weights=(0.6, 0.25, 0.15)
        )
        self.fusion = ScoreFusion()
        self.encoder = clip_encoder

    def search(self, query_text: str = None, query_image=None,
               k: int = 10, strategy: str = "adaptive",
               query_embedding: Optional[np.ndarray] = None) -> list[dict]:
        """Main search entry point.

        Args:
            query_text: Natural language query string
            query_image: PIL Image for image-to-text retrieval
            k: Number of results to return
            strategy: "pure_dense", "pure_sparse", "fixed_weight",
                      "adaptive", "multi_stage"
            query_embedding: Pre-computed embedding (bypasses encoder)

        Returns:
            List of dicts: [{id, score, caption, image_path}, ...]
        """
        if query_embedding is None and self.encoder is not None:
            if query_text:
                query_embedding = self.encoder.encode_texts([query_text])[0]
            elif query_image is not None:
                query_embedding = self.encoder.encode_images([query_image])[0]

        if strategy == "pure_dense":
            return self._dense_search(query_embedding, k)

        elif strategy == "pure_sparse":
            return self._sparse_search(query_text, k)

        elif strategy == "fixed_weight":
            return self._fixed_weight_search(query_text, query_embedding, k)

        elif strategy == "adaptive":
            return self._adaptive_search(query_text, query_embedding, k)

        elif strategy == "multi_stage":
            return self._multi_stage_search(query_text, query_embedding, k)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _dense_search(self, query_emb: np.ndarray, k: int) -> list[dict]:
        """Pure dense (vector) retrieval."""
        ids, distances = self.dense_index.search(query_emb.reshape(1, -1), k=k)
        scores = 1.0 - distances.flatten() if self.dense_index.metric == "cosine" else -distances.flatten()
        return self._format_results(ids.flatten(), scores)

    def _sparse_search(self, query_text: str, k: int) -> list[dict]:
        """Pure sparse (BM25) retrieval."""
        ids, scores = self.bm25.search(query_text, k=k)
        return self._format_results(ids, scores)

    def _fixed_weight_search(self, query_text: str, query_emb: np.ndarray,
                             k: int) -> list[dict]:
        """Fixed-weight hybrid dense + sparse."""
        w_d, w_s, w_m = 0.6, 0.25, 0.15

        # Dense top candidates
        dense_k = min(k * 20, self.dense_index.n_vectors)
        d_ids, d_dists = self.dense_index.search(query_emb.reshape(1, -1), k=dense_k)
        dense_scores = 1.0 - d_dists.flatten()

        # Sparse
        s_ids, sparse_scores = self.bm25.search(query_text, k=dense_k)

        # Align scores to common indices
        all_ids = sorted(set(d_ids.flatten()) | set(s_ids))
        dense_aligned = np.array([dense_scores[np.where(d_ids.flatten() == i)[0][0]] if i in d_ids.flatten() else 0.0 for i in all_ids])
        sparse_aligned = np.array([sparse_scores[np.where(s_ids == i)[0][0]] if i in s_ids else 0.0 for i in all_ids])

        fused = self.fusion.weighted_sum(dense_aligned, sparse_aligned,
                                         weights=(w_d, w_s, 0.0))
        top_k = np.argsort(-fused)[:k]
        return self._format_results(np.array(all_ids)[top_k], fused[top_k])

    def _adaptive_search(self, query_text: str, query_emb: np.ndarray,
                         k: int) -> list[dict]:
        """Adaptive hybrid search with query-router determined weights."""
        w_d, w_s, w_m = self.query_router.get_weights(query_text)
        q_type = self.query_router.classify(query_text)

        # Expand candidates
        candidate_k = min(k * 20, self.dense_index.n_vectors)
        d_ids, d_dists = self.dense_index.search(query_emb.reshape(1, -1), k=candidate_k)
        dense_scores = 1.0 - d_dists.flatten()

        s_ids, sparse_scores = self.bm25.search(query_text, k=candidate_k)

        # Metadata scores
        if self.metadata_df is not None:
            meta_scores = self.meta_filter.compute_metadata_scores(query_text, self.metadata_df)
        else:
            meta_scores = None

        # Align to common ID space
        all_ids = sorted(set(d_ids.flatten()) | set(s_ids))
        dense_aligned = np.zeros(len(all_ids))
        sparse_aligned = np.zeros(len(all_ids))
        meta_aligned = np.zeros(len(all_ids))

        id_to_idx = {i: idx for idx, i in enumerate(all_ids)}
        for idx, did in enumerate(d_ids.flatten()):
            dense_aligned[id_to_idx[did]] = dense_scores[idx]
        for idx, sid in enumerate(s_ids):
            if sid in id_to_idx:
                sparse_aligned[id_to_idx[sid]] = sparse_scores[idx]
        if meta_scores is not None:
            for idx, mid in enumerate(all_ids):
                if mid < len(meta_scores):
                    meta_aligned[idx] = meta_scores[mid]

        fused = self.fusion.weighted_sum(dense_aligned, sparse_aligned,
                                         meta_aligned if meta_scores is not None else None,
                                         weights=(w_d, w_s, w_m))
        top_k = np.argsort(-fused)[:k]
        return self._format_results(np.array(all_ids)[top_k], fused[top_k],
                                    extra={"query_type": q_type.value,
                                           "weights": {"dense": w_d, "sparse": w_s, "meta": w_m}})

    def _multi_stage_search(self, query_text: str, query_emb: np.ndarray,
                            k: int) -> list[dict]:
        """Multi-stage coarse-to-fine retrieval.

        Stage 1: Fast ANN -> 500 candidates
        Stage 2: RRF fusion with BM25 -> 200 candidates
        Stage 3: Re-rank with full cross-modal similarity -> 50 candidates
        """
        # Stage 1: Coarse ANN
        stage1_k = 500
        d_ids, _ = self.dense_index.search(query_emb.reshape(1, -1), k=stage1_k)

        # Stage 2: RRF fusion
        s_ids, _ = self.bm25.search(query_text, k=stage1_k)
        dense_ranking = [(int(i), float(1.0/(n+1))) for n, i in enumerate(d_ids.flatten())]
        sparse_ranking = [(int(i), float(1.0/(n+1))) for n, i in enumerate(s_ids)]

        rrf_results = self.fusion.reciprocal_rank_fusion(
            [dense_ranking, sparse_ranking], k=60
        )
        stage2_ids = [doc_id for doc_id, _ in rrf_results[:200]]

        # Stage 3: Full re-rank if encoder available
        if self.encoder is not None and hasattr(self, '_captions'):
            stage3_results = self._rerank_with_clip(query_text, query_emb, stage2_ids)
            return stage3_results[:k]

        return self._format_results(
            np.array(stage2_ids[:k]),
            np.array([score for _, score in rrf_results[:k]])
        )

    def _rerank_with_clip(self, query_text: str, query_emb: np.ndarray,
                          candidate_ids: list[int]) -> list[dict]:
        """Re-rank candidates using full CLIP cross-modal similarity."""
        # This would use the captions/embeddings to compute exact CLIP similarity
        # Placeholder for implementation
        results = []
        for rank, cid in enumerate(candidate_ids):
            results.append({"id": cid, "score": 1.0 / (1 + rank)})
        return results

    def _format_results(self, ids: np.ndarray, scores: np.ndarray,
                        extra: dict = None) -> list[dict]:
        """Format raw results into structured list."""
        results = []
        for i, (doc_id, score) in enumerate(zip(ids, scores)):
            result = {
                "rank": i + 1,
                "id": int(doc_id),
                "score": float(score),
            }
            if extra:
                result.update(extra)
            results.append(result)
        return results
