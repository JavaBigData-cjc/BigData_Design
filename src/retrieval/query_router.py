"""
Query Router: classifies user queries and determines adaptive fusion weights.
Inspired by DAT (Dynamic Alpha Tuning, Hsu & Tzeng 2025) and adaptive IR literature.
"""

import re
from enum import Enum
from typing import Optional

import numpy as np


class QueryType(Enum):
    SEMANTIC = "semantic"      # Conceptual / descriptive: "a peaceful sunset scene"
    KEYWORD = "keyword"        # Specific terms: "red car on blue sky"
    METADATA = "metadata"      # Attribute-based: "outdoor photos from 2023"
    HYBRID = "hybrid"          # Mixed: "outdoor photos of golden retrievers"


class QueryRouter:
    """Analyzes query type and determines adaptive weights for hybrid retrieval.

    The router uses heuristic features to classify queries:
    - Presence of temporal/location/category keywords -> metadata
    - High-IDF (rare) terms -> keyword-oriented
    - Longer, descriptive phrases -> semantic-oriented

    Reference: DAT paper (Hsu & Tzeng, arXiv:2503.23013, 2025) for the
    concept of per-query adaptive alpha tuning.
    """

    TEMPORAL_KEYWORDS = {
        "2020", "2021", "2022", "2023", "2024", "2025", "2026",
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "spring", "summer", "autumn", "winter", "fall",
        "morning", "afternoon", "evening", "night",
        "today", "yesterday", "tomorrow", "last", "recent",
        "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
    }

    LOCATION_KEYWORDS = {
        "indoor", "outdoor", "inside", "outside", "street", "beach",
        "mountain", "forest", "city", "park", "garden", "kitchen",
        "living", "room", "office", "restaurant", "airport",
    }

    CATEGORY_KEYWORDS = {
        "photo", "image", "picture", "painting", "sketch", "drawing",
        "portrait", "landscape",
    }

    def __init__(self, bm25_index=None, default_weights: tuple[float, float, float] = (0.6, 0.25, 0.15)):
        self.bm25 = bm25_index
        self.default_dense, self.default_sparse, self.default_meta = default_weights

    def classify(self, query: str) -> QueryType:
        """Classify query into one of four types."""
        tokens = query.lower().split()
        features = self._extract_features(query)

        meta_score = features["has_temporal"] + features["has_location"] + features["has_category"]
        keyword_score = features["avg_idf"]  # high IDF = rare words = keyword query
        semantic_score = features["query_length"] / 10.0  # longer = likely descriptive

        if meta_score >= 1 and keyword_score >= 3:
            return QueryType.HYBRID
        elif meta_score >= 1:
            return QueryType.METADATA
        elif keyword_score >= 3:
            return QueryType.KEYWORD
        else:
            return QueryType.SEMANTIC

    def get_weights(self, query: str) -> tuple[float, float, float]:
        """Compute adaptive (dense, sparse, metadata) fusion weights."""
        features = self._extract_features(query)

        # Base weights
        w_dense = self.default_dense
        w_sparse = self.default_sparse
        w_meta = self.default_meta

        # Adjust based on features
        # High IDF terms -> increase sparse weight
        if features["avg_idf"] > 4:
            w_sparse += 0.15
            w_dense -= 0.10
            w_meta -= 0.05

        # Temporal/location/category -> increase metadata weight
        meta_factor = (features["has_temporal"] + features["has_location"] +
                       features["has_category"])
        if meta_factor > 0:
            w_meta += 0.10 * min(meta_factor, 3)
            w_dense -= 0.05 * min(meta_factor, 3)
            w_sparse -= 0.05 * min(meta_factor, 3)

        # Long descriptive queries -> increase dense weight
        if features["query_length"] > 6:
            w_dense += 0.10
            w_sparse -= 0.05
            w_meta -= 0.05

        # Negation -> increase sparse (keyword matching matters for negations)
        if features["has_negation"]:
            w_sparse += 0.10
            w_dense -= 0.10

        # Clip to valid range
        total = w_dense + w_sparse + w_meta
        if total > 0:
            w_dense /= total
            w_sparse /= total
            w_meta /= total
        else:
            w_dense, w_sparse, w_meta = self.default_dense, self.default_sparse, self.default_meta

        return max(0.0, w_dense), max(0.0, w_sparse), max(0.0, w_meta)

    def _extract_features(self, query: str) -> dict:
        """Extract heuristic features from query string."""
        tokens = query.lower().split()

        features = {
            "query_length": len(tokens),
            "has_temporal": int(any(w in self.TEMPORAL_KEYWORDS for w in tokens)),
            "has_location": int(any(w in self.LOCATION_KEYWORDS for w in tokens)),
            "has_category": int(any(w in self.CATEGORY_KEYWORDS for w in tokens)),
            "has_negation": int(any(w in {"not", "no", "without", "except", "除了", "不", "没有"} for w in tokens)),
            "avg_idf": 0.0,
        }

        # Compute average IDF using BM25 index if available
        if self.bm25 is not None:
            idfs = []
            for token in tokens:
                df = self.bm25.doc_freq.get(token, 0)
                if df > 0:
                    idf = np.log((self.bm25.num_docs - df + 0.5) / (df + 0.5) + 1.0)
                    idfs.append(idf)
            features["avg_idf"] = float(np.mean(idfs)) if idfs else 0.0

        return features
