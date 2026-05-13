"""
Metadata filtering for augmented retrieval.
Supports time-based, category-based, and custom attribute filters.
"""

import re
from typing import Optional

import numpy as np
import pandas as pd


class MetadataFilter:
    """Apply structured metadata constraints to retrieval results."""

    SUPPORTED_OPERATORS = ["eq", "neq", "gt", "gte", "lt", "lte", "in", "contains"]

    def __init__(self, metadata_df: Optional[pd.DataFrame] = None):
        self.metadata_df = metadata_df  # indexed by document id

    def filter(self, metadata_df: pd.DataFrame,
               filters: list[dict]) -> np.ndarray:
        """Apply a list of filter conditions, return boolean mask.

        Each filter dict has: {"field": str, "op": str, "value": any}
        """
        mask = np.ones(len(metadata_df), dtype=bool)

        for f in filters:
            field = f["field"]
            op = f["op"]
            value = f["value"]

            if field not in metadata_df.columns:
                continue

            col = metadata_df[field]

            if op == "eq":
                mask &= (col == value)
            elif op == "neq":
                mask &= (col != value)
            elif op == "gt":
                mask &= (col > value)
            elif op == "gte":
                mask &= (col >= value)
            elif op == "lt":
                mask &= (col < value)
            elif op == "lte":
                mask &= (col <= value)
            elif op == "in":
                mask &= col.isin(value if isinstance(value, list) else [value])
            elif op == "contains":
                mask &= col.str.contains(str(value), na=False, regex=False)

        return mask

    def parse_query_metadata(self, query: str) -> list[dict]:
        """Extract metadata constraints from natural language query.

        Simple regex-based extraction of temporal, numeric, and categorical hints.
        """
        filters = []

        # Year extraction: "from 2023", "in 2024", "photos from 2022"
        year_match = re.search(r"(?:from|in|since|before|after)\s+(\d{4})", query.lower())
        if year_match:
            year = int(year_match.group(1))
            filters.append({"field": "year", "op": "eq", "value": year})

        # Category extraction: "outdoor", "indoor"
        if "outdoor" in query.lower() or "室外" in query:
            filters.append({"field": "scene", "op": "eq", "value": "outdoor"})
        if "indoor" in query.lower() or "室内" in query:
            filters.append({"field": "scene", "op": "eq", "value": "indoor"})

        return filters

    def compute_metadata_scores(self, query: str,
                                metadata_df: pd.DataFrame) -> np.ndarray:
        """Compute metadata match scores for each document.

        Returns scores in [0, 1] representing how well each doc matches
        the extracted metadata constraints.
        """
        filters = self.parse_query_metadata(query)
        if not filters:
            return np.ones(len(metadata_df))  # No metadata constraints

        mask = self.filter(metadata_df, filters)
        return mask.astype(np.float32)
