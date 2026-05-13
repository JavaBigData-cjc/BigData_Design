"""
BM25 sparse retrieval for keyword-based text matching.
Implements standard Okapi BM25 scoring.
"""

import pickle
from typing import Optional

import numpy as np


class BM25Index:
    """Okapi BM25 sparse retrieval index for captions."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = None
        self.doc_lengths = None
        self.avg_doc_length = None
        self.doc_freq = {}       # term -> number of docs containing it
        self.num_docs = 0
        self._built = False

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace tokenizer (can be replaced with jieba for Chinese)."""
        return text.lower().split()

    def build(self, documents: list[str]):
        """Build BM25 index from a list of documents."""
        self.corpus = [self._tokenize(doc) for doc in documents]
        self.num_docs = len(self.corpus)
        self.doc_lengths = np.array([len(doc) for doc in self.corpus])
        self.avg_doc_length = float(self.doc_lengths.mean())

        self.doc_freq = {}
        for doc in self.corpus:
            seen = set()
            for term in doc:
                if term not in seen:
                    self.doc_freq[term] = self.doc_freq.get(term, 0) + 1
                    seen.add(term)

        self._built = True

    def _idf(self, term: str) -> float:
        """Compute IDF for a term."""
        df = self.doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        return np.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)

    def get_scores(self, query: str) -> np.ndarray:
        """Compute BM25 scores for query against all documents."""
        query_terms = self._tokenize(query)
        scores = np.zeros(self.num_docs)

        for term in query_terms:
            idf = self._idf(term)
            if idf == 0:
                continue

            for i, doc in enumerate(self.corpus):
                tf = doc.count(term)
                if tf == 0:
                    continue
                doc_len_norm = self.doc_lengths[i] / self.avg_doc_length
                score = idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len_norm))
                scores[i] += score

        return scores

    def search(self, query: str, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Search and return (document_ids, scores)."""
        scores = self.get_scores(query)
        top_k_idx = np.argsort(-scores)[:k]
        return top_k_idx, scores[top_k_idx]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "k1": self.k1, "b": self.b,
                "corpus": self.corpus,
                "doc_lengths": self.doc_lengths,
                "avg_doc_length": self.avg_doc_length,
                "doc_freq": self.doc_freq,
                "num_docs": self.num_docs,
            }, f)

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.k1 = data["k1"]
        obj.b = data["b"]
        obj.corpus = data["corpus"]
        obj.doc_lengths = data["doc_lengths"]
        obj.avg_doc_length = data["avg_doc_length"]
        obj.doc_freq = data["doc_freq"]
        obj.num_docs = data["num_docs"]
        obj._built = True
        return obj
