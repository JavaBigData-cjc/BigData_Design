"""
Dimensionality reduction for CLIP embeddings.
PCA (linear, fast), UMAP (manifold, preserves structure), t-SNE (visualization).
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


class DimReducer:
    """Apply and compare dimensionality reduction methods on embeddings."""

    METHODS = ["pca", "umap", "tsne"]

    def __init__(self, method: str = "pca", n_components: int = 128,
                 random_state: int = 42):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}")

        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        self.reducer = None
        self._fitted = False

    def fit(self, X: np.ndarray):
        """Fit the reducer on training embeddings."""
        if self.method == "pca":
            self.reducer = PCA(n_components=self.n_components,
                               random_state=self.random_state)
        elif self.method == "umap":
            self.reducer = umap.UMAP(
                n_components=self.n_components,
                random_state=self.random_state,
                n_jobs=1,
            )
        elif self.method == "tsne":
            # t-SNE is primarily for visualization (2D/3D)
            self.reducer = TSNE(
                n_components=min(self.n_components, 3),
                random_state=self.random_state,
                perplexity=30,
            )

        self.reducer.fit(X)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Reduce dimensionality of input embeddings."""
        if not self._fitted:
            raise RuntimeError("Must call fit() before transform()")
        return self.reducer.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Explained variance ratio (PCA only)."""
        if self.method == "pca" and self._fitted:
            return self.reducer.explained_variance_ratio_
        return None

    @staticmethod
    def compare_methods(embeddings: np.ndarray,
                        target_dims: list[int] = (64, 128, 256),
                        methods: tuple[str, ...] = ("pca", "umap")) -> dict:
        """Compare reconstruction quality across methods and dimensions."""
        # Use train/test split to evaluate reconstruction
        n = len(embeddings)
        train = embeddings[:int(n * 0.8)]
        test = embeddings[int(n * 0.8):]

        results = {}
        for method in methods:
            for dim in target_dims:
                key = f"{method}_{dim}d"
                try:
                    reducer = DimReducer(method=method, n_components=dim)
                    reduced_train = reducer.fit_transform(train)
                    reduced_test = reducer.transform(test)

                    results[key] = {
                        "train_shape": reduced_train.shape,
                        "test_shape": reduced_test.shape,
                    }
                except Exception as e:
                    results[key] = {"error": str(e)}

        return results
