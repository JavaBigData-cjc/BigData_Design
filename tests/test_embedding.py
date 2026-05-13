"""Tests for embedding module."""
import numpy as np
import pytest
from PIL import Image


class TestCLIPEncoder:
    """Test CLIP encoding functionality (requires model download)."""

    def test_encoder_init(self):
        from src.embedding.clip_encoder import CLIPEncoder
        encoder = CLIPEncoder(device="cpu")
        assert encoder.dim == 512
        assert encoder.model_name == "openai/clip-vit-base-patch32"

    def test_encode_text(self):
        from src.embedding.clip_encoder import CLIPEncoder
        encoder = CLIPEncoder(device="cpu")
        texts = ["a dog running in a field", "a cat sleeping on a couch"]
        embs = encoder.encode_texts(texts, batch_size=1)
        assert embs.shape == (2, 512)
        # Check normalization
        norms = np.linalg.norm(embs, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_encode_image(self):
        from src.embedding.clip_encoder import CLIPEncoder
        encoder = CLIPEncoder(device="cpu")
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        images = [img, img]
        embs = encoder.encode_images(images, batch_size=1)
        assert embs.shape == (2, 512)


class TestDimReducer:
    """Test dimensionality reduction."""

    def test_pca_fit_transform(self):
        from src.embedding.dim_reduction import DimReducer
        X = np.random.randn(100, 512).astype(np.float32)
        reducer = DimReducer(method="pca", n_components=128)
        X_reduced = reducer.fit_transform(X)
        assert X_reduced.shape == (100, 128)

    def test_umap_fit_transform(self):
        from src.embedding.dim_reduction import DimReducer
        X = np.random.randn(100, 512).astype(np.float32)
        reducer = DimReducer(method="umap", n_components=64)
        X_reduced = reducer.fit_transform(X)
        assert X_reduced.shape == (100, 64)
