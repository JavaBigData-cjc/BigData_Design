"""
CLIP encoder for extracting text and image embeddings.
Supports both OpenAI CLIP and Chinese-CLIP variants.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


class CLIPEncoder:
    """Extract L2-normalized embeddings from CLIP model for both modalities."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 device: str = "cuda", normalize: bool = True):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.normalize = normalize

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

        self._dim = self.model.config.projection_dim

    @property
    def dim(self) -> int:
        return self._dim

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image],
                      batch_size: int = 64) -> np.ndarray:
        """Extract image embeddings. Returns (N, dim) numpy array."""
        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            inputs = self.processor(
                images=batch, return_tensors="pt", padding=True
            ).to(self.device)

            emb = self.model.get_image_features(**inputs)
            # Handle BaseModelOutputWithPooling in newer transformers
            if hasattr(emb, "pooler_output"):
                emb = emb.pooler_output
            if self.normalize:
                emb = F.normalize(emb, p=2, dim=-1)

            all_embeddings.append(emb.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    @torch.no_grad()
    def encode_texts(self, texts: list[str],
                     batch_size: int = 64) -> np.ndarray:
        """Extract text embeddings. Returns (N, dim) numpy array."""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.processor(
                text=batch, return_tensors="pt", padding=True,
                truncation=True, max_length=77
            ).to(self.device)

            emb = self.model.get_text_features(**inputs)
            if hasattr(emb, "pooler_output"):
                emb = emb.pooler_output
            if self.normalize:
                emb = F.normalize(emb, p=2, dim=-1)

            all_embeddings.append(emb.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def encode_single(self, text: str = None,
                      image: Image.Image = None) -> np.ndarray:
        """Encode a single text or image query."""
        if text is not None:
            return self.encode_texts([text])[0]
        elif image is not None:
            return self.encode_images([image])[0]
        else:
            raise ValueError("Must provide either text or image")

    def similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix between two embedding sets."""
        if self.normalize:
            return emb_a @ emb_b.T
        else:
            a_norm = emb_a / np.linalg.norm(emb_a, axis=-1, keepdims=True)
            b_norm = emb_b / np.linalg.norm(emb_b, axis=-1, keepdims=True)
            return a_norm @ b_norm.T

    def unload(self):
        """Free GPU memory by moving model to CPU and clearing cache."""
        self.model.to("cpu")
        torch.cuda.empty_cache()

    def reload(self):
        """Reload model to GPU."""
        self.model.to(self.device)
