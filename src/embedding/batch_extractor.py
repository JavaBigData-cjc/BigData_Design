"""
Batch embedding extraction using Spark for distributed processing.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from .clip_encoder import CLIPEncoder


class BatchExtractor:
    """Extract and save embeddings in batch mode, optionally with Spark."""

    def __init__(self, encoder: CLIPEncoder,
                 output_dir: str = "data/embeddings"):
        self.encoder = encoder
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_from_dataframe(self, df: pd.DataFrame,
                               image_col: str = "image_path",
                               caption_col: str = "caption",
                               batch_size: int = 64,
                               save: bool = True) -> dict:
        """Extract image and text embeddings from a pandas DataFrame."""
        from src.data.dataset import ImageOnlyDataset

        # Extract image embeddings
        unique_images = df[image_col].unique().tolist()
        image_dataset = ImageOnlyDataset(unique_images)
        # Process sequentially for memory
        image_embeddings = self.encoder.encode_images(
            [item["image"] for item in image_dataset],
            batch_size=batch_size
        )

        # Extract text embeddings
        captions = df[caption_col].tolist()
        text_embeddings = self.encoder.encode_texts(
            captions, batch_size=batch_size
        )

        result = {
            "image_paths": unique_images,
            "image_embeddings": image_embeddings,
            "captions": captions,
            "text_embeddings": text_embeddings,
        }

        if save:
            self.save_embeddings(result)

        return result

    def save_embeddings(self, data: dict, prefix: str = "embeddings"):
        """Save embeddings as numpy arrays and captions as CSV."""
        np.save(self.output_dir / f"{prefix}_image.npy",
                data["image_embeddings"])
        np.save(self.output_dir / f"{prefix}_text.npy",
                data["text_embeddings"])

        meta_df = pd.DataFrame({
            "image_path": data["image_paths"],
        })
        meta_df.to_parquet(self.output_dir / f"{prefix}_meta.parquet")

        caption_df = pd.DataFrame({
            "caption": data["captions"],
        })
        caption_df.to_parquet(self.output_dir / f"{prefix}_captions.parquet")

        print(f"[save] Embeddings saved to {self.output_dir}/")

    def load_embeddings(self, prefix: str = "embeddings") -> dict:
        """Load saved embeddings from disk."""
        return {
            "image_embeddings": np.load(
                self.output_dir / f"{prefix}_image.npy"),
            "text_embeddings": np.load(
                self.output_dir / f"{prefix}_text.npy"),
            "meta": pd.read_parquet(
                self.output_dir / f"{prefix}_meta.parquet"),
            "captions": pd.read_parquet(
                self.output_dir / f"{prefix}_captions.parquet"),
        }
