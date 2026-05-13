"""
PyTorch Dataset classes for cross-modal retrieval.
Supports COCO, Flickr30k and generic image-text pair datasets.
"""

import os
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class CrossModalDataset(Dataset):
    """Generic dataset for image-text pairs used in cross-modal retrieval."""

    def __init__(self, df: pd.DataFrame,
                 image_dir: str,
                 image_col: str = "image_path",
                 caption_col: str = "caption",
                 image_id_col: str = "image_id",
                 transform: Optional[Callable] = None):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.image_col = image_col
        self.caption_col = caption_col
        self.image_id_col = image_id_col
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # Build full image path
        img_path = row[self.image_col]
        if not os.path.isabs(img_path):
            img_path = str(self.image_dir / img_path)

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            image = Image.new("RGB", (224, 224))

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "caption": row[self.caption_col],
            "image_id": str(row[self.image_id_col]),
            "image_path": img_path,
        }


class ImageOnlyDataset(Dataset):
    """Dataset that yields only images (for batch embedding extraction)."""

    def __init__(self, image_paths: list[str],
                 transform: Optional[Callable] = None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
        except FileNotFoundError:
            image = Image.new("RGB", (224, 224))

        if self.transform:
            image = self.transform(image)

        return {"image": image, "path": path}


class TextOnlyDataset(Dataset):
    """Dataset that yields only captions (for batch embedding extraction)."""

    def __init__(self, captions: list[str]):
        self.captions = captions

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, idx: int) -> str:
        return self.captions[idx]


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate for mixed image-text batches."""
    images = torch.stack([item["image"] for item in batch])
    captions = [item["caption"] for item in batch]
    image_ids = [item["image_id"] for item in batch]

    return {
        "image": images,
        "caption": captions,
        "image_id": image_ids,
    }
