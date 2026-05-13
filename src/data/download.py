"""
Dataset download utilities for COCO 2017 and Flickr30k.
"""

import os
import subprocess
import zipfile
from pathlib import Path


class DataDownloader:
    """Download and extract benchmark datasets for cross-modal retrieval."""

    COCO_URLS = {
        "train2017": "http://images.cocodataset.org/zips/train2017.zip",
        "val2017": "http://images.cocodataset.org/zips/val2017.zip",
        "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    }

    FLICKR30K_URL = "https://github.com/BryanPlummer/flickr30k_entities/raw/master/annotations.zip"

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url: str, dest_name: str) -> Path:
        """Download a file with progress display."""
        dest_path = self.data_dir / dest_name
        if dest_path.exists():
            print(f"[skip] {dest_name} already exists")
            return dest_path

        print(f"[download] {url} -> {dest_path}")
        subprocess.run(
            ["wget", "-O", str(dest_path), url],
            check=True
        )
        return dest_path

    def extract_zip(self, zip_path: Path, extract_to: str = ""):
        """Extract a zip file."""
        target = self.data_dir / extract_to if extract_to else self.data_dir
        print(f"[extract] {zip_path} -> {target}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target)

    def download_coco(self):
        """Download COCO 2017 train/val images and annotations."""
        for name, url in self.COCO_URLS.items():
            zip_path = self.download_file(url, f"{name}.zip")
            self.extract_zip(zip_path)

    def download_flickr30k(self):
        """Download Flickr30k dataset (requires Kaggle credentials)."""
        print("Flickr30k download requires Kaggle API.")
        print("Manual steps:")
        print("  1. kaggle datasets download hsankesara/flickr-image-dataset")
        print("  2. Extract to data/raw/flickr30k/")

    def prepare_structure(self):
        """Create the standard directory structure for processed data."""
        dirs = [
            "data/raw/coco",
            "data/raw/flickr30k",
            "data/processed/coco",
            "data/processed/flickr30k",
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
        print("[ok] Directory structure ready")
