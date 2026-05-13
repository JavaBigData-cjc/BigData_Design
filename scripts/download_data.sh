#!/bin/bash
# Download benchmark datasets for cross-modal retrieval
# Usage: bash scripts/download_data.sh

set -e

echo "=== Downloading Cross-Modal Retrieval Datasets ==="

DATA_DIR="data/raw"
mkdir -p "$DATA_DIR"

# COCO 2017 Val images (smaller, ~1GB)
echo "[1/3] Downloading COCO 2017 validation images..."
if [ ! -f "$DATA_DIR/val2017.zip" ]; then
    wget -O "$DATA_DIR/val2017.zip" http://images.cocodataset.org/zips/val2017.zip
    echo "  -> Extracting..."
    unzip -q "$DATA_DIR/val2017.zip" -d "$DATA_DIR/coco/"
else
    echo "  -> Already downloaded, skipping"
fi

# COCO 2017 annotations
echo "[2/3] Downloading COCO 2017 annotations..."
if [ ! -f "$DATA_DIR/annotations_trainval2017.zip" ]; then
    wget -O "$DATA_DIR/annotations_trainval2017.zip" http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    echo "  -> Extracting..."
    unzip -q "$DATA_DIR/annotations_trainval2017.zip" -d "$DATA_DIR/coco/"
else
    echo "  -> Already downloaded, skipping"
fi

echo "[3/3] Setting up directory structure..."
mkdir -p data/processed/coco data/processed/flickr30k
mkdir -p data/embeddings data/indexes

echo "=== Done! ==="
echo "  COCO: data/raw/coco/"
echo "  For Flickr30k, use Kaggle: kaggle datasets download hsankesara/flickr-image-dataset"
