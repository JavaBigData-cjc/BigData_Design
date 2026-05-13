#!/bin/bash
# Extract CLIP embeddings from preprocessed data
# Usage: bash scripts/extract_embeddings.sh [coco|flickr30k]

set -e

DATASET="${1:-coco}"
echo "=== Extracting CLIP Embeddings for: $DATASET ==="

python -c "
from src.embedding.clip_encoder import CLIPEncoder
from src.embedding.batch_extractor import BatchExtractor
import pandas as pd
from pathlib import Path

# Check processed data exists
parquet_path = f'data/processed/$DATASET/train.parquet'
if not Path(parquet_path).exists():
    print(f'[WARN] Processed data not found: {parquet_path}')
    print('Run scripts/run_preprocessing.sh first')
    exit(1)

print('[1/3] Loading processed data...')
df = pd.read_parquet(parquet_path)
print(f'  Loaded {len(df)} rows')

print('[2/3] Initializing CLIP encoder (GPU)...')
encoder = CLIPEncoder(
    model_name='openai/clip-vit-base-patch32',
    device='cuda',
    normalize=True
)
print(f'  Model: {encoder.model_name}, Dim: {encoder.dim}')

print('[3/3] Extracting embeddings...')
extractor = BatchExtractor(encoder, output_dir='data/embeddings')

# For large datasets, sample first
sample_df = df.head(5000) if len(df) > 5000 else df
print(f'  Processing {len(sample_df)} samples...')

result = extractor.extract_from_dataframe(
    sample_df,
    image_col='image_path',
    caption_col='caption',
    batch_size=64,
    save=True
)

print(f'[OK] Embeddings extracted and saved')
print(f'  Image embeddings: {result[\"image_embeddings\"].shape}')
print(f'  Text embeddings:  {result[\"text_embeddings\"].shape}')

# Free GPU memory
encoder.unload()
"

echo "=== Done ==="
