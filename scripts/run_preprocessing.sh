#!/bin/bash
# Run Spark preprocessing pipeline on COCO/Flickr30k data
# Usage: bash scripts/run_preprocessing.sh [coco|flickr30k]

set -e

DATASET="${1:-coco}"
echo "=== Running Spark Preprocessing for: $DATASET ==="

python -c "
from src.data.preprocess import SparkPreprocessor
from pathlib import Path

# Initialize Spark
spark_proc = SparkPreprocessor(
    app_name='CrossModalRAG_Preprocess',
    master='local[4]',
    driver_memory='4g'
)

if '$DATASET' == 'coco':
    # Load COCO annotations
    ann_path = 'data/raw/coco/annotations/captions_val2017.json'
    img_dir = 'data/raw/coco/val2017'

    if not Path(ann_path).exists():
        print(f'[WARN] Annotation file not found: {ann_path}')
        print('Run scripts/download_data.sh first')
        exit(1)

    df = spark_proc.load_coco_captions(ann_path)
    df = spark_proc.clean_captions(df, min_length=5)
    df = spark_proc.add_image_path(df, img_dir)

elif '$DATASET' == 'flickr30k':
    ann_path = 'data/raw/flickr30k/results.csv'
    img_dir = 'data/raw/flickr30k/flickr30k_images'

    if not Path(ann_path).exists():
        print(f'[WARN] Annotation file not found: {ann_path}')
        print('Download Flickr30k from Kaggle first')
        exit(1)

    df = spark_proc.load_flickr_captions(ann_path)
    df = spark_proc.clean_captions(df, min_length=5)

else:
    print(f'Unknown dataset: $DATASET')
    exit(1)

# Train/val/test split
train_df, val_df, test_df = spark_proc.stratified_split(df)

# Save as Parquet
spark_proc.save_parquet(train_df, f'data/processed/$DATASET/train.parquet')
spark_proc.save_parquet(val_df, f'data/processed/$DATASET/val.parquet')
spark_proc.save_parquet(test_df, f'data/processed/$DATASET/test.parquet')

print(f'[OK] Preprocessing complete')
print(f'  Train: {train_df.count()} rows')
print(f'  Val:   {val_df.count()} rows')
print(f'  Test:  {test_df.count()} rows')

spark_proc.stop()
"

echo "=== Done ==="
