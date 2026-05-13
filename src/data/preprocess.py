"""
Spark-based data preprocessing pipeline.
Handles dataset loading, cleaning, normalization, and train/test splitting.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, length, regexp_replace, lower, rand
from pyspark.sql.types import StringType, StructType, StructField, IntegerType


class SparkPreprocessor:
    """Spark-based preprocessing for cross-modal datasets (COCO, Flickr30k)."""

    def __init__(self, app_name: str = "CrossModalRAG", master: str = "local[4]",
                 driver_memory: str = "4g"):
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .master(master) \
            .config("spark.driver.memory", driver_memory) \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()

    def load_coco_captions(self, annotation_path: str) -> DataFrame:
        """Load COCO-format caption annotations JSON and flatten."""
        df = self.spark.read.option("multiline", "true") \
            .json(annotation_path)

        # Flatten nested annotations array
        if "annotations" in df.columns:
            df = df.selectExpr("explode(annotations) as ann") \
                .select(
                    col("ann.image_id").alias("image_id"),
                    col("ann.caption").alias("caption"),
                    col("ann.id").alias("caption_id")
                )
        return df

    def load_flickr_captions(self, caption_path: str) -> DataFrame:
        """Load Flickr30k-format captions (CSV or tab-separated)."""
        df = self.spark.read \
            .option("header", "true") \
            .option("delimiter", "\t") \
            .csv(caption_path)
        return df

    def clean_captions(self, df: DataFrame, caption_col: str = "caption",
                       min_length: int = 5) -> DataFrame:
        """Clean captions: lowercase, remove special chars, filter short ones."""
        df_cleaned = df \
            .withColumn(caption_col, lower(col(caption_col))) \
            .withColumn(caption_col, regexp_replace(col(caption_col),
                        r"[^a-zA-Z0-9一-鿿\s.,!?']", "")) \
            .filter(length(col(caption_col)) > min_length)
        return df_cleaned

    def add_image_path(self, df: DataFrame, image_dir: str,
                       image_id_col: str = "image_id",
                       extension: str = ".jpg") -> DataFrame:
        """Add full image path column based on image_id."""
        df = df.withColumn("image_path",
                           regexp_replace(col(image_id_col), r"^",
                                          f"{image_dir}/"))
        return df.withColumn("image_path",
                             regexp_replace(col("image_path"), r"$", extension))

    def stratified_split(self, df: DataFrame, val_ratio: float = 0.2,
                         test_ratio: float = 0.1, seed: int = 42) -> Tuple[DataFrame, ...]:
        """Split into train/val/test with stratified sampling by image_id."""
        # Group by image_id to keep all captions of one image together
        images = df.select("image_id").distinct()

        train_images, val_images, test_images = images.randomSplit(
            [1.0 - val_ratio - test_ratio, val_ratio, test_ratio], seed=seed
        )

        train_df = df.join(train_images, "image_id", "inner")
        val_df = df.join(val_images, "image_id", "inner")
        test_df = df.join(test_images, "image_id", "inner")

        return train_df, val_df, test_df

    def save_parquet(self, df: DataFrame, output_path: str):
        """Save DataFrame as Parquet files."""
        df.write.mode("overwrite").parquet(output_path)

    def load_parquet(self, input_path: str) -> DataFrame:
        """Load Parquet files into DataFrame."""
        return self.spark.read.parquet(input_path)

    @staticmethod
    def to_numpy(df: DataFrame, columns: list[str]) -> Tuple[np.ndarray, ...]:
        """Convert Spark DataFrame columns to numpy arrays (for small datasets)."""
        pdf = df.select(columns).toPandas()
        return tuple(pdf[col].to_numpy() for col in columns)

    def stop(self):
        self.spark.stop()
