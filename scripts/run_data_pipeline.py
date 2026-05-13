"""
End-to-end data pipeline runner for cross-modal RAG system.

Stages:
  1. Generate demo data (or load real dataset)
  2. Spark preprocessing -> Parquet files
  3. CLIP embedding extraction -> .npy files
  4. Vector index construction -> .index / .pkl files

Architecture:
  data/raw/          <- Original CSV/JSON/images
  data/processed/    <- Spark Parquet (cleaned, normalized)
  data/embeddings/   <- CLIP vectors (.npy) + metadata (.parquet)
  data/indexes/      <- FAISS/HNSW/LSH/Annoy index files
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_flickr30k_data(data_dir: str = "data/raw/flickr30k"):
    """Load real Flickr30k dataset (images + captions).

    Expected structure (from AutoGluon S3 mirror):
      flickr30k_processed/
        ├── train.csv     (145K rows: ,caption,image)
        ├── val.csv       (5K rows)
        ├── test.csv      (5K rows)
        └── images/       (31,783 .jpg files)
    """
    import pandas as pd

    data_dir = Path(data_dir)
    processed_dir = data_dir / "flickr30k_processed"

    if not processed_dir.exists():
        print(f"  [WARN] flickr30k_processed not found at {processed_dir}")
        print(f"  Run: python scripts/download_flickr30k.py")
        return None

    # Load train.csv (main training data)
    csv_path = processed_dir / "train.csv"
    if not csv_path.exists():
        print(f"  [WARN] train.csv not found")
        return None

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} captions from {csv_path}")
    print(f"  Columns: {list(df.columns)}")

    # Build absolute image paths: images/xxx.jpg -> data/raw/flickr30k/flickr30k_processed/images/xxx.jpg
    img_dir = processed_dir / "images"

    if "image" in df.columns:
        # Paths are like "images/1000092795.jpg"
        df["image_path"] = df["image"].apply(
            lambda x: str(img_dir / Path(x).name)
        )
    elif "image_name" in df.columns:
        df["image_path"] = df["image_name"].apply(
            lambda x: str(img_dir / x) if pd.notna(x) else ""
        )

    # Filter rows where image files actually exist
    valid_mask = df["image_path"].apply(lambda p: Path(p).exists())
    if valid_mask.sum() < len(df):
        print(f"  Matched images: {valid_mask.sum()}/{len(df)}")
        df = df[valid_mask]

    # Standardize column names
    if "caption" not in df.columns and "comment" in df.columns:
        df["caption"] = df["comment"]
    elif "caption" not in df.columns and "description" in df.columns:
        df["caption"] = df["description"]

    # Use first N for fast development iteration
    if len(df) > 5000:
        print(f"  [dev mode] Sampling 5000 from {len(df)} rows (use --full for all)")
        df = df.sample(n=5000, random_state=42).reset_index(drop=True)

    print(f"  Final dataset: {len(df)} rows, {df['image_path'].nunique()} unique images")
    return df


def stage1_load_data(source: str = "demo"):
    """Load or generate data for the pipeline."""
    print("=" * 60)
    print(f"STAGE 1: Load Data (source={source})")
    print("=" * 60)

    if source == "flickr30k":
        df = load_flickr30k_data("data/raw/flickr30k")
        if df is None:
            print("  [FALLBACK] Using demo data instead")
            source = "demo"

    if source == "demo":
        from scripts.generate_demo_data import generate_demo_dataset
        df = generate_demo_dataset("data/demo", num_images_per_scene=10)

    if df is None or len(df) == 0:
        raise RuntimeError("Failed to load any data")

    print(f"  [OK] {len(df)} samples loaded")
    print(f"  Columns: {list(df.columns)}")
    print()
    return df


def stage2_spark_preprocessing(input_csv: str = None):
    """Run Spark preprocessing pipeline."""
    print("=" * 60)
    print("STAGE 2: Spark Preprocessing -> Parquet")
    print("=" * 60)

    import pandas as pd
    from src.data.preprocess import SparkPreprocessor

    if input_csv is None:
        input_csv = "data/demo/annotations.csv"

    if not Path(input_csv).exists():
        print(f"  [WARN] Input not found: {input_csv}, skipping Spark step")
        print(f"  Using pandas DataFrame directly instead.\n")
        return pd.read_csv("data/demo/annotations.csv")

    spark_proc = SparkPreprocessor(
        app_name="CrossModalRAG_Demo",
        master="local[2]",
        driver_memory="2g",
    )

    # Load CSV into Spark
    pdf = pd.read_csv(input_csv)
    df = spark_proc.spark.createDataFrame(pdf)

    # Clean captions
    df = spark_proc.clean_captions(df, min_length=3)

    # Save as Parquet
    output_path = "data/processed/demo.parquet"
    spark_proc.save_parquet(df, output_path)

    count = df.count()
    print(f"  [OK] {count} rows saved to {output_path}\n")

    spark_proc.stop()
    return pdf  # Return pandas for next stage


def stage3_extract_embeddings(df=None):
    """Extract CLIP embeddings from images and captions."""
    print("=" * 60)
    print("STAGE 3: CLIP Embedding Extraction -> .npy")
    print("=" * 60)

    if df is None:
        import pandas as pd
        csv_path = "data/demo/annotations.csv"
        if Path(csv_path).exists():
            df = pd.read_csv(csv_path)
        else:
            print("  [WARN] No data found, skipping embedding extraction")
            return

    from src.embedding.clip_encoder import CLIPEncoder
    from src.embedding.batch_extractor import BatchExtractor

    encoder = CLIPEncoder(
        model_name="openai/clip-vit-base-patch32",
        device="cuda",
        normalize=True,
    )
    print(f"  Model: {encoder.model_name}, Dim: {encoder.dim}")

    extractor = BatchExtractor(encoder, output_dir="data/embeddings")
    result = extractor.extract_from_dataframe(
        df, image_col="image_path", caption_col="caption",
        batch_size=32, save=True,
    )

    print(f"  Image embeddings: {result['image_embeddings'].shape}")
    print(f"  Text embeddings:  {result['text_embeddings'].shape}")
    print(f"  [OK] Saved to data/embeddings/\n")

    encoder.unload()
    return result


def stage4_build_indexes():
    """Build all four ANN indexes."""
    print("=" * 60)
    print("STAGE 4: Build Vector Indexes -> .index / .pkl")
    print("=" * 60)

    import numpy as np
    emb_path = Path("data/embeddings/embeddings_image.npy")
    if not emb_path.exists():
        print("  [WARN] No embeddings found, skipping index building")
        return

    embeddings = np.load(emb_path)
    dim = embeddings.shape[1]
    print(f"  Loaded {len(embeddings)} x {dim} vectors")

    # 1. FAISS IVF-PQ
    from src.indexing.faiss_index import FAISSIndex
    faiss_idx = FAISSIndex(dim=dim, index_type="ivfpq")
    t = faiss_idx.timed_build(embeddings)
    faiss_idx.save("data/indexes/faiss_ivfpq")
    print(f"  [1/4] FAISS IVF-PQ: {t:.2f}s build time")

    # 2. Manual HNSW
    from src.indexing.hnsw_manual import ManualHNSW
    hnsw = ManualHNSW(dim=dim, M=8, ef_construction=100)
    t = hnsw.timed_build(embeddings)
    hnsw.save("data/indexes/hnsw_manual.pkl")
    print(f"  [2/4] Manual HNSW: {t:.2f}s build time")

    # 3. LSH
    from src.indexing.lsh import CosineLSH
    lsh = CosineLSH(dim=dim, n_tables=5, n_hashes=8)
    t = lsh.timed_build(embeddings)
    lsh.save("data/indexes/lsh.pkl")
    print(f"  [3/4] LSH: {t:.2f}s build time")

    # 4. Annoy
    from src.indexing.annoy_index import AnnoyIndexWrapper
    annoy = AnnoyIndexWrapper(dim=dim, n_trees=10)
    t = annoy.timed_build(embeddings)
    annoy.save("data/indexes/annoy")
    print(f"  [4/4] Annoy: {t:.2f}s build time")

    print(f"  [OK] All indexes saved to data/indexes/\n")


def show_architecture():
    """Print the data storage architecture diagram."""
    print("=" * 60)
    print("DATA STORAGE ARCHITECTURE")
    print("=" * 60)
    print("""
┌─────────────────────────────────────────────────────────────────┐
│                      DATA STORAGE LAYERS                        │
└─────────────────────────────────────────────────────────────────┘

[Layer 0: Raw Data]          data/raw/    &  data/demo/
  ├── images/                 *.jpg        (原始图片, PIL decode)
  ├── annotations.json        JSON         (标注列表, 嵌套结构)
  └── annotations.csv         CSV          (扁平表, 快速预览)
       │
       │  Spark Preprocessing  ──────────────────────┐
       │  (clean, normalize,   │                      │
       │   stratified split)   │                      │
       ▼                       ▼                      ▼
[Layer 1: Processed Data]   data/processed/
  ├── train.parquet           Parquet      (列存, 压缩, 分区)
  ├── val.parquet             Parquet      (列存, 压缩, 分区)
  └── test.parquet            Parquet      (列存, 压缩, 分区)
       │
       │  CLIP Encoder (GPU)   ──────────────────────┐
       │  (ViT-B/32, 512d)     │                      │
       ▼                       ▼                      ▼
[Layer 2: Embeddings]       data/embeddings/
  ├── embeddings_image.npy    .npy         (N x 512, float32)
  ├── embeddings_text.npy     .npy         (N x 512, float32)
  ├── embeddings_meta.parquet  Parquet     (image_path 映射)
  └── embeddings_captions.parquet Parquet  (caption 文本)
       │
       │  ANN Index Builder    ──────────────────────┐
       │  (FAISS/HNSW/LSH/Annoy)                      │
       ▼                       ▼                      ▼
[Layer 3: Vector Indexes]   data/indexes/
  ├── faiss_ivfpq.faiss       FAISS binary (IVF+PQ, 倒排索引)
  ├── faiss_ivfpq.meta.pkl    Pickle       (元数据: dim, nlist, id_map)
  ├── hnsw_manual.pkl         Pickle       (图节点+邻接表, 多层)
  ├── lsh.pkl                 Pickle       (哈希表+随机投影矩阵)
  ├── lsh_config.pkl          Pickle       (LSH参数: n_tables, n_hashes)
  ├── annoy.ann               Annoy binary (多棵随机投影树)
  └── annoy.meta.pkl          Pickle       (元数据: dim, n_trees)

=== STORAGE FORMAT RATIONALE ===

  Format    │ Use Case                         │ Why
  ──────────┼──────────────────────────────────┼──────────────────────
  Parquet   │ Processed tabular data            │ 列存,压缩率高(5-10x),
            │ (annotations, metadata)          │ Spark原生支持,Schema进化
  .npy      │ Dense embedding vectors          │ 无开销直接mmap,与numpy无缝
            │ (N x 512 float32)                │ 读写,比Parquet快10x+
  FAISS     │ IVF-PQ compressed index          │ C++ mmap加载,支持GPU/CPU
  .index    │                                  │ 乘积量化压缩8-16x
  .pkl      │ Graph/hash-structured indexes    │ 图结构和哈希表不适合
            │ (HNSW, LSH)                      │ 矩阵格式,用pickle序列化
  Annoy     │ Multi-tree index format          │ 自研二进制格式,支持mmap
  .ann      │                                  │ 启动快,内存映射

=== DATA FLOW ===

  Query ──> CLIP Text Encoder ──> [512d] ──> FAISS/HNSW/LSH/Annoy
                                                 │
                                                 ▼
                                          Top-K IDs ──> Parquet Lookup
                                                            │
                                                            ▼
                                                    Image Paths + Captions
""")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Cross-Modal RAG Data Pipeline")
    ap.add_argument("--stage", type=int, default=0,
                    help="Run specific stage (0=all, 1=load, 2=spark, 3=embed, 4=index)")
    ap.add_argument("--source", type=str, default="demo",
                    choices=["demo", "flickr30k", "coco"],
                    help="Data source (default: demo)")
    ap.add_argument("--show-arch", action="store_true",
                    help="Print data storage architecture diagram")
    args = ap.parse_args()

    if args.show_arch or args.stage == 0:
        show_architecture()

    if args.stage == 0 or args.stage == 1:
        df = stage1_load_data(args.source)
    else:
        df = None

    if args.stage == 0 or args.stage == 2:
        stage2_spark_preprocessing()

    if args.stage == 0 or args.stage == 3:
        stage3_extract_embeddings(df)

    if args.stage == 0 or args.stage == 4:
        stage4_build_indexes()

    print("=" * 60)
    print("DATA PIPELINE COMPLETE")
    print("=" * 60)
