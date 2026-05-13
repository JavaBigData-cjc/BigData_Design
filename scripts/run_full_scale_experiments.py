"""
Full-scale experiment runner: Flickr30k (29K images, 145K captions).
Runs E1 (ANN comparison) and E2 (dimensionality reduction).
"""
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_flickr30k_full():
    """Load full Flickr30k dataset with real image paths."""
    processed_dir = Path("data/raw/flickr30k/flickr30k_processed")
    csv_path = processed_dir / "train.csv"
    img_dir = processed_dir / "images"

    df = pd.read_csv(csv_path)
    # Columns: Unnamed: 0, caption, image
    df["image_path"] = df["image"].apply(
        lambda x: str(img_dir / Path(x).name)
    )
    # Filter to images that exist on disk
    valid = df["image_path"].apply(lambda p: Path(p).exists())
    df = df[valid].reset_index(drop=True)
    print(f"[load] {len(df)} captions, {df['image_path'].nunique()} unique images")
    return df


def extract_embeddings(df, output_subdir="flickr30k_full"):
    """Extract CLIP embeddings for all images and captions."""
    from src.embedding.clip_encoder import CLIPEncoder
    from PIL import Image

    out_dir = Path("data/embeddings") / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check cache
    img_path = out_dir / "image_embeddings.npy"
    txt_path = out_dir / "text_embeddings.npy"
    if img_path.exists() and txt_path.exists():
        print(f"[embed] Using cached embeddings from {out_dir}")
        return {
            "image_embeddings": np.load(img_path),
            "text_embeddings": np.load(txt_path),
            "image_paths": pd.read_parquet(out_dir / "image_map.parquet")["image_path"].tolist(),
            "captions": pd.read_parquet(out_dir / "caption_map.parquet")["caption"].tolist(),
            "df": pd.read_parquet(out_dir / "df.parquet"),
        }

    print("[embed] Loading CLIP encoder...")
    encoder = CLIPEncoder(model_name="openai/clip-vit-base-patch32", device="cuda", normalize=True)
    print(f"[embed] Model: {encoder.model_name}, Dim: {encoder.dim}")

    # Extract image embeddings (unique images only)
    unique_paths = sorted(df["image_path"].unique())
    print(f"[embed] Extracting image embeddings for {len(unique_paths)} images...")
    all_img_emb = []
    batch_size = 64
    for i in range(0, len(unique_paths), batch_size):
        batch_paths = unique_paths[i:i + batch_size]
        images = []
        for p in batch_paths:
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception:
                images.append(Image.new("RGB", (224, 224)))
        emb = encoder.encode_images(images, batch_size=batch_size)
        all_img_emb.append(emb)
        if (i // batch_size) % 10 == 0:
            print(f"  images: {i + len(batch_paths)}/{len(unique_paths)}")
    image_embeddings = np.concatenate(all_img_emb, axis=0)
    print(f"  [OK] Image embeddings: {image_embeddings.shape}")

    # Extract text embeddings (all captions)
    captions = df["caption"].tolist()
    print(f"[embed] Extracting text embeddings for {len(captions)} captions...")
    all_txt_emb = []
    for i in range(0, len(captions), batch_size):
        batch = captions[i:i + batch_size]
        emb = encoder.encode_texts(batch, batch_size=batch_size)
        all_txt_emb.append(emb)
        if (i // batch_size) % 50 == 0:
            print(f"  texts: {i + len(batch)}/{len(captions)}")
    text_embeddings = np.concatenate(all_txt_emb, axis=0)
    print(f"  [OK] Text embeddings: {text_embeddings.shape}")

    encoder.unload()

    # Save
    np.save(img_path, image_embeddings)
    np.save(txt_path, text_embeddings)
    pd.DataFrame({"image_path": unique_paths}).to_parquet(out_dir / "image_map.parquet")
    pd.DataFrame({"caption": captions}).to_parquet(out_dir / "caption_map.parquet")
    df.to_parquet(out_dir / "df.parquet")
    print(f"[embed] Saved to {out_dir}")

    return {
        "image_embeddings": image_embeddings,
        "text_embeddings": text_embeddings,
        "image_paths": unique_paths,
        "captions": captions,
        "df": df,
    }


def build_ground_truth(image_embeddings, text_embeddings, df, num_queries=500):
    """Compute brute-force ground truth for text-to-image retrieval."""
    print(f"[gt] Computing brute-force ground truth ({num_queries} queries)...")

    # Each caption belongs to one image. Map caption_idx -> image_idx
    unique_images = sorted(df["image_path"].unique())
    path_to_img_idx = {p: i for i, p in enumerate(unique_images)}
    caption_to_img_idx = np.array([path_to_img_idx[p] for p in df["image_path"]])

    # Sample query indices (use captions as queries)
    rng = np.random.RandomState(42)
    query_indices = rng.choice(len(text_embeddings), size=num_queries, replace=False)
    query_vecs = text_embeddings[query_indices]
    gt_image_indices = caption_to_img_idx[query_indices]

    # Brute-force search
    similarity = query_vecs @ image_embeddings.T  # (num_queries, num_images)
    gt_top10 = np.argsort(-similarity, axis=1)[:, :10]

    # Also compute full ground truth rankings for recall calculation
    gt_rankings = np.argsort(-similarity, axis=1)  # full ranking

    print(f"  [OK] Ground truth computed: {gt_top10.shape}")
    return {
        "query_indices": query_indices,
        "query_vectors": query_vecs,
        "gt_image_indices": gt_image_indices,
        "gt_top10": gt_top10,
        "gt_rankings": gt_rankings,
    }


def evaluate_index(index, query_vectors, gt_indices):
    """Evaluate one index: compute recall@k and latency.

    Args:
        index: ANN index with timed_search method
        query_vectors: (N, dim) query embeddings
        gt_indices: (N,) ground truth item index for each query
    """
    from src.evaluation.metrics import RetrievalMetrics

    all_preds = []
    all_latencies = []
    for q in query_vectors:
        output = index.timed_search(q, k=10)
        preds = output["ids"]
        # timed_search returns 2D array (1, k) — flatten to 1D
        preds = np.atleast_1d(np.asarray(preds).flatten())[:10]
        if len(preds) < 10:
            preds = np.pad(preds, (0, 10 - len(preds)), constant_values=-1)
        all_preds.append(preds)
        all_latencies.append(output["latency_ms"])

    predictions = np.array(all_preds)  # (num_queries, 10)
    metrics = RetrievalMetrics()

    return {
        "recall@1": metrics.recall_at_k(predictions, gt_indices, k=1),
        "recall@5": metrics.recall_at_k_batch(predictions, gt_indices, [5])[5],
        "recall@10": metrics.recall_at_k_batch(predictions, gt_indices, [10])[10],
        "mAP": metrics.mean_average_precision(predictions, gt_indices, k=10),
        "mrr": metrics.mean_reciprocal_rank(predictions, gt_indices),
        "latency": metrics.latency_stats(all_latencies),
    }


def run_e1_ann_comparison(image_embeddings, gt_data):
    """Experiment 1: Compare all ANN algorithms at scale."""
    print("\n" + "=" * 70)
    print("E1: Full-Scale ANN Algorithm Comparison")
    print("=" * 70)
    print(f"  Database: {len(image_embeddings)} vectors × {image_embeddings.shape[1]}d")
    print(f"  Queries: {len(gt_data['query_vectors'])}")

    results = {}
    query_vecs = gt_data["query_vectors"]
    gt_indices = gt_data["gt_image_indices"]  # (N,) correct image index per query
    dim = image_embeddings.shape[1]

    # 1. FAISS IVF-PQ (tuned: nlist=400, nprobe=64, m=64)
    print("\n[1/4] FAISS IVF-PQ (nlist=400, nprobe=64)...")
    from src.indexing.faiss_index import FAISSIndex
    faiss_idx = FAISSIndex(dim=dim, index_type="ivfpq", nlist=400, m=64, nprobe=64)
    t0 = time.perf_counter()
    faiss_idx.build(image_embeddings)
    build_t = time.perf_counter() - t0
    results["FAISS_IVFPQ"] = evaluate_index(faiss_idx, query_vecs, gt_indices)
    results["FAISS_IVFPQ"]["build_time_s"] = build_t
    print(f"  Build: {build_t:.2f}s, R@10: {results['FAISS_IVFPQ']['recall@10']:.4f}, "
          f"P50: {results['FAISS_IVFPQ']['latency']['p50_ms']:.3f}ms")

    # 2. HNSW (hnswlib, tuned: M=32, ef_construction=300, ef_search=200)
    print("\n[2/4] HNSW (hnswlib, M=32, ef=300, ef_search=200)...")
    from src.indexing.hnsw_lib import HNSWLib
    hnsw = HNSWLib(dim=dim, M=32, ef_construction=300, ef_search=200, metric="cosine")
    t0 = time.perf_counter()
    hnsw.build(image_embeddings)
    build_t = time.perf_counter() - t0
    results["HNSW_hnswlib"] = evaluate_index(hnsw, query_vecs, gt_indices)
    results["HNSW_hnswlib"]["build_time_s"] = build_t
    print(f"  Build: {build_t:.2f}s, R@10: {results['HNSW_hnswlib']['recall@10']:.4f}, "
          f"P50: {results['HNSW_hnswlib']['latency']['p50_ms']:.3f}ms")

    # 3. LSH (tuned: 30 tables, 4 hashes — fewer bits = more collisions per table)
    print("\n[3/4] LSH (30 tables, 4 hashes)...")
    from src.indexing.lsh import CosineLSH
    lsh = CosineLSH(dim=dim, n_tables=30, n_hashes=4)
    t0 = time.perf_counter()
    lsh.build(image_embeddings)
    build_t = time.perf_counter() - t0
    results["LSH"] = evaluate_index(lsh, query_vecs, gt_indices)
    results["LSH"]["build_time_s"] = build_t
    print(f"  Build: {build_t:.2f}s, R@10: {results['LSH']['recall@10']:.4f}, "
          f"P50: {results['LSH']['latency']['p50_ms']:.3f}ms")

    # 4. Annoy (tuned: 100 trees)
    print("\n[4/4] Annoy (100 trees)...")
    from src.indexing.annoy_index import AnnoyIndexWrapper
    annoy = AnnoyIndexWrapper(dim=dim, n_trees=100, metric="cosine")
    t0 = time.perf_counter()
    annoy.build(image_embeddings)
    build_t = time.perf_counter() - t0
    results["Annoy"] = evaluate_index(annoy, query_vecs, gt_indices)
    results["Annoy"]["build_time_s"] = build_t
    print(f"  Build: {build_t:.2f}s, R@10: {results['Annoy']['recall@10']:.4f}, "
          f"P50: {results['Annoy']['latency']['p50_ms']:.3f}ms")

    # Save
    save_path = OUTPUT_DIR / "e1_full_scale.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)
    print(f"\n[OK] E1 results saved to {save_path}")

    return results


def run_e2_dimensionality_reduction(image_embeddings, gt_data):
    """Experiment 2: Effect of dimensionality reduction on retrieval quality."""
    print("\n" + "=" * 70)
    print("E2: Dimensionality Reduction Impact")
    print("=" * 70)

    from src.embedding.dim_reduction import DimReducer
    from src.indexing.faiss_index import FAISSIndex

    query_vecs = gt_data["query_vectors"]
    gt_indices = gt_data["gt_image_indices"]
    dim = image_embeddings.shape[1]

    results = {}
    methods = ["pca", "umap"]
    target_dims = [64, 128, 256, 512]

    for method in methods:
        for target_dim in target_dims:
            if target_dim >= dim:
                continue
            key = f"{method}_{target_dim}d"
            print(f"\n[{key}] Reducing from {dim}d to {target_dim}d...")

            try:
                reducer = DimReducer(method=method, n_components=target_dim, random_state=42)
                t0 = time.perf_counter()
                reduced_emb = reducer.fit_transform(image_embeddings)
                reduction_t = time.perf_counter() - t0

                # Also reduce query vectors
                reduced_queries = reducer.transform(query_vecs)

                # Build FAISS index on reduced vectors
                index = FAISSIndex(dim=target_dim)
                t0 = time.perf_counter()
                index.build(reduced_emb)
                build_t = time.perf_counter() - t0

                eval_result = evaluate_index(index, reduced_queries, gt_indices)
                eval_result["reduction_time_s"] = reduction_t
                eval_result["build_time_s"] = build_t

                # PCA: log explained variance
                if method == "pca" and reducer.explained_variance_ratio is not None:
                    eval_result["explained_variance"] = float(np.sum(reducer.explained_variance_ratio))
                    print(f"  Explained variance: {eval_result['explained_variance']:.4f}")

                results[key] = eval_result
                print(f"  R@10: {eval_result['recall@10']:.4f}, "
                      f"Build: {build_t:.2f}s, P50: {eval_result['latency']['p50_ms']:.3f}ms")

            except Exception as e:
                print(f"  [ERROR] {e}")
                results[key] = {"error": str(e)}

    save_path = OUTPUT_DIR / "e2_dim_reduction.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)
    print(f"\n[OK] E2 results saved to {save_path}")

    return results


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Full-scale experiments on Flickr30k")
    ap.add_argument("--stage", type=str, default="all",
                    choices=["all", "embed", "index", "e1", "e2"],
                    help="Which stage to run")
    ap.add_argument("--num-queries", type=int, default=500,
                    help="Number of query vectors for evaluation")
    args = ap.parse_args()

    # Stage 1: Load data
    df = load_flickr30k_full()

    if args.stage in ("all", "embed"):
        data = extract_embeddings(df)
    else:
        out_dir = Path("data/embeddings/flickr30k_full")
        data = {
            "image_embeddings": np.load(out_dir / "image_embeddings.npy"),
            "text_embeddings": np.load(out_dir / "text_embeddings.npy"),
            "image_paths": pd.read_parquet(out_dir / "image_map.parquet")["image_path"].tolist(),
            "captions": pd.read_parquet(out_dir / "caption_map.parquet")["caption"].tolist(),
            "df": pd.read_parquet(out_dir / "df.parquet"),
        }
        print(f"[load] Loaded cached embeddings: {data['image_embeddings'].shape}")

    # Build ground truth
    gt_data = build_ground_truth(
        data["image_embeddings"], data["text_embeddings"],
        data["df"], num_queries=args.num_queries
    )

    # Run experiments
    if args.stage in ("all", "e1"):
        run_e1_ann_comparison(data["image_embeddings"], gt_data)

    if args.stage in ("all", "e2"):
        run_e2_dimensionality_reduction(data["image_embeddings"], gt_data)

    print("\n" + "=" * 70)
    print("FULL-SCALE EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
