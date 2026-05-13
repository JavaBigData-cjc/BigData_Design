"""
E3 Hybrid Retrieval Ablation Experiment.
Compares: pure_dense, pure_sparse, fixed_weight, adaptive, multi_stage
Uses full Flickr30k data with text-to-image cross-modal retrieval.
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


def load_data():
    """Load pre-computed Flickr30k embeddings and metadata."""
    emb_dir = Path("data/embeddings/flickr30k_full")
    img_emb = np.load(emb_dir / "image_embeddings.npy")
    txt_emb = np.load(emb_dir / "text_embeddings.npy")
    df = pd.read_parquet(emb_dir / "df.parquet")
    img_paths = pd.read_parquet(emb_dir / "image_map.parquet")["image_path"].tolist()
    captions = pd.read_parquet(emb_dir / "caption_map.parquet")["caption"].tolist()
    return img_emb, txt_emb, df, img_paths, captions


def build_ground_truth(image_embeddings, text_embeddings, df, num_queries=200):
    """Build brute-force ground truth for evaluation."""
    unique_images = sorted(df["image_path"].unique())
    path_to_idx = {p: i for i, p in enumerate(unique_images)}
    caption_to_img = np.array([path_to_idx[p] for p in df["image_path"]])

    rng = np.random.RandomState(42)
    q_idx = rng.choice(len(text_embeddings), size=num_queries, replace=False)
    q_vecs = text_embeddings[q_idx]
    q_texts = [df.iloc[i]["caption"] for i in q_idx]
    gt_indices = caption_to_img[q_idx]

    # Brute-force top-10
    sim = q_vecs @ image_embeddings.T
    gt_top10 = np.argsort(-sim, axis=1)[:, :10]

    return {
        "query_indices": q_idx,
        "query_vectors": q_vecs,
        "query_texts": q_texts,
        "gt_indices": gt_indices,
        "gt_top10": gt_top10,
        "num_images": len(unique_images),
    }


def evaluate_results(predictions, gt_indices):
    """Evaluate list of prediction arrays."""
    from src.evaluation.metrics import RetrievalMetrics
    metrics = RetrievalMetrics()

    preds = np.array(predictions)  # (num_queries, k)
    return {
        "recall@1": metrics.recall_at_k(preds, gt_indices, k=1),
        "recall@5": metrics.recall_at_k_batch(preds, gt_indices, [5])[5],
        "recall@10": metrics.recall_at_k_batch(preds, gt_indices, [10])[10],
        "mAP@10": metrics.mean_average_precision(preds, gt_indices, k=10),
        "mrr": metrics.mean_reciprocal_rank(preds, gt_indices),
    }


def run_e3_hybrid_ablation():
    """E3: Ablation study of hybrid retrieval strategies."""
    print("=" * 70)
    print("E3: Hybrid Retrieval Ablation Study")
    print("=" * 70)

    print("[1/4] Loading data...")
    img_emb, txt_emb, df, img_paths, captions = load_data()
    print(f"  Images: {img_emb.shape}, Texts: {txt_emb.shape}")

    print("[2/4] Building ground truth...")
    gt_data = build_ground_truth(img_emb, txt_emb, df, num_queries=200)

    # Build BM25 index on captions
    print("[3/4] Building BM25 index...")
    from src.retrieval.bm25_index import BM25Index
    bm25 = BM25Index(k1=1.5, b=0.75)
    # Build on unique image captions (first caption per image for simplicity)
    unique_images = sorted(df["image_path"].unique())
    img_to_first_caption = df.groupby("image_path")["caption"].first()
    doc_captions = [img_to_first_caption[p] for p in unique_images]
    bm25.build(doc_captions)
    print(f"  BM25: {bm25.num_docs} documents, {len(bm25.doc_freq)} terms")

    # Build FAISS dense index
    print("[4/4] Building FAISS dense index...")
    from src.indexing.faiss_index import FAISSIndex
    dense_idx = FAISSIndex(dim=img_emb.shape[1], index_type="flat")
    dense_idx.build(img_emb)

    # Run strategies
    query_vecs = gt_data["query_vectors"]
    query_texts = gt_data["query_texts"]
    gt_indices = gt_data["gt_indices"]

    results = {}
    strategies = ["pure_dense", "pure_sparse", "fixed_weight", "adaptive"]

    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")
        all_preds = []
        all_latencies = []
        query_weights = []

        t0_total = time.perf_counter()
        for i, (q_text, q_vec) in enumerate(zip(query_texts, query_vecs)):
            t0 = time.perf_counter()
            preds = _run_strategy(strategy, q_text, q_vec, dense_idx, bm25, img_emb.shape[0])
            latency = (time.perf_counter() - t0) * 1000
            all_preds.append(preds)
            all_latencies.append(latency)

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(query_texts)} queries done")

        total_time = time.perf_counter() - t0_total
        eval_metrics = evaluate_results(all_preds, gt_indices)

        from src.evaluation.metrics import RetrievalMetrics
        metrics = RetrievalMetrics()
        results[strategy] = {
            **eval_metrics,
            "total_time_s": total_time,
            "qps": len(query_texts) / total_time,
            "latency": metrics.latency_stats(all_latencies),
        }
        print(f"  R@1={eval_metrics['recall@1']:.4f}, "
              f"R@5={eval_metrics['recall@5']:.4f}, "
              f"R@10={eval_metrics['recall@10']:.4f}, "
              f"mAP={eval_metrics['mAP@10']:.4f}, "
              f"MRR={eval_metrics['mrr']:.4f}")

    # Save results
    save_path = OUTPUT_DIR / "e3_hybrid_ablation.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"\n[OK] E3 results saved to {save_path}")

    return results


def _run_strategy(strategy, q_text, q_vec, dense_idx, bm25, num_images):
    """Execute a single query with a given strategy."""
    k = 10

    if strategy == "pure_dense":
        output = dense_idx.timed_search(q_vec, k=k)
        preds = np.atleast_1d(np.asarray(output["ids"]).flatten())[:k]

    elif strategy == "pure_sparse":
        ids, scores = bm25.search(q_text, k=k)
        preds = ids[:k]

    elif strategy == "fixed_weight":
        from src.retrieval.fusion import ScoreFusion
        fusion = ScoreFusion()
        # Dense: top-200
        d_ids, d_dists = dense_idx.search(q_vec.reshape(1, -1), k=200)
        d_ids = d_ids.flatten()
        d_scores = 1.0 - d_dists.flatten()

        # Sparse: top-200
        s_ids, s_scores = bm25.search(q_text, k=200)

        # Align
        all_ids = sorted(set(d_ids) | set(s_ids))
        id_to_idx = {i: idx for idx, i in enumerate(all_ids)}
        d_aligned = np.zeros(len(all_ids))
        s_aligned = np.zeros(len(all_ids))
        for idx, did in enumerate(d_ids):
            d_aligned[id_to_idx[did]] = d_scores[idx]
        for idx, sid in enumerate(s_ids):
            if sid in id_to_idx:
                s_aligned[id_to_idx[sid]] = s_scores[idx]

        fused = fusion.weighted_sum(d_aligned, s_aligned, weights=(0.6, 0.4, 0.0))
        top_k = np.argsort(-fused)[:k]
        preds = np.array(all_ids)[top_k]

    elif strategy == "adaptive":
        from src.retrieval.fusion import ScoreFusion
        from src.retrieval.query_router import QueryRouter
        router = QueryRouter(bm25_index=bm25)
        w_d, w_s, w_m = router.get_weights(q_text)
        q_type = router.classify(q_text)

        fusion = ScoreFusion()
        # Dense: top-200
        d_ids, d_dists = dense_idx.search(q_vec.reshape(1, -1), k=200)
        d_ids = d_ids.flatten()
        d_scores = 1.0 - d_dists.flatten()

        # Sparse: top-200
        s_ids, s_scores = bm25.search(q_text, k=200)

        # Align
        all_ids = sorted(set(d_ids) | set(s_ids))
        id_to_idx = {i: idx for idx, i in enumerate(all_ids)}
        d_aligned = np.zeros(len(all_ids))
        s_aligned = np.zeros(len(all_ids))
        for idx, did in enumerate(d_ids):
            d_aligned[id_to_idx[did]] = d_scores[idx]
        for idx, sid in enumerate(s_ids):
            if sid in id_to_idx:
                s_aligned[id_to_idx[sid]] = s_scores[idx]

        fused = fusion.weighted_sum(d_aligned, s_aligned, weights=(w_d, w_s, w_m))
        top_k = np.argsort(-fused)[:k]
        preds = np.array(all_ids)[top_k]

    else:
        # fallback to dense
        output = dense_idx.timed_search(q_vec, k=k)
        preds = np.atleast_1d(np.asarray(output["ids"]).flatten())[:k]

    return preds


if __name__ == "__main__":
    run_e3_hybrid_ablation()
