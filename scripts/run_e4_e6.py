"""
E4 + E6 combined experiment runner.
E4: Cross-modal symmetry (text→image vs image→text)
E6: Scale analysis (1K → 5K → 10K → 20K → 29K)
"""
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "report" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    emb_dir = Path("data/embeddings/flickr30k_full")
    img = np.load(emb_dir / "image_embeddings.npy")
    txt = np.load(emb_dir / "text_embeddings.npy")
    df = pd.read_parquet(emb_dir / "df.parquet")
    img_paths = pd.read_parquet(emb_dir / "image_map.parquet")["image_path"].tolist()
    return img, txt, df, img_paths


def build_crossmodal_ground_truth(img_emb, txt_emb, df, num_queries=300):
    """Build ground truth for both directions."""
    unique_images = sorted(df["image_path"].unique())
    path_to_img_idx = {p: i for i, p in enumerate(unique_images)}
    caption_to_img_idx = np.array([path_to_img_idx[p] for p in df["image_path"]])

    rng = np.random.RandomState(42)
    q_idx = rng.choice(len(txt_emb), size=num_queries, replace=False)

    # Text→Image: query=text, target=image
    t2i_queries = txt_emb[q_idx]
    t2i_gt = caption_to_img_idx[q_idx]

    # Image→Text: query=image, target=caption set (5 captions per image)
    # For each query image, we need the indices of its captions in txt_emb
    query_img_paths = [df.iloc[i]["image_path"] for i in q_idx]
    i2t_queries = img_emb[[path_to_img_idx[p] for p in query_img_paths]]

    # Build mapping: for each image, list its caption indices in txt_emb
    img_path_to_caption_indices = {}
    for i, p in enumerate(df["image_path"]):
        img_path_to_caption_indices.setdefault(p, []).append(i)

    i2t_gt = [img_path_to_caption_indices[p] for p in query_img_paths]  # list of lists

    # Brute-force for text→image
    t2i_sim = t2i_queries @ img_emb.T
    t2i_rankings = np.argsort(-t2i_sim, axis=1)

    # Brute-force for image→text
    i2t_sim = i2t_queries @ txt_emb.T
    i2t_rankings = np.argsort(-i2t_sim, axis=1)

    return {
        "t2i": {"queries": t2i_queries, "gt": t2i_gt, "rankings": t2i_rankings},
        "i2t": {"queries": i2t_queries, "gt": i2t_gt, "rankings": i2t_rankings},
        "query_captions": [df.iloc[i]["caption"] for i in q_idx],
        "query_image_paths": query_img_paths,
    }


def evaluate_retrieval(predictions, ground_truth, k_values=(1, 5, 10)):
    """Evaluate retrieval results. ground_truth can be single int or list of ints."""
    from src.evaluation.metrics import RetrievalMetrics
    metrics = RetrievalMetrics()

    results = {}
    for k in k_values:
        # Handle multi-label ground truth
        hits = 0
        for pred, gt in zip(predictions, ground_truth):
            gt_set = set(gt) if isinstance(gt, (list, np.ndarray)) else {gt}
            if len(set(pred[:k]) & gt_set) > 0:
                hits += 1
        results[f"recall@{k}"] = hits / len(predictions)

    # mAP (divide by number of relevant items for multi-label)
    aps = []
    for pred, gt in zip(predictions, ground_truth):
        gt_set = set(gt) if isinstance(gt, (list, np.ndarray)) else {gt}
        hits = 0
        ap = 0.0
        for j, pid in enumerate(pred[:10]):
            if pid in gt_set:
                hits += 1
                ap += hits / (j + 1)
        num_relevant = len(gt_set)
        if num_relevant > 0:
            ap = ap / num_relevant
        aps.append(ap)
    results["mAP@10"] = float(np.mean(aps))

    # MRR
    rrs = []
    for pred, gt in zip(predictions, ground_truth):
        gt_set = set(gt) if isinstance(gt, (list, np.ndarray)) else {gt}
        for j, pid in enumerate(pred):
            if pid in gt_set:
                rrs.append(1.0 / (j + 1))
                break
        else:
            rrs.append(0.0)
    results["mrr"] = float(np.mean(rrs))

    return results


def run_e4_crossmodal_symmetry():
    """E4: Compare text→image vs image→text retrieval."""
    print("=" * 70)
    print("E4: Cross-Modal Symmetry Analysis")
    print("=" * 70)

    print("[1/3] Loading data...")
    img_emb, txt_emb, df, img_paths = load_data()
    print(f"  Images: {img_emb.shape}, Texts: {txt_emb.shape}")

    print("[2/3] Building ground truth for both directions...")
    gt = build_crossmodal_ground_truth(img_emb, txt_emb, df, num_queries=300)

    # Build HNSW indexes
    print("[3/3] Building HNSW indexes and evaluating...")
    from src.indexing.hnsw_lib import HNSWLib

    results = {}

    # --- Text → Image ---
    print("\n  [T→I] Text query → Image database...")
    t2i_idx = HNSWLib(dim=img_emb.shape[1], M=32, ef_construction=300, ef_search=200, metric="cosine")
    t2i_idx.timed_build(img_emb)

    t2i_preds = []
    t2i_latencies = []
    for q in gt["t2i"]["queries"]:
        output = t2i_idx.timed_search(q, k=10)
        preds = np.atleast_1d(np.asarray(output["ids"]).flatten())[:10]
        t2i_preds.append(preds)
        t2i_latencies.append(output["latency_ms"])

    t2i_eval = evaluate_retrieval(t2i_preds, gt["t2i"]["gt"])
    results["text_to_image"] = {
        **t2i_eval,
        "build_time_s": t2i_idx.build_time,
        "latency": latency_stats(t2i_latencies),
    }
    print(f"    R@1={t2i_eval['recall@1']:.4f}, R@5={t2i_eval['recall@5']:.4f}, "
          f"R@10={t2i_eval['recall@10']:.4f}, mAP={t2i_eval['mAP@10']:.4f}")

    # --- Image → Text ---
    print("\n  [I→T] Image query → Text database...")
    i2t_idx = HNSWLib(dim=txt_emb.shape[1], M=32, ef_construction=300, ef_search=200, metric="cosine")
    i2t_idx.timed_build(txt_emb)

    i2t_preds = []
    i2t_latencies = []
    for q in gt["i2t"]["queries"]:
        output = i2t_idx.timed_search(q, k=10)
        preds = np.atleast_1d(np.asarray(output["ids"]).flatten())[:10]
        i2t_preds.append(preds)
        i2t_latencies.append(output["latency_ms"])

    i2t_eval = evaluate_retrieval(i2t_preds, gt["i2t"]["gt"])
    results["image_to_text"] = {
        **i2t_eval,
        "build_time_s": i2t_idx.build_time,
        "latency": latency_stats(i2t_latencies),
    }
    print(f"    R@1={i2t_eval['recall@1']:.4f}, R@5={i2t_eval['recall@5']:.4f}, "
          f"R@10={i2t_eval['recall@10']:.4f}, mAP={i2t_eval['mAP@10']:.4f}")

    # Compute brute-force upper bounds for both directions
    t2i_bf = evaluate_retrieval(
        [r[:10] for r in gt["t2i"]["rankings"]], gt["t2i"]["gt"]
    )
    i2t_bf = evaluate_retrieval(
        [r[:10] for r in gt["i2t"]["rankings"]], gt["i2t"]["gt"]
    )
    results["t2i_bruteforce_upperbound"] = t2i_bf
    results["i2t_bruteforce_upperbound"] = i2t_bf

    print(f"\n  Brute-force T→I upper bound: R@10={t2i_bf['recall@10']:.4f}")
    print(f"  Brute-force I→T upper bound: R@10={i2t_bf['recall@10']:.4f}")

    # Save
    save_path = RESULTS_DIR / "e4_crossmodal_symmetry.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=float_convert)
    print(f"\n[OK] E4 saved to {save_path}")

    # Generate table
    generate_e4_table(results)
    generate_e4_chart(results)

    return results


def run_e6_scale_analysis():
    """E6: Scale analysis - performance vs database size."""
    print("\n" + "=" * 70)
    print("E6: Scale Analysis (1K → 29K)")
    print("=" * 70)

    img_emb, txt_emb, df, img_paths = load_data()
    unique_images = sorted(df["image_path"].unique())
    path_to_img_idx = {p: i for i, p in enumerate(unique_images)}
    caption_to_img_idx = np.array([path_to_img_idx[p] for p in df["image_path"]])

    # Fixed query set (200 text queries)
    rng = np.random.RandomState(42)
    q_idx = rng.choice(len(txt_emb), size=200, replace=False)
    query_vecs = txt_emb[q_idx]
    gt_indices = caption_to_img_idx[q_idx]

    from src.indexing.hnsw_lib import HNSWLib
    from src.indexing.faiss_index import FAISSIndex
    from src.indexing.annoy_index import AnnoyIndexWrapper

    scales = [1000, 5000, 10000, 20000, 29000]
    all_results = {}
    for algo_name in ["HNSW", "FAISS_IVFPQ", "Annoy"]:
        print(f"\n[{algo_name}]")
        algo_results = {}
        for scale in scales:
            if scale > len(img_emb):
                scale = len(img_emb)

            subset = img_emb[:scale]
            # Filter gt_indices to only those < scale
            valid_q = gt_indices < scale
            valid_query_vecs = query_vecs[valid_q]
            valid_gt = gt_indices[valid_q]

            print(f"  scale={scale}: {len(subset)} images, {len(valid_query_vecs)} valid queries")

            # Scale nlist for FAISS based on data size
            if algo_name == "HNSW":
                index = HNSWLib(dim=img_emb.shape[1], M=32, ef_construction=300, ef_search=200, metric="cosine")
            elif algo_name == "FAISS_IVFPQ":
                nlist = max(10, min(scale // 40, 400))  # at least 40 samples per cluster
                # m=64 matches E1: 512/64=8 dims per sub-vector with 8-bit codes (moderate compression)
                index = FAISSIndex(dim=img_emb.shape[1], index_type="ivfpq", nlist=nlist, m=64, nprobe=64)
            elif algo_name == "Annoy":
                index = AnnoyIndexWrapper(dim=img_emb.shape[1], n_trees=min(50, scale // 20 + 10), metric="cosine")
            t0 = time.perf_counter()
            index.build(subset)
            build_t = time.perf_counter() - t0

            preds_list = []
            latencies = []
            for q in valid_query_vecs:
                output = index.timed_search(q, k=10)
                preds = np.atleast_1d(np.asarray(output["ids"]).flatten())[:10]
                preds_list.append(preds)
                latencies.append(output["latency_ms"])

            eval_result = evaluate_retrieval(preds_list, valid_gt)
            algo_results[str(scale)] = {
                **eval_result,
                "build_time_s": build_t,
                "num_valid_queries": int(valid_q.sum()),
                "latency": latency_stats(latencies),
            }
            print(f"    R@10={eval_result['recall@10']:.4f}, "
                  f"Build={build_t:.2f}s, P50={latency_stats(latencies)['p50_ms']:.3f}ms")

        all_results[algo_name] = algo_results

    # Save
    save_path = RESULTS_DIR / "e6_scale_analysis.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float_convert)
    print(f"\n[OK] E6 saved to {save_path}")

    # Generate charts
    generate_e6_charts(all_results)

    return all_results


def latency_stats(latencies):
    arr = np.array(latencies)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }


def float_convert(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def generate_e4_table(results):
    """Generate E4 comparison table as markdown."""
    rows = []
    for name in ["text_to_image", "image_to_text"]:
        r = results[name]
        ub = results[f"{'t2i' if name == 'text_to_image' else 'i2t'}_bruteforce_upperbound"]
        rows.append({
            "Direction": "Text → Image" if name == "text_to_image" else "Image → Text",
            "R@1": f"{r['recall@1']:.3f}",
            "R@5": f"{r['recall@5']:.3f}",
            "R@10": f"{r['recall@10']:.3f}",
            "mAP": f"{r['mAP@10']:.3f}",
            "MRR": f"{r['mrr']:.3f}",
            "BF Upper Bound R@10": f"{ub['recall@10']:.3f}",
            "Build(s)": f"{r['build_time_s']:.1f}",
            "P50(ms)": f"{r['latency']['p50_ms']:.2f}",
        })

    table_md = "| Direction | R@1 | R@5 | R@10 | mAP | MRR | BF UB R@10 | Build(s) | P50(ms) |\n"
    table_md += "|-----------|-----|-----|------|-----|-----|------------|----------|----------|\n"
    for row in rows:
        table_md += f"| {row['Direction']} | {row['R@1']} | {row['R@5']} | {row['R@10']} | {row['mAP']} | {row['MRR']} | {row['BF Upper Bound R@10']} | {row['Build(s)']} | {row['P50(ms)']} |\n"

    table_path = FIGURES_DIR / "e4_symmetry_table.md"
    with open(table_path, "w") as f:
        f.write(table_md)
    print(f"[E4] Table saved to {table_path}")


def generate_e4_chart(results):
    """Generate E4 bar chart comparing T→I vs I→T."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        t2i = results["text_to_image"]
        i2t = results["image_to_text"]
        t2i_bf = results["t2i_bruteforce_upperbound"]
        i2t_bf = results["i2t_bruteforce_upperbound"]

        metrics = ["recall@1", "recall@5", "recall@10", "mAP@10", "mrr"]
        labels = ["R@1", "R@5", "R@10", "mAP", "MRR"]

        x = np.arange(len(metrics))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 5))
        bars1 = ax.bar(x - width, [t2i[m] for m in metrics], width, label='Text→Image (HNSW)', color='#2196F3')
        bars2 = ax.bar(x, [i2t[m] for m in metrics], width, label='Image→Text (HNSW)', color='#FF9800')
        bars3 = ax.bar(x + width, [t2i_bf[m] for m in metrics], width, label='T→I Brute-Force (UB)', color='#BBDEFB', alpha=0.7)

        ax.set_ylabel('Score')
        ax.set_title('E4: Cross-Modal Retrieval Symmetry')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "e4_crossmodal_symmetry.png", dpi=150)
        plt.close()
        print(f"[E4] Chart saved to {FIGURES_DIR / 'e4_crossmodal_symmetry.png'}")
    except Exception as e:
        print(f"[E4] Chart generation failed: {e}")


def generate_e6_charts(all_results):
    """Generate E6 scale analysis charts."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        scales = [1000, 5000, 10000, 20000, 29000]
        colors = {"HNSW": "#2196F3", "FAISS_IVFPQ": "#4CAF50", "Annoy": "#FF9800"}
        markers = {"HNSW": "o", "FAISS_IVFPQ": "s", "Annoy": "^"}

        # Chart 1: Recall@10 vs Scale
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        ax = axes[0]
        for algo_name in ["HNSW", "FAISS_IVFPQ", "Annoy"]:
            algo_data = all_results[algo_name]
            r10 = [algo_data[str(s)]["recall@10"] for s in scales]
            ax.plot(scales, r10, marker=markers[algo_name], color=colors[algo_name],
                    label=algo_name, linewidth=2, markersize=8)
        ax.set_xlabel("Database Size")
        ax.set_ylabel("Recall@10")
        ax.set_title("Recall@10 vs Scale")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 0.7)

        ax = axes[1]
        for algo_name in ["HNSW", "FAISS_IVFPQ", "Annoy"]:
            algo_data = all_results[algo_name]
            p50 = [algo_data[str(s)]["latency"]["p50_ms"] for s in scales]
            ax.plot(scales, p50, marker=markers[algo_name], color=colors[algo_name],
                    label=algo_name, linewidth=2, markersize=8)
        ax.set_xlabel("Database Size")
        ax.set_ylabel("P50 Latency (ms)")
        ax.set_title("Query Latency vs Scale")
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[2]
        for algo_name in ["HNSW", "FAISS_IVFPQ", "Annoy"]:
            algo_data = all_results[algo_name]
            bt = [algo_data[str(s)]["build_time_s"] for s in scales]
            ax.plot(scales, bt, marker=markers[algo_name], color=colors[algo_name],
                    label=algo_name, linewidth=2, markersize=8)
        ax.set_xlabel("Database Size")
        ax.set_ylabel("Build Time (s)")
        ax.set_title("Build Time vs Scale")
        ax.legend()
        ax.grid(alpha=0.3)

        fig.suptitle("E6: Scale Analysis — Performance vs Database Size", fontsize=14)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "e6_scale_analysis.png", dpi=150)
        plt.close()
        print(f"[E6] Chart saved to {FIGURES_DIR / 'e6_scale_analysis.png'}")

        # Chart 2: Recall-Latency tradeoff scatter
        fig, ax = plt.subplots(figsize=(8, 6))
        for algo_name in ["HNSW", "FAISS_IVFPQ", "Annoy"]:
            algo_data = all_results[algo_name]
            r10_vals = [algo_data[str(s)]["recall@10"] for s in scales]
            lat_vals = [algo_data[str(s)]["latency"]["p50_ms"] for s in scales]
            ax.scatter(lat_vals, r10_vals, s=[s/100 for s in scales],
                      color=colors[algo_name], label=algo_name, alpha=0.7)
            for i, s in enumerate(scales):
                ax.annotate(f"{s//1000}K", (lat_vals[i], r10_vals[i]),
                           fontsize=7, ha='left')

        ax.set_xlabel("P50 Latency (ms)")
        ax.set_ylabel("Recall@10")
        ax.set_title("E6: Recall-Latency Tradeoff (bubble size = database size)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "e6_recall_latency_tradeoff.png", dpi=150)
        plt.close()
        print(f"[E6] Tradeoff chart saved to {FIGURES_DIR / 'e6_recall_latency_tradeoff.png'}")

        # Generate markdown table
        table_md = "| Scale | Algorithm | R@10 | Build(s) | P50(ms) | P95(ms) |\n"
        table_md += "|-------|-----------|------|----------|----------|----------|\n"
        for s in scales:
            for algo_name in ["HNSW", "FAISS_IVFPQ", "Annoy"]:
                d = all_results[algo_name][str(s)]
                table_md += f"| {s//1000}K | {algo_name} | {d['recall@10']:.3f} | {d['build_time_s']:.1f} | {d['latency']['p50_ms']:.1f} | {d['latency']['p95_ms']:.1f} |\n"
            table_md += f"|-------|-----------|------|----------|----------|----------|\n"

        table_path = FIGURES_DIR / "e6_scale_table.md"
        with open(table_path, "w") as f:
            f.write(table_md)
        print(f"[E6] Table saved to {table_path}")

    except Exception as e:
        print(f"[E6] Chart generation failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["e4", "e6", "all"], default="all")
    args = ap.parse_args()

    if args.stage in ("e4", "all"):
        run_e4_crossmodal_symmetry()
    if args.stage in ("e6", "all"):
        run_e6_scale_analysis()

    print("\n" + "=" * 70)
    print("E4 + E6 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
