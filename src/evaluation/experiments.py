"""
Experiment runner for systematic benchmarking of retrieval strategies.
Orchestrates all 6 experiments defined in experiment_config.yaml.
"""

import time
import json
from pathlib import Path
from typing import Optional

import numpy as np

from .metrics import RetrievalMetrics


class ExperimentRunner:
    """Run systematic retrieval benchmark experiments."""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = RetrievalMetrics()

    def run_ann_comparison(self, indices: dict, test_queries: np.ndarray,
                           ground_truth: np.ndarray, k_values: list[int] = (1, 5, 10, 50),
                           num_runs: int = 3) -> dict:
        """Experiment 1: Compare multiple ANN algorithms."""
        results = {}

        for name, index in indices.items():
            print(f"  [{name}] benchmarking...")
            all_latencies = []
            all_predictions = []

            for _ in range(num_runs):
                for q in test_queries:
                    output = index.timed_search(q, k=max(k_values))
                    all_latencies.append(output["latency_ms"])
                    all_predictions.append(output["ids"])

            predictions = np.array(all_predictions[-len(test_queries):])  # Last run
            recall = self.metrics.recall_at_k_batch(predictions, ground_truth, k_values)

            results[name] = {
                "recall": recall,
                "build_time_s": index.build_time,
                "latency": self.metrics.latency_stats(all_latencies),
            }
            print(f"    Recall@10: {recall[10]:.4f}, P50: {results[name]['latency']['p50_ms']:.1f}ms")

        return results

    def run_dim_reduction_experiment(self, embeddings: np.ndarray,
                                     test_queries: np.ndarray,
                                     ground_truth: np.ndarray,
                                     dims: list[int] = (64, 128, 256, 512),
                                     methods: tuple[str, ...] = ("pca", "umap")):
        """Experiment 2: Effect of dimensionality reduction on retrieval."""
        from src.embedding.dim_reduction import DimReducer
        from src.indexing.faiss_index import FAISSIndex

        results = {}
        for method in methods:
            for dim in dims:
                if dim >= embeddings.shape[1]:
                    continue
                key = f"{method}_{dim}d"
                print(f"  [{key}] reducing and indexing...")
                try:
                    reducer = DimReducer(method=method, n_components=dim)
                    reduced = reducer.fit_transform(embeddings)

                    index = FAISSIndex(dim=dim)
                    index.timed_build(reduced)

                    all_preds = []
                    for q in test_queries:
                        q_reduced = reducer.transform(q.reshape(1, -1))
                        ids, _ = index.search(q_reduced, k=10)
                        all_preds.append(ids)

                    predictions = np.array(all_preds)
                    recall_10 = self.metrics.recall_at_k(predictions, ground_truth, k=10)

                    results[key] = {
                        "recall@10": recall_10,
                        "build_time_s": index.build_time,
                    }
                    print(f"    Recall@10: {recall_10:.4f}")
                except Exception as e:
                    results[key] = {"error": str(e)}

        return results

    def run_hybrid_ablation(self, retriever, test_queries: list[str],
                            ground_truth: np.ndarray,
                            strategies: list[str] = None):
        """Experiment 3: Ablation study of hybrid retrieval strategies."""
        if strategies is None:
            strategies = ["pure_dense", "pure_sparse", "fixed_weight",
                          "adaptive", "multi_stage"]

        results = {}
        for strategy in strategies:
            print(f"  [{strategy}] evaluating...")
            all_predictions = []
            all_latencies = []

            for query_text in test_queries:
                t0 = time.perf_counter()
                res = retriever.search(query_text=query_text, k=10, strategy=strategy)
                latency = (time.perf_counter() - t0) * 1000
                all_latencies.append(latency)
                all_predictions.append([r["id"] for r in res])

            predictions = np.array(all_predictions)
            recall = self.metrics.recall_at_k_batch(predictions, ground_truth,
                                                    (1, 5, 10))

            results[strategy] = {
                "recall": recall,
                "latency": self.metrics.latency_stats(all_latencies),
            }
            print(f"    Recall@10: {recall[10]:.4f}, P50: {results[strategy]['latency']['p50_ms']:.1f}ms")

        return results

    def run_scale_experiment(self, index_cls, embeddings: np.ndarray,
                             test_queries: np.ndarray, ground_truth: np.ndarray,
                             scales: list[int] = (1000, 5000, 10000, 50000, 100000)):
        """Experiment 6: Latency and recall vs. database size."""
        results = {}
        for scale in scales:
            if scale > len(embeddings):
                scale = len(embeddings)

            sub_embeddings = embeddings[:scale]
            sub_gt = ground_truth[np.where(ground_truth < scale)]

            print(f"  [scale={scale}] indexing {len(sub_embeddings)} vectors...")
            index = index_cls(dim=embeddings.shape[1])
            index.timed_build(sub_embeddings)

            latencies = []
            all_preds = []
            for q in test_queries:
                output = index.timed_search(q, k=10)
                latencies.append(output["latency_ms"])
                all_preds.append(output["ids"])

            predictions = np.array(all_preds)
            recall_10 = self.metrics.recall_at_k(predictions[:len(sub_gt)], sub_gt, k=10)

            results[scale] = {
                "recall@10": recall_10,
                "build_time_s": index.build_time,
                "latency": self.metrics.latency_stats(latencies),
            }
            print(f"    Recall@10: {recall_10:.4f}, P50: {results[scale]['latency']['p50_ms']:.1f}ms")

        return results

    def save_results(self, results: dict, filename: str):
        """Save experiment results as JSON."""
        path = self.output_dir / filename

        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=convert)

        print(f"[save] Results saved to {path}")
