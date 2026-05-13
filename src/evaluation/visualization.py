"""
Visualization utilities for experiment results.
Produces charts for the course report (matplotlib + seaborn).
"""

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend


class ResultVisualizer:
    """Generate publication-quality charts for experiment results."""

    def __init__(self, output_dir: str = "report/figures", style: str = "seaborn-v0_8-whitegrid"):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use(style)
        sns.set_palette("husl")

    def recall_comparison_chart(self, ann_results: dict,
                                k_values: list[int] = (1, 5, 10, 50)):
        """Bar chart: Recall@K across ANN algorithms."""
        fig, ax = plt.subplots(figsize=(10, 6))

        algorithms = list(ann_results.keys())
        x = np.arange(len(algorithms))
        width = 0.2

        for i, k in enumerate(k_values):
            values = [ann_results[alg]["recall"].get(k, 0) for alg in algorithms]
            ax.bar(x + i * width, values, width, label=f"K={k}")

        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Recall")
        ax.set_title("ANN Algorithm Comparison: Recall@K")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(algorithms, rotation=30, ha="right")
        ax.legend()
        ax.set_ylim(0, 1.05)

        fig.tight_layout()
        fig.savefig(f"{self.output_dir}/recall_comparison.png", dpi=150)
        plt.close(fig)

    def latency_recall_scatter(self, ann_results: dict):
        """Scatter plot: latency vs recall trade-off (Pareto frontier)."""
        fig, ax = plt.subplots(figsize=(8, 6))

        for name, res in ann_results.items():
            recall = res["recall"].get(10, 0)
            latency = res["latency"]["p50_ms"]
            ax.scatter(latency, recall, s=150, label=name, alpha=0.8)
            ax.annotate(name, (latency, recall), fontsize=9,
                        xytext=(5, 5), textcoords="offset points")

        ax.set_xlabel("P50 Latency (ms)")
        ax.set_ylabel("Recall@10")
        ax.set_title("Recall vs. Latency Trade-off")
        ax.legend()
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.05)

        fig.tight_layout()
        fig.savefig(f"{self.output_dir}/latency_recall_scatter.png", dpi=150)
        plt.close(fig)

    def hybrid_ablation_chart(self, ablation_results: dict):
        """Grouped bar chart: hybrid retrieval ablation study."""
        fig, ax = plt.subplots(figsize=(10, 6))

        strategies = list(ablation_results.keys())
        x = np.arange(len(strategies))
        width = 0.25

        for i, k in enumerate([1, 5, 10]):
            values = [ablation_results[s]["recall"].get(k, 0) for s in strategies]
            ax.bar(x + i * width, values, width, label=f"K={k}")

        ax.set_xlabel("Retrieval Strategy")
        ax.set_ylabel("Recall")
        ax.set_title("Hybrid Retrieval Ablation Study")
        ax.set_xticks(x + width)
        ax.set_xticklabels(strategies, rotation=30, ha="right")
        ax.legend()
        ax.set_ylim(0, 1.05)

        fig.tight_layout()
        fig.savefig(f"{self.output_dir}/hybrid_ablation.png", dpi=150)
        plt.close(fig)

    def scale_analysis_chart(self, scale_results: dict):
        """Line chart: latency and recall vs. database scale."""
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()

        scales = sorted(scale_results.keys())
        recall_10 = [scale_results[s]["recall@10"] for s in scales]
        p50 = [scale_results[s]["latency"]["p50_ms"] for s in scales]

        line1, = ax1.plot(scales, recall_10, "b-o", label="Recall@10")
        line2, = ax2.plot(scales, p50, "r-s", label="P50 Latency (ms)")

        ax1.set_xlabel("Database Size")
        ax1.set_ylabel("Recall@10", color="b")
        ax2.set_ylabel("P50 Latency (ms)", color="r")
        ax1.set_title("Scale Analysis: Recall and Latency vs. Database Size")
        ax1.set_xscale("log")

        lines = [line1, line2]
        ax1.legend(lines, [l.get_label() for l in lines], loc="center right")

        fig.tight_layout()
        fig.savefig(f"{self.output_dir}/scale_analysis.png", dpi=150)
        plt.close(fig)

    def embedding_visualization(self, embeddings_2d: np.ndarray,
                                labels: np.ndarray = None,
                                title: str = "Embedding Space Visualization"):
        """2D scatter of embeddings (after t-SNE/UMAP reduction)."""
        fig, ax = plt.subplots(figsize=(10, 8))

        if labels is not None:
            scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                 c=labels, cmap="tab20", alpha=0.6, s=10)
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                       alpha=0.5, s=10)

        ax.set_title(title)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

        fig.tight_layout()
        fig.savefig(f"{self.output_dir}/embedding_viz.png", dpi=150)
        plt.close(fig)
