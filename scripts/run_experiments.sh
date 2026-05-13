#!/bin/bash
# Run all benchmark experiments
# Usage: bash scripts/run_experiments.sh

set -e

echo "=== Running Retrieval Benchmark Experiments ==="

python -c "
import numpy as np
from src.evaluation.experiments import ExperimentRunner
from src.indexing.faiss_index import FAISSIndex
from src.indexing.hnsw_manual import ManualHNSW
from src.indexing.hnsw_lib import HNSWLib
from src.indexing.lsh import CosineLSH
from src.indexing.annoy_index import AnnoyIndexWrapper
from pathlib import Path

# Load embeddings
emb_path = 'data/embeddings/embeddings_image.npy'
if not Path(emb_path).exists():
    print('[WARN] Embeddings not found. Run extract_embeddings.sh first.')
    exit(1)

embeddings = np.load(emb_path)
dim = embeddings.shape[1]
n = len(embeddings)

print(f'Embeddings: {n} x {dim}')

# Sample test queries
np.random.seed(42)
test_indices = np.random.choice(n, min(100, n), replace=False)
test_queries = embeddings[test_indices]
ground_truth = test_indices  # Self-retrieval: each query should retrieve itself

runner = ExperimentRunner(output_dir='results')

# Experiment 1: ANN Comparison
print('\n=== E1: ANN Algorithm Comparison ===')
indices = {
    'FAISS-IVFPQ': FAISSIndex(dim=dim, index_type='ivfpq'),
    'HNSW-Manual': ManualHNSW(dim=dim, M=16),
    'HNSW-lib': HNSWLib(dim=dim, M=16),
    'LSH': CosineLSH(dim=dim, n_tables=10, n_hashes=16),
    'Annoy': AnnoyIndexWrapper(dim=dim, n_trees=10),
}

for name, idx in indices.items():
    print(f'  Building {name}...')
    idx.timed_build(embeddings[:5000])  # Use subset for speed

e1_results = runner.run_ann_comparison(
    indices, test_queries, ground_truth
)
runner.save_results(e1_results, 'e1_ann_comparison.json')

# Experiment 2: Dimensionality Reduction
print('\n=== E2: Dimensionality Reduction ===')
e2_results = runner.run_dim_reduction_experiment(
    embeddings[:5000], test_queries, ground_truth,
    dims=[64, 128, 256]
)
runner.save_results(e2_results, 'e2_dim_reduction.json')

# Experiment 6: Scale Analysis
print('\n=== E6: Scale Analysis ===')
e6_results = runner.run_scale_experiment(
    HNSWLib, embeddings, test_queries, ground_truth,
    scales=[1000, 5000, 10000]
)
runner.save_results(e6_results, 'e6_scale_analysis.json')

print('\n=== All experiments complete ===')
print('Results saved in results/ directory')
"

echo "=== Done ==="
