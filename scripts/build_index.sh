#!/bin/bash
# Build all ANN vector indexes from extracted embeddings
# Usage: bash scripts/build_index.sh

set -e

echo "=== Building Vector Indexes ==="

python -c "
import numpy as np
from pathlib import Path

# Load embeddings
emb_path = 'data/embeddings/embeddings_image.npy'
if not Path(emb_path).exists():
    print(f'[WARN] Embeddings not found: {emb_path}')
    print('Run scripts/extract_embeddings.sh first')
    exit(1)

image_embs = np.load(emb_path)
print(f'Loaded embeddings: {image_embs.shape}')

dim = image_embs.shape[1]
n = len(image_embs)

# 1. FAISS IVF-PQ
print('[1/4] Building FAISS IVF-PQ index...')
from src.indexing.faiss_index import FAISSIndex
faiss_idx = FAISSIndex(dim=dim, index_type='ivfpq')
t = faiss_idx.timed_build(image_embs)
print(f'  Build time: {t:.2f}s')
faiss_idx.save('data/indexes/faiss_ivfpq')

# 2. Manual HNSW
print('[2/4] Building Manual HNSW index...')
from src.indexing.hnsw_manual import ManualHNSW
hnsw_manual = ManualHNSW(dim=dim, M=16, ef_construction=200)
t = hnsw_manual.timed_build(image_embs)
print(f'  Build time: {t:.2f}s')
hnsw_manual.save('data/indexes/hnsw_manual.pkl')

# 3. LSH
print('[3/4] Building LSH index...')
from src.indexing.lsh import CosineLSH
lsh = CosineLSH(dim=dim, n_tables=10, n_hashes=16)
t = lsh.timed_build(image_embs)
print(f'  Build time: {t:.2f}s')
lsh.save('data/indexes/lsh.pkl')

# 4. Annoy
print('[4/4] Building Annoy index...')
from src.indexing.annoy_index import AnnoyIndexWrapper
annoy = AnnoyIndexWrapper(dim=dim, n_trees=10)
t = annoy.timed_build(image_embs)
print(f'  Build time: {t:.2f}s')
annoy.save('data/indexes/annoy')

print('[OK] All indexes built and saved to data/indexes/')
"

echo "=== Done ==="
