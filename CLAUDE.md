# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- Conda environment: `D:\conda_envs\bigdata_rag` (Python 3.11)
- Active when running any Python command: `conda run -p "D:\conda_envs\bigdata_rag" python ...`
- Tests: `conda run -p "D:\conda_envs\bigdata_rag" python -m pytest tests/ -v`

## Common commands

```bash
# Run all tests (except embedding tests that need CLIP model download)
conda run -p "D:\conda_envs\bigdata_rag" python -m pytest tests/test_retrieval.py tests/test_rag.py tests/test_indexing.py -v

# Run a single test
conda run -p "D:\conda_envs\bigdata_rag" python -m pytest tests/test_indexing.py::TestFAISSIndex::test_build_and_search -v

# Launch Gradio web UI
conda run -p "D:\conda_envs\bigdata_rag" python -m src.ui.app
# Then open http://127.0.0.1:7860

# Run experiments (order matters for cached embeddings)
conda run -p "D:\conda_envs\bigdata_rag" python -u scripts/run_full_scale_experiments.py --stage all   # E1+E2
conda run -p "D:\conda_envs\bigdata_rag" python scripts/run_e3_hybrid.py                            # E3
conda run -p "D:\conda_envs\bigdata_rag" python scripts/run_e4_e6.py                                # E4+E6
```

## Architecture

**Data flow**: Raw CSV/images → Spark preprocessing → CLIP ViT-B/32 (512d) → `.npy` embeddings → ANN indexes → hybrid retrieval → LLM generation

**Module dependency chain** (each depends on the previous):
`src/data/` → `src/embedding/` → `src/indexing/` → `src/retrieval/` → `src/rag/` → `src/ui/`

- **`src/data/`** — `SparkPreprocessor` (PySpark, outputs Parquet), `CrossModalDataset` (PyTorch)
- **`src/embedding/`** — `CLIPEncoder` (HuggingFace, GPU, 512d normalized), `DimReducer` (PCA/UMAP), `BatchExtractor`
- **`src/indexing/`** — All inherit `BaseIndex`: `FAISSIndex` (flat/ivfpq/hnsw), `HNSWLib` (hnswlib C++), `ManualHNSW` (~270 lines, pedagogical), `CosineLSH` (SimHash), `AnnoyIndexWrapper` — all share `build()/search()/save()/load()` interface
- **`src/retrieval/`** — `HybridRetriever` orchestrates 5 strategies. `QueryRouter` classifies queries (SEMANTIC/KEYWORD/METADATA/HYBRID) and assigns adaptive weights. `ScoreFusion` provides weighted_sum and RRF. `BM25Index` for sparse retrieval.
- **`src/rag/`** — `LLMGenerator`: OpenAI-compatible API client (DeepSeek/通义千问/OpenAI/SiliconFlow), configured via `.env`. `RAGPromptBuilder`: Chinese system prompt. `ContextBuilder`: formats retrieval hits into prompt context.
- **`src/ui/`** — Gradio app with lazy `_state` dict pattern. 3 tabs: text-to-image (with Chinese auto-translation), image-to-text, setup guide. Indexes auto-save to `data/indexes/` for fast restart.

## Key patterns

- Every script prepends `PROJECT_ROOT` to `sys.path` for `from src.xxx import ...` imports
- `BaseIndex` ABC in `indexing/` defines the interface all 4 algorithms implement
- Score alignment in hybrid retrieval: dense/sparse results mapped to a unified ID space, missing entries zero-padded, then fused
- Lazy loading in UI: each `_load_*()` checks `_state` dict before initializing
- First-run HNSW build takes ~200s but saves to `data/indexes/` for near-instant reload
- Experiment results always saved as JSON with native Python types (no numpy scalars)

## Chinese + English

- CLIP model is English-only. Chinese queries are auto-translated to English via LLM before encoding.
- All LLM prompts are in Chinese to ensure consistent Chinese responses.
- Code comments are mixed Chinese/English.

## Data files (all in `data/`, gitignored)

- `data/raw/` — original Flickr30k zip/CSV/images
- `data/embeddings/flickr30k_full/` — `image_embeddings.npy`, `df.parquet`, `image_map.parquet`, `caption_map.parquet`
- `data/indexes/` — persisted `flickr30k_hnsw.hnsw` + `.meta.pkl`, `flickr30k_bm25.pkl`
- `results/` — 5 experiment JSON files (not gitignored)
- `report/figures/` — generated charts + markdown tables
