from setuptools import setup, find_packages

setup(
    name="cross_modal_rag",
    version="0.1.0",
    description="Multi-Modal Hybrid Retrieval-Augmented Generation System",
    author="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "faiss-cpu>=1.7.4",
        "hnswlib>=0.7.0",
        "annoy>=1.17.0",
        "rank-bm25>=0.2.2",
        "pyspark>=3.5.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "umap-learn>=0.5.4",
        "gradio>=4.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.13.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
    ],
)
