# Cross-Modal Hybrid RAG System

## 项目简介

面向多模态数据的混合检索增强生成（RAG）系统。支持文本搜图和图搜文本，集成多种向量索引算法（HNSW / LSH / FAISS IVF-PQ / Annoy），并实现自适应混合检索策略（稠密向量 + 稀疏BM25 + 元数据过滤），结合本地LLM（Qwen2.5-7B）实现端到端的检索与生成。

## 技术栈

- **数据处理**: Apache Spark (PySpark)
- **向量嵌入**: CLIP ViT-B/32 (OpenAI / Transformers)
- **向量索引**: FAISS / 手写 HNSW / 手写 LSH / Annoy
- **混合检索**: BM25 + 稠密向量 + 自适应查询路由 + RRF 融合
- **LLM生成**: Qwen2.5-7B-Instruct (GPTQ 4-bit)
- **Web界面**: Gradio

## 项目结构

```
cross_modal_rag/
├── config/                  # YAML配置（模型路径、实验参数）
├── src/
│   ├── data/                # 数据处理（Spark预处理、PyTorch Dataset）
│   ├── embedding/           # 多模态嵌入（CLIP编码、降维）
│   ├── indexing/            # 向量索引（FAISS/HNSW/LSH/Annoy）
│   ├── retrieval/           # 混合检索（查询路由、BM25、融合策略）
│   ├── rag/                 # LLM生成（Qwen推理、Prompt模板）
│   ├── evaluation/          # 评估（指标计算、实验运行、可视化）
│   └── ui/                  # Gradio Web界面
├── scripts/                 # 一键运行脚本
├── notebooks/               # Jupyter分析笔记
├── tests/                   # 单元测试
├── data/                    # 数据目录（不入git）
│   ├── raw/                 # 原始数据集
│   ├── processed/           # Spark处理后Parquet
│   ├── embeddings/          # 预计算CLIP向量
│   └── indexes/             # 索引文件
└── report/                  # 课程设计报告
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 下载数据集
bash scripts/download_data.sh

# 数据预处理
bash scripts/run_preprocessing.sh

# 提取特征向量
bash scripts/extract_embeddings.sh

# 构建索引
bash scripts/build_index.sh

# 运行实验
bash scripts/run_experiments.sh

# 启动Web界面
python src/ui/app.py
```
