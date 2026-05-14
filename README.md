# 跨模态混合检索增强生成 (RAG) 系统

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-orange.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

面向多模态数据的**混合检索增强生成（RAG）系统**，支持文本搜图和图搜文本。集成 CLIP 跨模态嵌入、4种向量索引算法（HNSW / FAISS IVF-PQ / LSH / Annoy），创新性地提出**自适应混合检索策略**（稠密向量 + 稀疏 BM25 + 动态权重分配），结合大模型 API 实现端到端的检索与智能生成。

> 大数据课程设计项目

## 系统架构

```
用户查询 (文本/图片)
       │
       ▼
┌─────────────────┐
│  CLIP ViT-B/32  │ ← 多模态编码（统一向量空间）
└────────┬────────┘
         │ 512维向量
         ▼
┌─────────────────────────────────────────┐
│          混合检索引擎                     │
│  ┌──────────┐  ┌──────┐  ┌───────────┐ │
│  │ HNSW向量 │  │ BM25 │  │ 查询路由器  │ │
│  │ 稠密检索 │ +│ 稀疏 │ +│ 自适应权重  │ │
│  └──────────┘  └──────┘  └───────────┘ │
│         │           │           │        │
│         └───────────┴───────────┘        │
│                     │                    │
│              ▼ 分数融合 ▼                │
│               Top-K 候选                 │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────┐
│   LLM 生成 (API)    │ ← DeepSeek / 通义千问 / OpenAI
│ RAG Prompt + 上下文  │
└──────────┬──────────┘
           │
           ▼
      最终回答 + 检索图片
```

## 核心特性

- **多模态检索**：文本搜图 + 图搜文本，CLIP 统一嵌入空间（512d）
- **4种向量索引**：HNSW (hnswlib C++)、FAISS IVF-PQ、余弦 LSH (SimHash)、Annoy —— 完整对比实验
- **自适应混合检索**（创新点）：查询类型分类 + 动态权重分配 + 稠密/稀疏融合，R@1 提升 77%
- **中文友好**：中文查询自动翻译为英文检索，LLM 稳定中文回答
- **LLM API 灵活配置**：支持 DeepSeek、通义千问、OpenAI、SiliconFlow 等兼容接口
- **索引持久化**：首次构建后自动保存，重启秒级加载
- **Gradio Web 界面**：可视化检索结果画廊 + LLM 智能回答

## 快速开始

### 1. 环境准备

```bash
# 创建并激活 conda 环境
conda create -n bigdata_rag python=3.11 -y
conda activate bigdata_rag

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 LLM API

```bash
cp .env.example .env
# 编辑 .env，填入你的 API 密钥
```

`.env` 文件内容示例：

```ini
LLM_API_KEY=sk-your-api-key-here
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

支持的 LLM 提供商：

| 提供商 | LLM_BASE_URL | LLM_MODEL |
|--------|-------------|-----------|
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| 通义千问 (DashScope) | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| SiliconFlow (免费额度) | `https://api.siliconflow.cn/v1` | `Qwen/Qwen2.5-7B-Instruct` |

### 3. 准备数据

```bash
# 下载 Flickr30k 数据集（~4.4GB）
python scripts/download_flickr30k.py

# 运行完整数据处理流水线
python scripts/run_data_pipeline.py --stage all
```

流水线包含 4 个阶段：

1. **Spark 预处理**：CSV → 清洗 → Parquet 列存
2. **CLIP 嵌入**：图片 + 文本 → 512维向量 → `.npy`
3. **构建 BM25**：29K 条图片描述 → 稀疏倒排索引
4. **构建 HNSW**：29K 个 512维向量 → 分层导航图索引

### 4. 启动 Web 界面

```bash
python -m src.ui.app
# 浏览器打开 http://127.0.0.1:7860
```

首次运行时，**点击「初始化系统」按钮**：
- 首次：构建 HNSW 索引（~200 秒）并自动保存到磁盘
- 再次：从磁盘加载索引（~5 秒）

### 5. 使用方式

**文本搜图**（Text-to-Image）：
- 输入英文描述（精准）或中文描述（自动翻译）
- 选择检索策略：混合检索（自适应融合）效果最佳
- 查看检索结果图片和 LLM 智能回答

**图搜文本**（Image-to-Text）：
- 上传任意图片（非训练集图片）
- 系统用 CLIP 编码图片，在 29K 图片库中找最相似的
- LLM 分析检索结果

## 实验成果

完整实验覆盖 6 个研究问题，基于 Flickr30k 真实数据集（29K 图片, 145K 标注）：

| 实验 | 研究问题 | 核心发现 |
|------|---------|---------|
| **E1: ANN 对比** | 4种向量索引算法在跨模态数据上的表现 | HNSW (hnswlib) 综合最优：R@10=54.8%, <1ms/query |
| **E2: 降维影响** | PCA/UMAP 降维对检索质量的影响 | 跨模态检索对降维极敏感，PCA 256d 丢失 ~38% R@10 |
| **E3: 混合检索** | 稠密+稀疏融合能否超越纯向量检索？ | 自适应混合 R@1=9.2% vs 纯向量 R@1=5.2%，提升 **77%** |
| **E4: 跨模态对称** | 文搜图 vs 图搜文双向对比 | I→T 召回 72.3% > T→I 55.0%，图文不对称 |
| **E5: RAG 质量** | LLM 生成的回答质量评估 | 人工评分（相关性/准确性/有用性） |
| **E6: 规模扩展** | 1K→29K 规模下延迟和召回变化 | HNSW 延迟亚线性增长，全量 <1ms |

详细实验设置和结果分析见 [`report/technical_summary.md`](report/technical_summary.md)。

## 项目结构

```
BigData_Design/
├── config/                    # YAML 配置文件（实验参数、模型配置）
├── src/
│   ├── data/                  # 数据处理（Spark 预处理、PyTorch Dataset）
│   ├── embedding/             # 多模态嵌入（CLIP 编码、PCA/UMAP 降维）
│   ├── indexing/              # 向量索引（FAISS/HNSW/LSH/Annoy，统一接口）
│   ├── retrieval/             # 混合检索（查询路由、BM25、分数融合）
│   ├── rag/                   # LLM 生成（API 调用、Prompt 模板、上下文构建）
│   ├── evaluation/            # 评估（指标计算、实验运行、可视化）
│   └── ui/                    # Gradio Web 界面
├── scripts/                   # 数据流水线 & 实验运行脚本
├── tests/                     # 单元测试（14项，pytest）
├── report/                    # 课程报告 & 实验图表
│   ├── technical_summary.md   # 技术总结文档
│   └── figures/               # 可视化图表
├── data/                      # 数据目录（gitignored）
│   ├── raw/                   # 原始数据集
│   ├── embeddings/            # CLIP 向量 (.npy + .parquet)
│   └── indexes/               # 持久化索引文件
├── .env.example               # API 配置模板
├── requirements.txt           # Python 依赖
└── CLAUDE.md                  # AI 助手指南
```

## 运行测试

```bash
# 运行全部测试（14项，3.4秒）
pytest tests/ -v

# 按模块运行
pytest tests/test_indexing.py -v    # 索引模块（FAISS/HNSW/LSH）
pytest tests/test_retrieval.py -v   # 检索模块（BM25/查询路由/融合）
pytest tests/test_rag.py -v         # RAG模块（上下文构建/Prompt模板）
pytest tests/test_embedding.py -v   # 嵌入模块（需要下载CLIP模型）
```

## 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| 数据处理 | Apache Spark (PySpark) | 分布式预处理、Parquet 列存 |
| 嵌入模型 | CLIP ViT-B/32 (OpenAI) | 图片+文本 -> 512维归一化向量 |
| 向量索引 | HNSW (hnswlib) / FAISS / LSH / Annoy | ANN 近似最近邻搜索 |
| 稀疏检索 | BM25 (Okapi) | TF-IDF 关键词匹配 |
| 查询路由 | 启发式特征 + IDF 分析 | 自适应分配稠密/稀疏权重 |
| 分数融合 | RRF / Weighted Sum | 多路结果融合排序 |
| LLM 生成 | OpenAI 兼容 API | 检索结果增强生成 |
| Web 界面 | Gradio 4.x | 交互式检索演示 |
| 评估 | Recall@K / mAP / MRR / P50/P95 | 6组对照实验 |

## 创新点

1. **自适应混合检索**：结合 DAT (Hsu & Tzeng, 2025) 的查询路由思想与 RRF (Cormack et al., SIGIR 2009) 的无参融合，实现查询类型感知的动态权重分配
2. **手写 HNSW + LSH**：纯 Python 实现分层导航小世界图和余弦 LSH，体现对底层算法的深入掌握
3. **跨模态对称性分析**：首次系统对比文搜图 vs 图搜文在相同嵌入空间中的检索不对称性
4. **端到端工程集成**：Spark → CLIP → ANN → BM25 → LLM 全链路打通，可演示可交付

## 后续扩展方向

- **向量数据库集成**：接入 Milvus / Qdrant 实现生产级检索服务
- **Graph RAG**：构建图文知识图谱，支持多跳推理
- **多轮对话**：支持上下文的交互式检索对话
- **跨语言 CLIP**：替换为多语言 CLIP（`sentence-transformers/clip-ViT-B-32-multilingual-v1`）原生支持中文
- **Workshop 投稿**：整理实验结果，投稿相关 workshop

## 参考文献

| 论文 | 出处 | 本课题对应 |
|------|------|-----------|
| Radford et al. "Learning Transferable Visual Models From Natural Language Supervision" | ICML 2021 | CLIP 嵌入 |
| Malkov & Yashunin. "Efficient and Robust ANN Search Using HNSW Graphs" | IEEE TPAMI, 2018 | HNSW 索引 |
| Johnson, Douze, Jégou. "Billion-Scale Similarity Search with GPUs" | IEEE Trans. Big Data, 2019 | FAISS IVF-PQ |
| Charikar. "Similarity Estimation Techniques from Rounding Algorithms" | STOC 2002 | LSH / SimHash |
| Cormack, Clarke, Büttcher. "RRF Outperforms Condorcet" | SIGIR 2009 | RRF 融合 |
| Hsu & Tzeng. "DAT: Dynamic Alpha Tuning for Hybrid Retrieval" | arXiv 2503.23013, 2025 | 自适应权重 |
