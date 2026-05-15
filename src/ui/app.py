"""
Gradio Web UI for Cross-Modal RAG System.
Text-to-Image | Image-to-Text | Hybrid Search + LLM Generation.

Quick start:
    1. cp .env.example .env   # fill in LLM_API_KEY
    2. python -m src.ui.app
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

import gradio as gr

# ── Lazy-loaded global state ─────────────────────────────────────────
_state = {}


def _load_encoder():
    if "encoder" in _state:
        return _state["encoder"]
    from src.embedding.clip_encoder import CLIPEncoder
    m = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
    d = os.getenv("CLIP_DEVICE", "cuda")
    print(f"[init] CLIP {m} on {d}")
    _state["encoder"] = CLIPEncoder(model_name=m, device=d, normalize=True)
    return _state["encoder"]


def _load_data():
    if "img_paths" in _state:
        return
    d = PROJECT_ROOT / "data" / "embeddings" / "flickr30k_full"
    _state["df"] = pd.read_parquet(d / "df.parquet")
    _state["img_paths"] = list(pd.read_parquet(d / "image_map.parquet")["image_path"])
    _state["captions"] = list(pd.read_parquet(d / "caption_map.parquet")["caption"])
    _state["unique_imgs"] = sorted(_state["df"]["image_path"].unique())
    _state["img_to_idx"] = {p: i for i, p in enumerate(_state["unique_imgs"])}
    print(f"[init] {len(_state['img_paths'])} images, {len(_state['captions'])} captions")


def _load_index():
    if "index" in _state:
        return
    idx_dir = PROJECT_ROOT / "data" / "indexes" / "flickr30k"
    idx_dir.mkdir(parents=True, exist_ok=True)
    hnsw_path = str(idx_dir / "hnsw_lib")
    from src.indexing.hnsw_lib import HNSWLib
    if os.path.exists(f"{hnsw_path}.hnsw"):
        print(f"[init] Loading saved HNSW index...")
        idx = HNSWLib.load(hnsw_path)
        print(f"[init] HNSW loaded from disk ({idx._n_vectors if hasattr(idx, '_n_vectors') else '?'} vectors)")
    else:
        emb = np.load(PROJECT_ROOT / "data" / "embeddings" / "flickr30k_full" / "image_embeddings.npy")
        print(f"[init] Building HNSW on {len(emb)} vectors (one-time, ~200s)...")
        idx = HNSWLib(dim=emb.shape[1], M=32, ef_construction=300, ef_search=200, metric="cosine")
        idx.timed_build(emb)
        print(f"[init] Saving HNSW to disk for future fast load...")
        idx.save(hnsw_path)
    _state["index"] = idx
    _state["img_emb"] = None


def _load_bm25():
    if "bm25" in _state:
        return
    idx_dir = PROJECT_ROOT / "data" / "indexes" / "flickr30k"
    idx_dir.mkdir(parents=True, exist_ok=True)
    bm25_path = str(idx_dir / "bm25.pkl")
    from src.retrieval.bm25_index import BM25Index
    if os.path.exists(bm25_path):
        print(f"[init] Loading saved BM25 index...")
        _state["bm25"] = BM25Index.load(bm25_path)
        print(f"[init] BM25 loaded: {_state['bm25'].num_docs} docs")
    else:
        udf = _state["df"]
        uimgs = _state["unique_imgs"]
        _state["bm25"] = BM25Index(k1=1.5, b=0.75)
        _state["bm25"].build([udf[udf["image_path"] == p]["caption"].iloc[0] for p in uimgs])
        print(f"[init] BM25 built: {_state['bm25'].num_docs} docs, saving...")
        _state["bm25"].save(bm25_path)


def _load_llm():
    if "llm" in _state:
        return _state["llm"]
    from src.rag.llm_generator import LLMGenerator
    try:
        llm = LLMGenerator()
        llm.connect()
        _state["llm"] = llm
        print(f"[init] LLM: {llm.model}")
    except ValueError:
        _state["llm"] = None
        print("[init] LLM not configured")
    return _state["llm"]


# ── Search helpers ───────────────────────────────────────────────────

def _dense_search(q_vec, k=10):
    o = _state["index"].timed_search(q_vec, k=k)
    ids = np.atleast_1d(np.asarray(o["ids"]).flatten())[:k]
    dists = 1.0 - np.atleast_1d(np.asarray(o.get("distances", [0]*k)).flatten())[:k]
    return ids, dists, o["latency_ms"]


def _sparse_search(q_text, k=10):
    ids, scores = _state["bm25"].search(q_text, k=k)
    return ids[:k], scores[:k]


def _hybrid_search(q_text, q_vec, k=10):
    from src.retrieval.query_router import QueryRouter
    from src.retrieval.fusion import ScoreFusion

    router = QueryRouter(bm25_index=_state["bm25"])
    w_d, w_s, w_m = router.get_weights(q_text)
    q_type = router.classify(q_text)
    fusion = ScoreFusion()

    d_ids, d_scores, _ = _dense_search(q_vec, k=200)
    s_ids, s_scores = _sparse_search(q_text, k=200)

    uid = sorted(set(d_ids) | set(int(s) for s in s_ids))
    m = {i: idx for idx, i in enumerate(uid)}
    da = np.zeros(len(uid))
    sa = np.zeros(len(uid))
    for idx, did in enumerate(d_ids):
        if int(did) in m:
            da[m[int(did)]] = d_scores[idx]
    for idx, sid in enumerate(s_ids):
        if int(sid) in m:
            sa[m[int(sid)]] = s_scores[idx]

    fused = fusion.weighted_sum(da, sa, weights=(w_d, w_s, w_m))
    top = np.argsort(-fused)[:k]
    return np.array(uid)[top], fused[top], w_d, w_s, q_type


def _has_chinese(text: str) -> bool:
    """Check if text contains Chinese characters."""
    return any('一' <= c <= '鿿' for c in text)


def _translate_if_chinese(query: str) -> str:
    """Translate Chinese queries to English for better CLIP retrieval.
    Uses the loaded LLM; falls back to original query if LLM unavailable."""
    if not _has_chinese(query):
        return query
    llm = _state.get("llm")
    if llm is None:
        return query
    try:
        from src.rag.prompt_templates import RAGPromptBuilder
        prompt = RAGPromptBuilder.format_translate_prompt(query)
        messages = [
            {"role": "system", "content": "You are a translator. Translate Chinese to English concisely."},
            {"role": "user", "content": prompt},
        ]
        response = llm.client.chat.completions.create(
            model=llm.model,
            messages=messages,
            max_tokens=128,
            temperature=0.3,
        )
        translated = response.choices[0].message.content.strip()
        # Clean up common prefixes
        for prefix in ["English translation:", "Translation:", "English:"]:
            if translated.lower().startswith(prefix.lower()):
                translated = translated[len(prefix):].strip()
        print(f"[translate] '{query[:50]}' -> '{translated[:80]}'")
        return translated
    except Exception as e:
        print(f"[translate] failed: {e}")
        return query


def _format(ids, scores, show_images=True):
    gallery, cap_text, rows = [], "", []
    uimgs = _state["unique_imgs"]
    for rank, (iid, sc) in enumerate(zip(ids, scores)):
        if int(iid) < 0 or int(iid) >= len(uimgs):
            continue
        p = uimgs[int(iid)]
        cap = _state["df"][_state["df"]["image_path"] == p]["caption"].iloc[0]
        if show_images and Path(p).exists():
            gallery.append((p, f"#{rank+1} (score={float(sc):.4f})"))
        cap_text += f"[{rank+1}] {float(sc):.4f}  {cap}\n"
        rows.append([rank + 1, round(float(sc), 4), cap[:150]])
    return gallery, cap_text, rows


# ── Event handlers ──────────────────────────────────────────────────

def on_init():
    try:
        _load_data()
        _load_encoder()
        _load_index()
        _load_bm25()
        llm = _load_llm()
        n = len(_state["unique_imgs"])
        msg = f"✅ 系统就绪：已索引 {n} 张图片，HNSW + BM25 已加载"
        if llm:
            msg += f"，LLM 模型: {llm.model}"
        else:
            msg += "（LLM 未配置，请在 .env 中设置 LLM_API_KEY）"
        return msg
    except Exception as e:
        import traceback
        return f"❌ {e}\n{traceback.format_exc()}"


def on_text_search(query, strategy):
    if "index" not in _state:
        return [], "❌ 请先点击「初始化系统」按钮", [], "", ""
    if not query.strip():
        return [], "请输入查询文本", [], "", ""

    # Auto-translate Chinese queries to English for better CLIP retrieval
    search_query = _translate_if_chinese(query) if _has_chinese(query) else query

    enc = _state["encoder"]
    qv = enc.encode_texts([search_query])[0]
    rows, info, llm_out = [], "", ""

    if strategy == "Vector (HNSW)":
        ids, sc, lat = _dense_search(qv)
        info = f"HNSW 向量检索 | {lat:.1f}ms"
        if search_query != query:
            info += f" | 翻译: {search_query}"
    elif strategy == "BM25 (Keyword)":
        ids, sc = _sparse_search(search_query)
        info = "BM25 关键词检索"
        if search_query != query:
            info += f" | 翻译: {search_query}"
    else:
        ids, sc, wd, ws, qt = _hybrid_search(search_query, qv)
        info = f"混合自适应检索 | 向量权重={wd:.2f} 关键词权重={ws:.2f} | 查询类型={qt.value}"
        if search_query != query:
            info += f" | 翻译: {search_query}"

    gallery, cap_txt, rows = _format(ids, sc)

    # LLM
    if _state.get("llm"):
        try:
            llm_in = [{"rank": r[0], "score": r[1], "caption": r[2]} for r in rows]
            llm_out = _state["llm"].generate(query, llm_in)
        except Exception as e:
            llm_out = f"[LLM: {e}]"

    return gallery, info, rows, llm_out, cap_txt


def on_image_search(image):
    if "index" not in _state:
        return "❌ 请先点击「初始化系统」按钮", "", ""
    if image is None:
        return "请先上传一张图片", "", ""

    enc = _state["encoder"]
    qv = enc.encode_images([image])[0]
    ids, sc, lat = _dense_search(qv, k=10)
    uimgs = _state["unique_imgs"]

    txt = f"### 图搜文本结果（延迟 {lat:.1f}ms）\n\n"
    llm_in = []
    for rank, (iid, s) in enumerate(zip(ids, sc)):
        if int(iid) >= len(uimgs):
            continue
        caps = _state["df"][_state["df"]["image_path"] == uimgs[int(iid)]]["caption"].head(3)
        txt += f"**#{rank+1}** ({float(s):.4f})\n"
        for c in caps:
            txt += f"  - {c}\n"
        txt += "\n"
        llm_in.append({"rank": rank + 1, "score": float(s), "caption": caps.iloc[0] if len(caps) else ""})

    llm_out = ""
    if _state.get("llm") and llm_in:
        try:
            llm_out = _state["llm"].generate("image query", llm_in, prompt_type="image_query")
        except Exception as e:
            llm_out = f"[LLM: {e}]"

    return f"检索延迟: {lat:.1f}ms", txt, llm_out


# ── UI ──────────────────────────────────────────────────────────────

def create_ui():
    with gr.Blocks(title="跨模态混合检索增强生成系统", theme=gr.themes.Soft(primary_hue="indigo")) as app:
        gr.Markdown("""
        # 跨模态混合检索增强生成 (RAG) 系统
        **CLIP ViT-B/32 + HNSW + BM25 + 自适应融合 + LLM**

        支持文本搜图、图搜文本，结合大模型生成智能回答。请先点击 **初始化系统** 按钮加载索引。
        """)

        with gr.Row():
            init_btn = gr.Button("🚀 初始化系统", variant="primary", size="sm")
            init_msg = gr.Textbox(label="系统状态", interactive=False, scale=3)

        gr.Markdown("---")

        with gr.Tabs():
            with gr.TabItem("📝 文本搜图"):
                with gr.Row():
                    with gr.Column(scale=3):
                        q = gr.Textbox(
                            label="输入查询（支持中文，自动翻译为英文检索）",
                            placeholder="例如：a dog running on the beach / 一个在沙滩上跑步的人 / a red car parked on the street",
                            lines=2
                        )
                        with gr.Row():
                            strat = gr.Radio(
                                [("向量检索 (HNSW)", "Vector (HNSW)"),
                                 ("关键词检索 (BM25)", "BM25 (Keyword)"),
                                 ("混合检索 (自适应融合)", "Hybrid (Adaptive)")],
                                value="Hybrid (Adaptive)",
                                label="检索策略"
                            )
                            btn = gr.Button("🔍 搜索", variant="primary")
                        sinfo = gr.Textbox(label="检索信息", interactive=False)
                        llm = gr.Textbox(label="LLM 智能回答", lines=5, interactive=False, placeholder="（请先在 .env 中配置 LLM_API_KEY）")
                    with gr.Column(scale=2):
                        gal = gr.Gallery(label="检索到的图片", columns=2, height=400, object_fit="contain")
                with gr.Accordion("💡 使用技巧 & 示例查询", open=False):
                    gr.Markdown("""
                    ### 怎么提问效果好？

                    **CLIP 模型是纯英文的**，中文输入会自动翻译成英文再检索。翻译质量会影响结果。
                    直接用英文提问效果最好，中文也能用。

                    **推荐示例查询**（英文更精准）：
                    | 类别 | 查询示例 |
                    |------|---------|
                    | 人物活动 | `a man playing guitar on stage` |
                    | 人物活动 | `children playing in the park` |
                    | 动物 | `a dog catching a frisbee` |
                    | 动物 | `a cat sleeping on a sofa` |
                    | 街景 | `people walking on a busy city street` |
                    | 自然 | `a mountain landscape at sunset` |
                    | 运动 | `a soccer player kicking a ball` |

                    **中文查询也可以**（系统会自动翻译）：
                    - `一个在舞台上弹吉他的男人`
                    - `在公园里玩耍的孩子们`
                    - `一只正在接飞盘的狗`

                    **避免的提问方式**（太宽泛，检索效果差）：
                    - `找一张男人的照片` → 改成 `a man wearing a suit in an office`
                    - `有没有动物的图片` → 改成 `a brown dog running in the grass`
                    """)
                with gr.Accordion("详细结果", open=False):
                    tbl = gr.Dataframe(headers=["排名", "相似度", "图片描述"], label="检索结果列表")
                    capt = gr.Textbox(label="图片描述文本", lines=6, interactive=False)

            with gr.TabItem("🖼️ 图搜文本"):
                with gr.Row():
                    with gr.Column(scale=2):
                        img_in = gr.Image(label="上传图片（请用非训练集的图片测试）", type="pil", height=300)
                        ibtn = gr.Button("🔍 查找相似", variant="primary")
                        ilat = gr.Textbox(label="检索延迟", interactive=False)
                    with gr.Column(scale=3):
                        ires = gr.Markdown("上传图片后点击搜索，结果将显示在这里...")
                        illm = gr.Textbox(label="LLM 图片分析", lines=5, interactive=False)
                with gr.Accordion("💡 图搜文测试技巧", open=False):
                    gr.Markdown("""
                    ### 如何测试图搜文？

                    **重要：不要使用 Flickr30k 数据集中的图片！**
                    如果上传训练集图片，系统会匹配到它自己，相似度≈1.0（相当于"作弊"）。

                    **推荐测试图片来源**：
                    1. 用手机随便拍一张照片（你的桌面、窗外风景、书本等）
                    2. 从网上下载任意图片（不属于 Flickr30k）
                    3. 截图一张网页或社交媒体上的图片

                    **效果预期**：
                    - 相似度通常在 0.3~0.6 之间（取决于图片内容）
                    - 如果图片内容是常见场景（街道、海滩、人物），匹配效果较好
                    - 如果是非常小众或抽象的内容，匹配效果一般
                    """)

            with gr.TabItem("⚙️ 配置说明"):
                gr.Markdown("""
                ### 如何配置 LLM 大模型

                1. 复制环境变量模板并填入你的 API 密钥：
                ```bash
                cp .env.example .env
                # 编辑 .env 文件，填入你的 API 密钥
                ```

                2. `.env` 文件示例：
                ```ini
                LLM_API_KEY=sk-your-api-key-here
                LLM_BASE_URL=https://api.deepseek.com/v1
                LLM_MODEL=deepseek-chat
                LLM_MAX_TOKENS=512
                LLM_TEMPERATURE=0.7
                ```

                **支持的 LLM 提供商**（兼容 OpenAI API 格式即可）：

                | 提供商 | LLM_BASE_URL | LLM_MODEL |
                |--------|-------------|-----------|
                | DeepSeek | https://api.deepseek.com/v1 | deepseek-chat |
                | 通义千问 | https://dashscope.aliyuncs.com/compatible-mode/v1 | qwen-plus |
                | OpenAI | https://api.openai.com/v1 | gpt-4o-mini |
                | SiliconFlow | https://api.siliconflow.cn/v1 | Qwen/Qwen2.5-7B-Instruct |

                ### 环境依赖
                ```bash
                pip install gradio openai python-dotenv
                ```

                ### 启动方式
                ```bash
                python -m src.ui.app
                # 浏览器打开 http://127.0.0.1:7860
                ```
                """)

        # Wire events
        init_btn.click(on_init, outputs=[init_msg])
        btn.click(on_text_search, [q, strat], [gal, sinfo, tbl, llm, capt])
        ibtn.click(on_image_search, [img_in], [ilat, ires, illm])

    return app


def main():
    app = create_ui()
    host = os.getenv("GRADIO_HOST", "127.0.0.1")
    port = int(os.getenv("GRADIO_PORT", "7860"))
    print(f"\n  🌐 跨模态RAG系统已启动: http://{host}:{port}")
    print("  按 Ctrl+C 停止服务\n")
    app.launch(server_name=host, server_port=port, share=False)


if __name__ == "__main__":
    main()
