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
    emb = np.load(PROJECT_ROOT / "data" / "embeddings" / "flickr30k_full" / "image_embeddings.npy")
    from src.indexing.hnsw_lib import HNSWLib
    print(f"[init] HNSW on {len(emb)} vectors...")
    idx = HNSWLib(dim=emb.shape[1], M=32, ef_construction=300, ef_search=200, metric="cosine")
    idx.timed_build(emb)
    _state["index"] = idx
    _state["img_emb"] = emb


def _load_bm25():
    if "bm25" in _state:
        return
    from src.retrieval.bm25_index import BM25Index
    udf = _state["df"]
    uimgs = _state["unique_imgs"]
    _state["bm25"] = BM25Index(k1=1.5, b=0.75)
    _state["bm25"].build([udf[udf["image_path"] == p]["caption"].iloc[0] for p in uimgs])
    print(f"[init] BM25: {_state['bm25'].num_docs} docs")


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
        msg = f"✅ Ready: {n} images indexed, HNSW+BM25 loaded"
        if llm:
            msg += f", LLM: {llm.model}"
        else:
            msg += " (LLM not configured - set LLM_API_KEY in .env)"
        return msg
    except Exception as e:
        import traceback
        return f"❌ {e}\n{traceback.format_exc()}"


def on_text_search(query, strategy):
    if "index" not in _state:
        return [], "❌ Click Initialize first", [], "", ""
    if not query.strip():
        return [], "Enter a query", [], "", ""

    enc = _state["encoder"]
    qv = enc.encode_texts([query])[0]
    rows, info, llm_out = [], "", ""

    if strategy == "Vector (HNSW)":
        ids, sc, lat = _dense_search(qv)
        info = f"HNSW vector search | {lat:.1f}ms"
    elif strategy == "BM25 (Keyword)":
        ids, sc = _sparse_search(query)
        info = "BM25 keyword search"
    else:
        ids, sc, wd, ws, qt = _hybrid_search(query, qv)
        info = f"Hybrid Adaptive | dense={wd:.2f} sparse={ws:.2f} | query_type={qt.value}"

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
        return "❌ Click Initialize first", "", ""
    if image is None:
        return "Upload an image", "", ""

    enc = _state["encoder"]
    qv = enc.encode_images([image])[0]
    ids, sc, lat = _dense_search(qv, k=10)
    uimgs = _state["unique_imgs"]

    txt = f"### Image-to-Text Results ({lat:.1f}ms)\n\n"
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

    return f"Search latency: {lat:.1f}ms", txt, llm_out


# ── UI ──────────────────────────────────────────────────────────────

def create_ui():
    with gr.Blocks(title="Cross-Modal RAG", theme=gr.themes.Soft(primary_hue="indigo")) as app:
        gr.Markdown("""
        # Cross-Modal Hybrid RAG System
        **CLIP ViT-B/32 + HNSW + BM25 + Adaptive Fusion + LLM**

        Text → Image retrieval with AI-generated answers. Configure `.env` for LLM.
        """)

        with gr.Row():
            init_btn = gr.Button("Initialize System", variant="primary", size="sm")
            init_msg = gr.Textbox(label="Status", interactive=False, scale=3)

        gr.Markdown("---")

        with gr.Tabs():
            with gr.TabItem("Text-to-Image"):
                with gr.Row():
                    with gr.Column(scale=3):
                        q = gr.Textbox(label="Query", placeholder="Describe an image...", lines=2)
                        with gr.Row():
                            strat = gr.Radio(["Vector (HNSW)", "BM25 (Keyword)", "Hybrid (Adaptive)"],
                                             value="Hybrid (Adaptive)", label="Strategy")
                            btn = gr.Button("Search", variant="primary")
                        sinfo = gr.Textbox(label="Info", interactive=False)
                        llm = gr.Textbox(label="LLM Answer", lines=5, interactive=False, placeholder="(set LLM_API_KEY in .env)")
                    with gr.Column(scale=2):
                        gal = gr.Gallery(label="Retrieved Images", columns=2, height=400, object_fit="contain")
                with gr.Accordion("Details", open=False):
                    tbl = gr.Dataframe(headers=["Rank", "Score", "Caption"], label="Results")
                    capt = gr.Textbox(label="Captions", lines=6, interactive=False)

            with gr.TabItem("Image-to-Text"):
                with gr.Row():
                    with gr.Column(scale=2):
                        img_in = gr.Image(label="Upload Image", type="pil", height=300)
                        ibtn = gr.Button("Find Similar", variant="primary")
                        ilat = gr.Textbox(label="Latency", interactive=False)
                    with gr.Column(scale=3):
                        ires = gr.Markdown("Results will appear here...")
                        illm = gr.Textbox(label="LLM Analysis", lines=5, interactive=False)

            with gr.TabItem("Setup"):
                gr.Markdown("""
                ### How to configure LLM
                Copy the example file and fill in your API key:
                ```bash
                cp .env.example .env
                # Edit .env with your credentials
                ```

                **Supported providers** (any OpenAI-compatible API):

                | Provider | LLM_BASE_URL | LLM_MODEL |
                |----------|-------------|-----------|
                | DeepSeek | https://api.deepseek.com/v1 | deepseek-chat |
                | 通义千问 | https://dashscope.aliyuncs.com/compatible-mode/v1 | qwen-plus |
                | OpenAI | https://api.openai.com/v1 | gpt-4o-mini |
                | SiliconFlow | https://api.siliconflow.cn/v1 | Qwen/Qwen2.5-7B-Instruct |

                ### Requirements
                ```bash
                pip install gradio openai python-dotenv
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
    print(f"\n  Gradio: http://{host}:{port}")
    print("  Press Ctrl+C to stop\n")
    app.launch(server_name=host, server_port=port, share=False)


if __name__ == "__main__":
    main()
