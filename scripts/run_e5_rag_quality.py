"""
E5: RAG Generation Quality Evaluation
50 text-to-image + 50 image-to-text queries = 100 evaluation samples.
Each sample records: query, top-5 retrieval results, LLM answer.
Human scoring (relevance/accuracy/usefulness, 1-5) to be filled later.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


# ── Helpers ─────────────────────────────────────────────────────────
def _to_native(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


# ── 50 Diverse Text Queries ────────────────────────────────────────
TEXT_QUERIES = [
    # People & Activities (10)
    "a man playing guitar on a stage",
    "children playing soccer in a park",
    "a woman reading a book on a bench",
    "people dancing at a wedding party",
    "a chef cooking in a kitchen",
    "a musician performing with a band",
    "a person walking a dog on the street",
    "kids swimming in a pool",
    "a man in a suit giving a presentation",
    "two people having coffee at a cafe",
    # Animals (8)
    "a brown dog running in the grass",
    "a cat sleeping on a sofa",
    "a horse galloping in a field",
    "birds flying over the ocean",
    "a fish swimming in an aquarium",
    "a squirrel eating a nut on a tree",
    "two dogs playing together",
    "a white cat sitting by the window",
    # Nature & Landscapes (8)
    "a mountain landscape at sunset",
    "a beach with waves and palm trees",
    "a forest path in autumn colors",
    "a river flowing through a valley",
    "snow-covered mountains under blue sky",
    "a field of sunflowers in summer",
    "a lake reflecting the sky at dawn",
    "waterfall in a tropical rainforest",
    # Urban & City (8)
    "people walking on a busy city street",
    "a red double-decker bus on the road",
    "cars stuck in traffic on a highway",
    "a street market with fruit stands",
    "a modern skyscraper against the sky",
    "a cyclist riding through the city",
    "a subway train arriving at the station",
    "outdoor dining at a restaurant patio",
    # Sports (6)
    "a soccer player kicking a ball",
    "a basketball player making a dunk",
    "a surfer riding a big wave",
    "a tennis player serving the ball",
    "a skier going down a snowy slope",
    "a person doing yoga on a mat",
    # Food & Dining (5)
    "a table with pizza and drinks",
    "fresh fruits displayed at a market",
    "a birthday cake with candles",
    "a person eating sushi with chopsticks",
    "a barbecue grill with meat cooking",
    # Vehicles & Transportation (5)
    "a red sports car parked on the street",
    "an airplane flying in the blue sky",
    "a sailboat on calm water",
    "a train passing through countryside",
    "a motorcycle rider on a winding road",
]


def main():
    print("=" * 60)
    print("E5: RAG Generation Quality Evaluation")
    print(f"  Text queries: {len(TEXT_QUERIES)}")
    print(f"  Image queries: 50 (sampled from Flickr30k)")
    print("=" * 60)

    # ── Load all components ────────────────────────────────────────
    print("\n[Loading data]")
    d = PROJECT_ROOT / "data" / "embeddings" / "flickr30k_full"
    df = pd.read_parquet(d / "df.parquet")
    uimgs = sorted(df["image_path"].unique())
    print(f"  {len(df)} rows, {len(uimgs)} unique images")

    print("\n[Loading CLIP]")
    from src.embedding.clip_encoder import CLIPEncoder
    m = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
    dev = os.getenv("CLIP_DEVICE", "cuda")
    enc = CLIPEncoder(model_name=m, device=dev, normalize=True)
    print(f"  {m} on {dev}")

    print("\n[Loading HNSW]")
    from src.indexing.hnsw_lib import HNSWLib
    hnsw_path = str(PROJECT_ROOT / "data" / "indexes" / "flickr30k" / "hnsw_lib")
    if os.path.exists(f"{hnsw_path}.hnsw"):
        hnsw = HNSWLib.load(hnsw_path)
        print("  Loaded from disk")
    else:
        emb = np.load(PROJECT_ROOT / "data" / "embeddings" / "flickr30k_full" / "image_embeddings.npy")
        idx = HNSWLib(dim=emb.shape[1], M=32, ef_construction=300, ef_search=200, metric="cosine")
        idx.timed_build(emb)
        idx.save(hnsw_path)
        hnsw = idx
        print(f"  Built and saved ({len(emb)} vectors)")

    print("\n[Loading BM25]")
    from src.retrieval.bm25_index import BM25Index
    bm25_path = str(PROJECT_ROOT / "data" / "indexes" / "flickr30k" / "bm25.pkl")
    if os.path.exists(bm25_path):
        bm25 = BM25Index.load(bm25_path)
        print(f"  Loaded from disk ({bm25.num_docs} docs)")
    else:
        bm25 = BM25Index(k1=1.5, b=0.75)
        bm25.build([df[df["image_path"] == p]["caption"].iloc[0] for p in uimgs])
        bm25.save(bm25_path)
        print(f"  Built and saved ({bm25.num_docs} docs)")

    print("\n[Loading LLM]")
    from src.rag.llm_generator import LLMGenerator
    llm = None
    try:
        llm = LLMGenerator()
        llm.connect()
        print(f"  {llm.model}")
    except Exception as e:
        print(f"  Skipped: {e}")

    from src.retrieval.query_router import QueryRouter
    from src.retrieval.fusion import ScoreFusion
    router = QueryRouter(bm25_index=bm25)
    fusion = ScoreFusion()

    # ── Text-to-Image ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[1/2] Text-to-Image: 50 queries")
    print("=" * 60)
    t2i_results = []

    for i, query in enumerate(TEXT_QUERIES):
        # Encode
        qv = enc.encode_texts([query])[0]

        # Dense search
        dense_out = hnsw.timed_search(qv, k=200)
        d_ids = np.atleast_1d(np.asarray(dense_out["ids"]).ravel())[:200]
        d_scores = 1.0 - np.atleast_1d(np.asarray(dense_out["distances"]).ravel())[:200]

        # Sparse search
        s_ids, s_scores_raw = bm25.search(query, k=200)

        # Fusion
        w_d, w_s, w_m = router.get_weights(query)
        q_type = router.classify(query)
        uid = sorted(set(d_ids) | set(int(s) for s in s_ids))
        m_idx = {v: j for j, v in enumerate(uid)}
        da = np.zeros(len(uid))
        sa = np.zeros(len(uid))
        for j, did in enumerate(d_ids):
            if int(did) in m_idx:
                da[m_idx[int(did)]] = d_scores[j]
        for j, sid in enumerate(s_ids):
            if int(sid) in m_idx:
                sa[m_idx[int(sid)]] = s_scores_raw[j]

        fused = fusion.weighted_sum(da, sa, weights=(w_d, w_s, w_m))
        top = np.argsort(-fused)[:5]

        # Format results
        top5 = []
        for rank, ti in enumerate(top):
            fid = int(uid[ti])
            if 0 <= fid < len(uimgs):
                cap = df[df["image_path"] == uimgs[fid]]["caption"].iloc[0]
                top5.append({"rank": rank + 1, "score": round(float(fused[ti]), 4), "caption": str(cap)})

        llm_out = ""
        if llm and top5:
            try:
                llm_in = [{"rank": r["rank"], "score": r["score"], "caption": r["caption"]} for r in top5]
                llm_out = llm.generate(query, llm_in)
            except Exception as e:
                llm_out = f"[LLM Error: {e}]"

        result = {
            "query": query,
            "query_type": q_type.value,
            "weights": {"dense": round(float(w_d), 3), "sparse": round(float(w_s), 3)},
            "latency_ms": round(float(dense_out["latency_ms"]), 1),
            "top_results": top5,
            "llm_answer": llm_out,
            "human_scores": {"relevance": None, "accuracy": None, "usefulness": None},
        }
        t2i_results.append(_to_native(result))
        top_cap = top5[0]["caption"][:60] if top5 else "?"
        llm_preview = llm_out[:60].replace("\n", " ") if llm_out else "(none)"
        print(f"  [{i+1:2d}/50] {query[:50]}... | top1={top_cap} | LLM={llm_preview}")

    # ── Image-to-Text ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[2/2] Image-to-Text: 50 queries")
    print("=" * 60)
    i2t_results = []

    # Sample 50 images not used in text queries (use different seed)
    rng = np.random.RandomState(42)
    sample_ix = sorted(rng.choice(len(uimgs), size=50, replace=False))

    for i, orig_idx in enumerate(sample_ix):
        path = uimgs[orig_idx]
        if not os.path.exists(path):
            print(f"  [{i+1:2d}/50] SKIP: file not found {path}")
            continue

        img = Image.open(path).convert("RGB")
        qv = enc.encode_images([img])[0]
        dense_out = hnsw.timed_search(qv, k=10)
        ids_arr = np.atleast_1d(np.asarray(dense_out["ids"]).ravel())[:10]
        scores_arr = 1.0 - np.atleast_1d(np.asarray(dense_out["distances"]).ravel())[:10]

        top5 = []
        for rank, (iid, sc) in enumerate(zip(ids_arr, scores_arr)):
            fid = int(iid)
            if 0 <= fid < len(uimgs):
                caps = df[df["image_path"] == uimgs[fid]]["caption"].head(3).tolist()
                top5.append({"rank": rank + 1, "score": round(float(sc), 4), "captions": [str(c) for c in caps]})

        llm_out = ""
        if llm and top5:
            try:
                llm_in = [{"rank": r["rank"], "score": r["score"], "caption": r["captions"][0]} for r in top5]
                llm_out = llm.generate("image query", llm_in, prompt_type="image_query")
            except Exception as e:
                llm_out = f"[LLM Error: {e}]"

        result = {
            "image_path": str(path),
            "image_index": int(orig_idx),
            "latency_ms": round(float(dense_out["latency_ms"]), 1),
            "top_results": top5,
            "llm_answer": llm_out,
            "human_scores": {"relevance": None, "accuracy": None, "usefulness": None},
        }
        i2t_results.append(_to_native(result))
        sc0 = top5[0]["score"] if top5 else 0
        self_flag = "(SELF)" if sc0 > 0.99 else ""
        print(f"  [{i+1:2d}/50] {Path(path).name[:40]} cos={sc0:.4f} {self_flag}")

    # ── Save ───────────────────────────────────────────────────────
    output = {
        "description": "E5: RAG Generation Quality Evaluation",
        "dataset": "Flickr30k (29K images, 145K captions)",
        "num_text_queries": len(t2i_results),
        "num_image_queries": len(i2t_results),
        "scoring_guide": {
            "relevance": "1=完全无关 2=略有关联 3=基本相关 4=比较相关 5=高度相关",
            "accuracy": "1=完全错误 2=多处错误 3=基本准确 4=比较准确 5=非常准确",
            "usefulness": "1=完全无用 2=用处不大 3=基本有用 4=比较有用 5=非常有用",
        },
        "text_to_image": t2i_results,
        "image_to_text": i2t_results,
    }

    out_path = PROJECT_ROOT / "results" / "e5_rag_quality.json"
    # Final safety pass: ensure everything is JSON-serializable
    output = _to_native(output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f" Saved to {out_path}")
    print(f" Text queries:  {len(t2i_results)}")
    print(f" Image queries: {len(i2t_results)}")
    print(f" Total samples: {len(t2i_results) + len(i2t_results)}")


if __name__ == "__main__":
    main()
