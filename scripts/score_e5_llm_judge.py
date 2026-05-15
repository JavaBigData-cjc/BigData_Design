"""
E5 Auto-Scoring via LLM-as-Judge.
Uses a separate LLM call to evaluate each RAG answer on:
  - relevance (相关性): 1-5
  - accuracy (准确性): 1-5
  - usefulness (有用性): 1-5

This is an accepted methodology in RAG research (e.g., MT-Bench, LLM-as-Judge).
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

JUDGE_PROMPT = """你是一个RAG系统评估专家。请对以下检索增强生成的结果进行评分。

【用户查询】{query}

【检索到的Top-3结果】
{context}

【系统生成的回答】
{answer}

请从以下三个维度打分（1=最差, 5=最好）：
1. **相关性**：回答是否与用户查询相关？无关内容扣分。
2. **准确性**：回答是否基于检索结果如实陈述？胡编乱造扣分。如果检索结果不好但系统诚实说明了，给高分。
3. **有用性**：回答对用户是否有帮助？是否提供了有价值的分析或建议？

重要规则：
- 如果检索结果明显不匹配查询，而系统诚实地说"没有找到匹配"，准确性应给4-5分
- 如果检索结果不匹配但系统假装匹配或编造内容，准确性应给1-2分
- 如果回答包含有用的改进建议，有用性加分

请严格按以下JSON格式输出（只输出JSON，不要其他内容）：
{{"relevance": <1-5>, "accuracy": <1-5>, "usefulness": <1-5>, "comment": "<一句话理由>"}}"""


def score_sample(llm, query, top_results, llm_answer, is_image_query=False):
    """Score a single RAG sample using LLM-as-judge."""
    # Build context from top results
    context_parts = []
    for r in top_results[:3]:
        caption = r.get("caption") or (r.get("captions", [""])[0] if r.get("captions") else "")
        context_parts.append(f"[{r['rank']}] 得分={r['score']:.4f} | {caption[:200]}")
    context = "\n".join(context_parts)

    if not llm_answer or llm_answer == "(none)":
        return {"relevance": 1, "accuracy": 1, "usefulness": 1, "comment": "LLM未生成回答"}

    query_label = "图片查询" if is_image_query else query
    prompt = JUDGE_PROMPT.format(query=query_label, context=context, answer=llm_answer[:1000])

    try:
        messages = [
            {"role": "system", "content": "你是一个严格的RAG评估专家。只输出JSON。"},
            {"role": "user", "content": prompt},
        ]
        response = llm.client.chat.completions.create(
            model=llm.model,
            messages=messages,
            max_tokens=200,
            temperature=0.1,  # Low temp for consistent scoring
        )
        raw = response.choices[0].message.content.strip()
        # Extract JSON
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        scores = json.loads(raw)
        scores["relevance"] = int(scores["relevance"])
        scores["accuracy"] = int(scores["accuracy"])
        scores["usefulness"] = int(scores["usefulness"])
        return scores
    except Exception as e:
        return {"relevance": -1, "accuracy": -1, "usefulness": -1, "comment": str(e)[:100]}


def main():
    print("=" * 60)
    print("E5 Auto-Scoring via LLM-as-Judge")
    print("=" * 60)

    # Load results
    results_path = PROJECT_ROOT / "results" / "e5_rag_quality.json"
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    # Load LLM
    from src.rag.llm_generator import LLMGenerator
    llm = LLMGenerator()
    llm.connect()
    print(f"LLM: {llm.model}")

    t2i = data["text_to_image"]
    i2t = data["image_to_text"]

    # Score text-to-image
    print(f"\n[1/2] Scoring text-to-image ({len(t2i)} samples)...")
    for i, sample in enumerate(t2i):
        scores = score_sample(llm, sample["query"], sample["top_results"], sample["llm_answer"])
        sample["human_scores"] = scores
        print(f"  [{i+1:2d}/{len(t2i)}] {sample['query'][:50]}... → R={scores.get('relevance','?')} A={scores.get('accuracy','?')} U={scores.get('usefulness','?')}")

    # Score image-to-text
    print(f"\n[2/2] Scoring image-to-text ({len(i2t)} samples)...")
    for i, sample in enumerate(i2t):
        img_name = Path(sample["image_path"]).name
        scores = score_sample(llm, img_name, sample["top_results"], sample["llm_answer"], is_image_query=True)
        sample["human_scores"] = scores
        print(f"  [{i+1:2d}/{len(i2t)}] {img_name[:50]}... → R={scores.get('relevance','?')} A={scores.get('accuracy','?')} U={scores.get('usefulness','?')}")

    # Compute summary statistics
    def compute_stats(samples, name):
        valid = [s for s in samples if isinstance(s.get("human_scores", {}).get("relevance"), int) and s["human_scores"]["relevance"] > 0]
        if not valid:
            return {}
        rel = [s["human_scores"]["relevance"] for s in valid]
        acc = [s["human_scores"]["accuracy"] for s in valid]
        use = [s["human_scores"]["usefulness"] for s in valid]
        return {
            "name": name,
            "count": len(valid),
            "failed": len(samples) - len(valid),
            "relevance_mean": round(sum(rel) / len(rel), 2),
            "accuracy_mean": round(sum(acc) / len(acc), 2),
            "usefulness_mean": round(sum(use) / len(use), 2),
            "overall_mean": round((sum(rel) + sum(acc) + sum(use)) / (3 * len(rel)), 2),
            "score_distribution": {
                "relevance": {i: rel.count(i) for i in range(1, 6)},
                "accuracy": {i: acc.count(i) for i in range(1, 6)},
                "usefulness": {i: use.count(i) for i in range(1, 6)},
            },
        }

    t2i_stats = compute_stats(t2i, "text_to_image")
    i2t_stats = compute_stats(i2t, "image_to_text")

    # Update data
    data["llm_judge_evaluation"] = {
        "method": "LLM-as-Judge (automated scoring via LLM)",
        "model_used": llm.model,
        "scoring_prompt": JUDGE_PROMPT[:300] + "...",
        "text_to_image_stats": t2i_stats,
        "image_to_text_stats": i2t_stats,
        "note": "使用LLM自动评分作为人工评分的替代方案。这是RAG研究中的标准做法(MT-Bench, LLM-as-Judge)。建议人工抽查10-15条验证评分合理性。"
    }

    # Save
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print("E5 Scoring Complete!")
    print(f"  Text-to-Image: {t2i_stats}")
    print(f"  Image-to-Text: {i2t_stats}")
    print(f"  Saved to {results_path}")


if __name__ == "__main__":
    main()
