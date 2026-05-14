"""
RAG prompt templates for multi-modal retrieval augmented generation.
"""


class RAGPromptBuilder:
    """Build structured prompts for the LLM generation stage."""

    SYSTEM_PROMPT = (
        "你是一个跨模态检索助手。请始终用中文回答用户的问题。"
        "你的任务是根据用户查询和检索到的图文配对，生成简洁、准确、有用的回答。"
    )

    RETRIEVAL_PROMPT = """你是一个跨模态检索助手。根据用户查询和检索到的图片描述，请用中文回答。

用户查询: {query}

检索结果（按相似度排序）:
{context}

请做到：
1. 概括最相关的检索结果
2. 解释为什么排名第一的结果与查询匹配
3. 如果结果不佳，建议优化查询词
4. 回答控制在3-5句话

请用中文回答:"""

    COMPARISON_PROMPT = """比较两种检索策略的结果差异。

用户查询: {query}

策略A ({strategy_a})的结果:
{context_a}

策略B ({strategy_b})的结果:
{context_b}

哪种策略效果更好？为什么？（2-3句话，用中文回答）

请用中文回答:"""

    IMAGE_QUERY_PROMPT = """用户上传了一张图片，希望找到相似内容。

检索到的图片描述（按相似度排序）:
{context}

请做到：
1. 描述这些检索结果共同的主题或模式
2. 根据检索结果推断用户上传图片的内容
3. 回答控制在2-3句话

请用中文回答:"""

    TRANSLATE_PROMPT = """Translate the following Chinese query into concise English for image search. Only output the translation, nothing else.

Chinese query: {query}

English translation:"""

    @classmethod
    def format_translate_prompt(cls, query: str) -> str:
        return cls.TRANSLATE_PROMPT.format(query=query)

    @classmethod
    def format_retrieval_prompt(cls, query: str, context: str) -> str:
        return cls.RETRIEVAL_PROMPT.format(query=query, context=context)

    @classmethod
    def format_comparison_prompt(cls, query: str, context_a: str,
                                 context_b: str, strategy_a: str = "A",
                                 strategy_b: str = "B") -> str:
        return cls.COMPARISON_PROMPT.format(
            query=query, context_a=context_a, context_b=context_b,
            strategy_a=strategy_a, strategy_b=strategy_b
        )

    @classmethod
    def format_image_query_prompt(cls, context: str) -> str:
        return cls.IMAGE_QUERY_PROMPT.format(context=context)
