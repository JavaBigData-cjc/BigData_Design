"""
RAG prompt templates for multi-modal retrieval augmented generation.
"""


class RAGPromptBuilder:
    """Build structured prompts for the LLM generation stage."""

    SYSTEM_PROMPT = (
        "You are a cross-modal retrieval assistant. Your task is to help users "
        "understand the relationship between their query and the retrieved image-text "
        "pairs. Be concise, accurate, and helpful."
    )

    RETRIEVAL_PROMPT = """You are a cross-modal retrieval assistant. Based on the user's query and the retrieved image-caption pairs, provide a helpful response.

User query: {query}

Retrieved results:
{context}

Please:
1. Summarize the most relevant results
2. Explain why the top result matches the query
3. If results seem poor, suggest query refinements
4. Keep the response concise (3-5 sentences)

Answer:"""

    COMPARISON_PROMPT = """You are analyzing multiple retrieval strategies for the same query. Compare the results.

User query: {query}

Results from strategy A ({strategy_a}):
{context_a}

Results from strategy B ({strategy_b}):
{context_b}

Which strategy produced better results and why? (2-3 sentences)

Answer:"""

    IMAGE_QUERY_PROMPT = """A user has uploaded an image and wants to find similar content.

Retrieved captions:
{context}

Please:
1. Describe what patterns/themes the retrieved captions share
2. Summarize what the retrieved content reveals about the uploaded image
3. Be concise (2-3 sentences)

Answer:"""

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
