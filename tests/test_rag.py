"""Tests for RAG module."""
import pytest


class TestContextBuilder:
    """Test context building from retrieval results."""

    def test_build_context(self):
        from src.rag.context_builder import ContextBuilder
        results = [
            {"rank": 1, "score": 0.95, "id": 42},
            {"rank": 2, "score": 0.87, "id": 17},
            {"rank": 3, "score": 0.76, "id": 99},
        ]
        captions = {42: "a dog in a park", 17: "a cat on a couch", 99: "a bird in the sky"}

        builder = ContextBuilder(max_context_items=2)
        context = builder.build(results, captions)

        assert "a dog in a park" in context
        assert "a cat on a couch" in context
        assert "0.95" in context
        # Third result should be excluded (max=2)
        assert "a bird in the sky" not in context


class TestPromptTemplates:
    """Test RAG prompt generation."""

    def test_retrieval_prompt(self):
        from src.rag.prompt_templates import RAGPromptBuilder
        prompt = RAGPromptBuilder.format_retrieval_prompt(
            query="What animals are in the park?",
            context="[1] Score: 0.95 Caption: a dog in a park"
        )
        assert "What animals are in the park?" in prompt
        assert "a dog in a park" in prompt

    def test_image_query_prompt(self):
        from src.rag.prompt_templates import RAGPromptBuilder
        prompt = RAGPromptBuilder.format_image_query_prompt(
            context="[1] Score: 0.88 Caption: sunset over mountains"
        )
        assert "uploaded an image" in prompt.lower()
        assert "sunset over mountains" in prompt
