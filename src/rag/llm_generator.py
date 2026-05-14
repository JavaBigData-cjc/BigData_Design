"""
LLM Generator for RAG using OpenAI-compatible API.
Supports 通义千问 (DashScope), DeepSeek, OpenAI, SiliconFlow, etc.
Configure via .env file or environment variables.
"""
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from .prompt_templates import RAGPromptBuilder
from .context_builder import ContextBuilder

# Load .env if present
load_dotenv()


class LLMGenerator:
    """API-based LLM answer generator for RAG pipeline."""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
        max_new_tokens: int = None,
        temperature: float = None,
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
        self.model = model or os.getenv("LLM_MODEL", "deepseek-chat")
        self.max_new_tokens = max_new_tokens or int(os.getenv("LLM_MAX_TOKENS", "512"))
        self.temperature = temperature or float(os.getenv("LLM_TEMPERATURE", "0.7"))

        self.client = None
        self._connected = False
        self.prompt_builder = RAGPromptBuilder()
        self.context_builder = ContextBuilder()

    @property
    def is_loaded(self) -> bool:
        return self._connected

    def connect(self):
        """Initialize API client. No model loading needed for API."""
        if not self.api_key:
            raise ValueError(
                "LLM_API_KEY not set. Copy .env.example to .env and fill in your API key."
            )
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self._connected = True

    def generate(
        self,
        query: str,
        retrieval_results: list[dict],
        prompt_type: str = "retrieval",
    ) -> str:
        """Generate RAG answer from retrieval results.

        Args:
            query: User's query text (or "image upload" for image queries)
            retrieval_results: List of result dicts from retriever/search
            prompt_type: "retrieval" | "image_query"

        Returns:
            Generated answer string
        """
        if not self._connected:
            self.connect()

        # Build context from results
        context = self.context_builder.build_from_text(
            retrieval_results, caption_key="caption"
        )

        # Build prompt
        if prompt_type == "image_query":
            user_prompt = self.prompt_builder.format_image_query_prompt(context)
        else:
            user_prompt = self.prompt_builder.format_retrieval_prompt(query, context)

        messages = [
            {"role": "system", "content": self.prompt_builder.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[LLM Error] {e}"

    @staticmethod
    def check_connection() -> dict:
        """Verify API configuration is valid. Returns status dict."""
        info = {
            "api_key_set": bool(os.getenv("LLM_API_KEY", "")),
            "base_url": os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
            "model": os.getenv("LLM_MODEL", "deepseek-chat"),
        }

        if not info["api_key_set"]:
            info["status"] = "missing_key"
            info["message"] = "LLM_API_KEY not set in .env"
            return info

        try:
            client = OpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=info["base_url"])
            # Quick connectivity test
            client.models.list(extra_headers={"X-Test": "1"})
            info["status"] = "ok"
            info["message"] = f"Connected to {info['base_url']}"
        except Exception as e:
            info["status"] = "error"
            info["message"] = str(e)

        return info
