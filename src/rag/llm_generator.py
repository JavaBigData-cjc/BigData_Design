"""
LLM Generator for RAG (Retrieval-Augmented Generation).
Uses Qwen2.5-7B-Instruct (GPTQ 4-bit) for answer generation.
"""

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompt_templates import RAGPromptBuilder
from .context_builder import ContextBuilder


class LLMGenerator:
    """LLM-powered answer generator for RAG pipeline."""

    DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"

    def __init__(self, model_name: str = None,
                 device_map: str = "auto",
                 torch_dtype: str = "float16",
                 max_new_tokens: int = 256,
                 temperature: float = 0.7):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.tokenizer = None
        self.model = None
        self.device_map = device_map
        self.torch_dtype = getattr(torch, torch_dtype)

        self.prompt_builder = RAGPromptBuilder()
        self.context_builder = ContextBuilder()

    def load(self):
        """Load the LLM model and tokenizer into memory."""
        print(f"[load] Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"[load] Model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    def unload(self):
        """Free GPU memory."""
        self.model = None
        self.tokenizer = None
        torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def generate(self, query: str, retrieval_results: list[dict],
                 captions: Optional[dict] = None,
                 prompt_type: str = "retrieval") -> str:
        """Generate an answer using RAG.

        Args:
            query: User's query text
            retrieval_results: List of result dicts from HybridRetriever
            captions: Optional dict[id] -> caption text
            prompt_type: "retrieval" | "comparison" | "image_query"

        Returns:
            Generated answer string
        """
        if not self.is_loaded:
            self.load()

        # Build context from retrieved results
        context = self.context_builder.build(retrieval_results, captions)

        # Build prompt
        if prompt_type == "retrieval":
            prompt = self.prompt_builder.format_retrieval_prompt(query, context)
        elif prompt_type == "image_query":
            prompt = self.prompt_builder.format_image_query_prompt(context)
        else:
            prompt = self.prompt_builder.format_retrieval_prompt(query, context)

        # Apply chat template
        messages = [
            {"role": "system", "content": self.prompt_builder.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )

        return response.strip()
