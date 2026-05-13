"""
Context Builder: formats retrieved cross-modal results into LLM-readable context.
"""

from typing import Optional


class ContextBuilder:
    """Build structured context strings from retrieval results for LLM input."""

    def __init__(self, max_context_items: int = 5,
                 include_scores: bool = True,
                 include_captions: bool = True):
        self.max_items = max_context_items
        self.include_scores = include_scores
        self.include_captions = include_captions

    def build(self, results: list[dict],
              captions: Optional[dict[int, str]] = None) -> str:
        """Build a formatted context string from retrieval results.

        Args:
            results: List of retrieval result dicts with keys [id, score, ...]
            captions: Optional dict mapping result id to caption text

        Returns:
            Formatted string for LLM prompt
        """
        lines = []
        for item in results[:self.max_items]:
            line = f"[{item['rank']}]"
            if self.include_scores:
                line += f" (Score: {item['score']:.3f})"
            if captions and item["id"] in captions:
                line += f" Caption: {captions[item['id']]}"
            if self.include_captions:
                lines.append(line)

        return "\n".join(lines) if lines else "No relevant results found."

    def build_from_text(self, results: list[dict],
                        caption_key: str = "caption") -> str:
        """Build context using caption field directly from results."""
        lines = []
        for item in results[:self.max_items]:
            line = f"[{item['rank']}]"
            if self.include_scores:
                line += f" (Score: {item['score']:.3f})"
            if caption_key in item:
                line += f" {item[caption_key]}"
            lines.append(line)
        return "\n".join(lines) if lines else "No relevant results found."
