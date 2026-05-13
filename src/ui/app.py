"""
Gradio Web interface for cross-modal RAG system.
Supports text-to-image and image-to-text search with LLM generation.
"""

import gradio as gr

from pathlib import Path
import numpy as np
from PIL import Image


class CrossModalApp:
    """Gradio web application for the cross-modal RAG system."""

    def __init__(self, retriever=None, encoder=None, llm_generator=None):
        self.retriever = retriever
        self.encoder = encoder
        self.llm = llm_generator

    def search_text(self, query: str, k: int = 10,
                    strategy: str = "adaptive") -> tuple:
        """Text-to-image search handler."""
        if not query.strip():
            return [], [], "Please enter a query."

        results = self.retriever.search(
            query_text=query, k=k, strategy=strategy
        )

        # Build results display
        gallery = []
        captions_display = ""
        for r in results:
            # If image_path available, add to gallery
            if "image_path" in r and Path(r["image_path"]).exists():
                gallery.append((r["image_path"], f"#{r['rank']} Score: {r['score']:.3f}"))
            captions_display += f"#{r['rank']} | Score: {r['score']:.3f} | ID: {r['id']}\n"

        # Generate LLM response if available
        llm_response = ""
        if self.llm and self.llm.is_loaded:
            llm_response = self.llm.generate(query, results).replace("\n", "\n")

        # Strategy info
        info = f"Strategy: {strategy} | Results: {len(results)}"
        if results and "query_type" in results[0]:
            info += f" | Query Type: {results[0]['query_type']}"

        return gallery, captions_display, info, llm_response

    def search_image(self, image, k: int = 10) -> tuple:
        """Image-to-text search handler."""
        if image is None:
            return "Please upload an image.", "", []

        results = self.retriever.search(query_image=image, k=k, strategy="pure_dense")

        captions = ""
        for r in results:
            captions += f"#{r['rank']} | Score: {r['score']:.3f}\n"

        llm_response = ""
        if self.llm and self.llm.is_loaded:
            llm_response = self.llm.generate("image query", results,
                                             prompt_type="image_query")

        gallery = []
        if self.retriever and hasattr(self.retriever, '_image_paths'):
            for r in results:
                if r["id"] < len(self.retriever._image_paths):
                    path = self.retriever._image_paths[r["id"]]
                    if Path(path).exists():
                        gallery.append((path, f"Score: {r['score']:.3f}"))

        return captions, llm_response, gallery


def create_app(retriever=None, encoder=None, llm_generator=None) -> gr.Blocks:
    """Create and configure the Gradio application."""
    app = CrossModalApp(
        retriever=retriever,
        encoder=encoder,
        llm_generator=llm_generator,
    )

    with gr.Blocks(title="Cross-Modal Hybrid RAG System",
                   theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # Cross-Modal Hybrid RAG System
        Multi-modal retrieval with adaptive hybrid search and LLM generation.
        """)

        with gr.Tabs():
            # Text-to-Image Tab
            with gr.TabItem("Text Search"):
                with gr.Row():
                    with gr.Column(scale=1):
                        query_input = gr.Textbox(
                            label="Query",
                            placeholder="Describe what you're looking for...",
                            lines=3,
                        )
                        strategy_dd = gr.Dropdown(
                            label="Retrieval Strategy",
                            choices=["adaptive", "pure_dense", "pure_sparse",
                                     "fixed_weight", "multi_stage"],
                            value="adaptive",
                        )
                        k_slider = gr.Slider(label="Results (K)", minimum=1,
                                             maximum=50, value=10, step=1)
                        search_btn = gr.Button("Search", variant="primary")

                    with gr.Column(scale=2):
                        gallery = gr.Gallery(label="Retrieved Images",
                                             columns=5, height=400)
                        info_text = gr.Textbox(label="Search Info", lines=1)

                with gr.Row():
                    results_text = gr.Textbox(label="Retrieval Results", lines=8)
                    llm_output = gr.Textbox(label="LLM Response", lines=8)

                search_btn.click(
                    fn=app.search_text,
                    inputs=[query_input, k_slider, strategy_dd],
                    outputs=[gallery, results_text, info_text, llm_output],
                )

            # Image-to-Text Tab
            with gr.TabItem("Image Search"):
                with gr.Row():
                    with gr.Column(scale=1):
                        img_input = gr.Image(label="Upload Image", type="pil")
                        img_k = gr.Slider(label="Results (K)", minimum=1,
                                          maximum=50, value=10, step=1)
                        img_btn = gr.Button("Search", variant="primary")

                    with gr.Column(scale=2):
                        img_gallery = gr.Gallery(label="Similar Images",
                                                 columns=5, height=400)

                with gr.Row():
                    img_results = gr.Textbox(label="Retrieved Captions", lines=8)
                    img_llm = gr.Textbox(label="LLM Description", lines=8)

                img_btn.click(
                    fn=app.search_image,
                    inputs=[img_input, img_k],
                    outputs=[img_results, img_llm, img_gallery],
                )

    return demo


if __name__ == "__main__":
    demo = create_app()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
