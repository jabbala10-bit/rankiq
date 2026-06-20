"""
Gradio UI for RankIQ: a shopper-facing search demo (try different fusion
strategies side by side) and an admin tab for bulk catalog indexing.
"""
from __future__ import annotations

import json
import os

import gradio as gr
import httpx

API_BASE_URL = os.environ.get("RANKIQ_API_URL", "http://localhost:8000")


def run_search(query_text: str, fusion_strategy: str, rerank: bool, top_k: int) -> str:
    if not query_text:
        return "⚠️ Enter a search query."

    strategy_map = {
        "Hybrid (RRF)": "rrf",
        "Hybrid (Weighted Sum)": "weighted_sum",
        "Vector Only": "vector_only",
        "Keyword Only (BM25)": "keyword_only",
    }
    try:
        resp = httpx.post(
            f"{API_BASE_URL}/search",
            json={
                "query_text": query_text,
                "fusion_strategy": strategy_map[fusion_strategy],
                "rerank": rerank,
                "top_k": int(top_k),
            },
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()
    except httpx.RequestError as exc:
        return f"❌ Could not reach API: {exc}"
    except httpx.HTTPStatusError as exc:
        return f"❌ API error: {exc.response.text[:300]}"

    if not result["results"]:
        return "No results found."

    lines = [
        f"**{len(result['results'])} results** "
        f"(vector candidates: {result['vector_candidate_count']}, "
        f"keyword candidates: {result['keyword_candidate_count']}, "
        f"latency: {result['total_latency_ms']:.1f}ms)\n"
    ]
    for i, hit in enumerate(result["results"], 1):
        p = hit["product"]
        rank_info = []
        if hit.get("vector_rank"):
            rank_info.append(f"vec#{hit['vector_rank']}")
        if hit.get("keyword_rank"):
            rank_info.append(f"kw#{hit['keyword_rank']}")
        if hit.get("rerank_score") is not None:
            rank_info.append(f"rerank={hit['rerank_score']:.3f}")
        rank_str = f" ({', '.join(rank_info)})" if rank_info else ""

        lines.append(
            f"**{i}. {p['title']}** — score {hit['score']:.4f}{rank_str}\n"
            f"   {p.get('brand', '—')} | {p.get('category', '—')} | "
            f"${p.get('price', '—')} | {'In stock' if p['in_stock'] else 'Out of stock'}"
        )
    return "\n\n".join(lines)


def submit_catalog_json(catalog_json: str) -> str:
    if not catalog_json.strip():
        return "⚠️ Paste a JSON array of products."
    try:
        products = json.loads(catalog_json)
    except json.JSONDecodeError as exc:
        return f"❌ Invalid JSON: {exc}"

    try:
        resp = httpx.post(f"{API_BASE_URL}/catalog/index", json=products, timeout=120)
        resp.raise_for_status()
        job = resp.json()
    except httpx.RequestError as exc:
        return f"❌ Could not reach API: {exc}"
    except httpx.HTTPStatusError as exc:
        return f"❌ API error: {exc.response.text[:300]}"

    return (
        f"✅ Indexing job `{job['job_id']}` — status: **{job['status']}**\n\n"
        f"Indexed {job['products_indexed']}/{job['products_total']} products "
        f"using backend `{job['backend']}`."
    )


def check_catalog_stats() -> str:
    try:
        resp = httpx.get(f"{API_BASE_URL}/catalog/stats", timeout=10)
        resp.raise_for_status()
        stats = resp.json()
    except httpx.RequestError as exc:
        return f"❌ Could not reach API: {exc}"
    return f"📦 Catalog size: **{stats['product_count']}** products"


_SAMPLE_CATALOG = json.dumps(
    [
        {
            "sku": "JKT-001",
            "title": "Men's Waterproof Hiking Jacket",
            "description": "Lightweight, breathable, fully waterproof shell jacket for hiking and backpacking.",
            "brand": "TrailPeak",
            "category": "Outerwear",
            "price": 129.99,
            "attributes": {"color": "forest green", "size_range": "S-XXL"},
        },
        {
            "sku": "JKT-002",
            "title": "Women's Insulated Winter Parka",
            "description": "Warm down-insulated parka rated for sub-zero temperatures.",
            "brand": "Northgale",
            "category": "Outerwear",
            "price": 199.99,
            "attributes": {"color": "black", "size_range": "XS-XL"},
        },
        {
            "sku": "SHO-010",
            "title": "Trail Running Shoes",
            "description": "Grippy outsole, breathable mesh upper, ideal for muddy or wet trails.",
            "brand": "TrailPeak",
            "category": "Footwear",
            "price": 89.99,
            "attributes": {"color": "grey/orange"},
        },
    ],
    indent=2,
)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="RankIQ", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔎 RankIQ — Hybrid Product Search")
        gr.Markdown("BM25 keyword search + vector semantic search, fused via RRF, with optional cross-encoder reranking.")

        with gr.Tab("🛍️ Search"):
            query_input = gr.Textbox(label="Search Query", placeholder="e.g. warm waterproof jacket for hiking")
            with gr.Row():
                fusion_choice = gr.Dropdown(
                    choices=["Hybrid (RRF)", "Hybrid (Weighted Sum)", "Vector Only", "Keyword Only (BM25)"],
                    value="Hybrid (RRF)",
                    label="Fusion Strategy",
                )
                top_k_input = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Top K")
                rerank_checkbox = gr.Checkbox(label="Apply cross-encoder rerank", value=False)
            search_btn = gr.Button("Search", variant="primary")
            results_output = gr.Markdown()
            search_btn.click(
                fn=run_search,
                inputs=[query_input, fusion_choice, rerank_checkbox, top_k_input],
                outputs=results_output,
            )

        with gr.Tab("⚙️ Admin: Index Catalog"):
            gr.Markdown("Paste a JSON array of products to index (or use the sample below).")
            catalog_input = gr.Code(value=_SAMPLE_CATALOG, language="json", label="Product Catalog (JSON)", lines=15)
            index_btn = gr.Button("Index Catalog", variant="primary")
            index_status = gr.Markdown()
            index_btn.click(fn=submit_catalog_json, inputs=catalog_input, outputs=index_status)

            gr.Markdown("---")
            stats_btn = gr.Button("Check Catalog Stats")
            stats_output = gr.Markdown()
            stats_btn.click(fn=check_catalog_stats, outputs=stats_output)

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860)
