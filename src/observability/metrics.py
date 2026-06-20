"""Prometheus metrics for RankIQ."""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------

EMBEDDING_REQUESTS_TOTAL = Counter(
    "rankiq_embedding_requests_total",
    "Total embedding generation calls",
    labelnames=["status"],
)

EMBEDDING_DURATION_SECONDS = Histogram(
    "rankiq_embedding_duration_seconds",
    "Time spent generating embeddings for a batch",
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10),
)

# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

SEARCH_REQUESTS_TOTAL = Counter(
    "rankiq_search_requests_total",
    "Total search requests",
    labelnames=["fusion_strategy", "status"],
)

SEARCH_LATENCY_SECONDS = Histogram(
    "rankiq_search_latency_seconds",
    "End-to-end search request latency",
    labelnames=["stage"],  # vector|keyword|fusion|rerank|total
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2),
)

VECTOR_BACKEND_REQUESTS_TOTAL = Counter(
    "rankiq_vector_backend_requests_total",
    "Total requests to the configured vector backend",
    labelnames=["backend", "operation", "status"],  # operation=upsert|search|delete
)

RERANK_REQUESTS_TOTAL = Counter(
    "rankiq_rerank_requests_total",
    "Total cross-encoder reranking calls",
    labelnames=["status"],
)

# --------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------

INDEXING_JOBS_TOTAL = Counter(
    "rankiq_indexing_jobs_total",
    "Total catalog indexing jobs run",
    labelnames=["status"],
)

PRODUCTS_INDEXED_TOTAL = Counter(
    "rankiq_products_indexed_total",
    "Total products successfully indexed",
)

CATALOG_SIZE = Gauge(
    "rankiq_catalog_size",
    "Current number of products in the catalog",
)
