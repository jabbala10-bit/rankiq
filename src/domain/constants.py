"""
Domain-level constants for RankIQ.

Centralizing thresholds here keeps the fusion (ADR-002) and reranking
(ADR-004) design auditable in one place.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------

DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # sentence-transformers, 384 dims
DEFAULT_EMBEDDING_DIMENSIONS: int = 384
EMBEDDING_BATCH_SIZE: int = 64

# --------------------------------------------------------------------------
# Retrieval candidate pool sizes
# --------------------------------------------------------------------------

DEFAULT_VECTOR_CANDIDATES: int = 50  # how many ANN candidates to fetch before fusion
DEFAULT_KEYWORD_CANDIDATES: int = 50  # how many BM25 candidates to fetch before fusion
DEFAULT_RERANK_CANDIDATES: int = 20  # how many fused results get passed to the cross-encoder

# --------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# --------------------------------------------------------------------------

RRF_K: int = 60  # standard RRF damping constant (Cormack et al.)

# --------------------------------------------------------------------------
# Weighted-sum fusion (alternative to RRF)
# --------------------------------------------------------------------------

DEFAULT_VECTOR_WEIGHT: float = 0.6
DEFAULT_KEYWORD_WEIGHT: float = 0.4

# --------------------------------------------------------------------------
# Cross-encoder reranking
# --------------------------------------------------------------------------

DEFAULT_RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --------------------------------------------------------------------------
# BM25 keyword index
# --------------------------------------------------------------------------

BM25_K1: float = 1.5
BM25_B: float = 0.75

# --------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------

INDEXING_BATCH_SIZE: int = 100

# --------------------------------------------------------------------------
# API / rate limiting
# --------------------------------------------------------------------------

DEFAULT_RATE_LIMIT_PER_MINUTE: int = 300  # search traffic is typically higher-volume than write traffic

# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

DEFAULT_SQLITE_PATH: str = "data/rankiq.db"
DEFAULT_FAISS_INDEX_DIR: str = "data/indexes/faiss"
DEFAULT_CATALOG_DIR: str = "data/catalog"
