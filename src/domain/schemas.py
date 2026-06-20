"""
Domain schemas for RankIQ.

Pure Pydantic models — no FAISS/Qdrant/pgvector imports, no FastAPI. The
VectorStore strategies, BM25 index, and fusion/reranking logic all
operate on these types, never on backend-specific objects directly.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------

class VectorBackend(str, Enum):
    FAISS = "faiss"
    QDRANT = "qdrant"
    PGVECTOR = "pgvector"


class FusionStrategy(str, Enum):
    RRF = "rrf"              # Reciprocal Rank Fusion
    WEIGHTED_SUM = "weighted_sum"
    VECTOR_ONLY = "vector_only"
    KEYWORD_ONLY = "keyword_only"


class RetrievalSource(str, Enum):
    VECTOR = "vector"
    KEYWORD = "keyword"
    FUSED = "fused"
    RERANKED = "reranked"


# --------------------------------------------------------------------------
# Catalog
# --------------------------------------------------------------------------

class Product(BaseModel):
    """A single catalog item to be indexed and made searchable."""

    product_id: str = Field(default_factory=_new_id)
    sku: str
    title: str = Field(min_length=1, max_length=300)
    description: str = Field(default="", max_length=5000)
    brand: Optional[str] = None
    category: Optional[str] = None
    price: Optional[float] = Field(default=None, ge=0)
    attributes: dict[str, str] = Field(default_factory=dict)
    in_stock: bool = True
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    @property
    def searchable_text(self) -> str:
        """
        The text actually embedded and indexed for keyword search —
        concatenates the fields a shopper's query is likely to match
        against. Kept as a single property so embedding generation and
        BM25 indexing are guaranteed to operate on identical text
        (avoiding train/index skew, same principle as FieldOpsIQ's/
        StreamGuardIQ's train/serve-skew-avoidance patterns).
        """
        parts = [self.title]
        if self.brand:
            parts.append(self.brand)
        if self.category:
            parts.append(self.category)
        if self.description:
            parts.append(self.description)
        parts.extend(self.attributes.values())
        return " ".join(parts)


class ProductEmbedding(BaseModel):
    """The dense vector representation of a Product's searchable_text."""

    product_id: str
    vector: list[float]
    model_name: str
    dimensions: int
    created_at: datetime = Field(default_factory=_utcnow)

    @field_validator("vector")
    @classmethod
    def _vector_not_empty(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("Embedding vector must not be empty.")
        return v


# --------------------------------------------------------------------------
# Query / search
# --------------------------------------------------------------------------

class SearchQuery(BaseModel):
    query_text: str = Field(min_length=1, max_length=500)
    filters: dict[str, str] = Field(default_factory=dict)
    top_k: int = Field(default=10, ge=1, le=100)
    fusion_strategy: FusionStrategy = FusionStrategy.RRF
    rerank: bool = False


class ScoredProduct(BaseModel):
    """A single result entry: a Product plus how it scored and why."""

    product: Product
    score: float
    source: RetrievalSource
    vector_rank: Optional[int] = None
    keyword_rank: Optional[int] = None
    rerank_score: Optional[float] = None


class SearchResult(BaseModel):
    query: SearchQuery
    results: list[ScoredProduct] = Field(default_factory=list)
    vector_candidate_count: int = 0
    keyword_candidate_count: int = 0
    total_latency_ms: float = 0.0
    searched_at: datetime = Field(default_factory=_utcnow)


# --------------------------------------------------------------------------
# Indexing job tracking
# --------------------------------------------------------------------------

class IndexingJobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class IndexingJob(BaseModel):
    job_id: str = Field(default_factory=_new_id)
    status: IndexingJobStatus = IndexingJobStatus.PENDING
    products_total: int = 0
    products_indexed: int = 0
    backend: VectorBackend
    error: Optional[str] = None
    started_at: datetime = Field(default_factory=_utcnow)
    completed_at: Optional[datetime] = None

    @property
    def progress_pct(self) -> float:
        if self.products_total == 0:
            return 0.0
        return round(100.0 * self.products_indexed / self.products_total, 2)
