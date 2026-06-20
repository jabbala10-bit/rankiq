"""
Centralized configuration for RankIQ using Pydantic Settings.

`vector_backend` is the key strategy-selector field — it determines which
VectorStore implementation `get_vector_store()` (api/dependencies.py)
constructs at runtime, with zero code changes elsewhere in the pipeline
(see ADR-001).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.domain.constants import (
    DEFAULT_CATALOG_DIR,
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_FAISS_INDEX_DIR,
    DEFAULT_KEYWORD_WEIGHT,
    DEFAULT_RATE_LIMIT_PER_MINUTE,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_SQLITE_PATH,
    DEFAULT_VECTOR_WEIGHT,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_name: str = "RankIQ"
    environment: str = Field(default="development")
    debug: bool = False
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Embedding
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS

    # Vector backend selection — the strategy-pattern switch (ADR-001)
    vector_backend: str = "faiss"  # faiss|qdrant|pgvector
    faiss_index_dir: str = DEFAULT_FAISS_INDEX_DIR
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "rankiq_products"
    pgvector_dsn: str = "postgresql://rankiq:rankiq@localhost:5432/rankiq"
    pgvector_table: str = "product_embeddings"

    # Fusion
    vector_weight: float = DEFAULT_VECTOR_WEIGHT
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT

    # Reranking
    reranker_model: str = DEFAULT_RERANKER_MODEL
    reranker_enabled_by_default: bool = False

    # Storage
    sqlite_path: str = DEFAULT_SQLITE_PATH
    catalog_dir: str = DEFAULT_CATALOG_DIR

    # Security
    api_auth_token: str = Field(default="", repr=False)
    rate_limit_per_minute: int = DEFAULT_RATE_LIMIT_PER_MINUTE
    cors_allowed_origins: list[str] = Field(default_factory=lambda: ["http://localhost:7860"])

    # Observability
    log_level: str = "INFO"
    log_format: str = "json"
    metrics_enabled: bool = True

    @field_validator("environment")
    @classmethod
    def _validate_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"environment must be one of {allowed}, got '{v}'")
        return v

    @field_validator("vector_backend")
    @classmethod
    def _validate_vector_backend(cls, v: str) -> str:
        allowed = {"faiss", "qdrant", "pgvector"}
        if v not in allowed:
            raise ValueError(f"vector_backend must be one of {allowed}, got '{v}'")
        return v

    def validate_production_secrets(self) -> None:
        from src.domain.exceptions import ConfigurationError

        if self.environment != "production":
            return
        if not self.api_auth_token:
            raise ConfigurationError("Missing required production secret: API_AUTH_TOKEN")

    def ensure_directories(self) -> None:
        for path_str in (self.faiss_index_dir, self.catalog_dir):
            Path(path_str).mkdir(parents=True, exist_ok=True)
        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
