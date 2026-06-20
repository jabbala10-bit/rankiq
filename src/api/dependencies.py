"""FastAPI dependency providers for RankIQ."""
from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, Header, HTTPException, status

from src.config.settings import Settings, get_settings
from src.domain.exceptions import AuthenticationError
from src.domain.schemas import VectorBackend
from src.services.embedding.embedding_service import EmbeddingService
from src.services.indexing.pipeline import IndexingPipeline
from src.services.retrieval.bm25_index import BM25KeywordIndex
from src.services.retrieval.reranker import CrossEncoderReranker
from src.services.retrieval.retrieval_service import RetrievalService
from src.services.storage.sqlite_service import SQLiteStorageService
from src.services.vectorstore.base import VectorStore
from src.services.vectorstore.factory import create_vector_store


@lru_cache
def get_storage_service() -> SQLiteStorageService:
    return SQLiteStorageService(get_settings())


@lru_cache
def get_embedding_service() -> EmbeddingService:
    service = EmbeddingService(get_settings())
    service.load_model()
    return service


@lru_cache
def get_vector_store() -> VectorStore:
    """
    The strategy-pattern injection point: constructs whichever backend
    `Settings.vector_backend` selects, via the factory in
    services/vectorstore/factory.py. No other code in the API layer
    knows or cares which concrete class this is (ADR-001).
    """
    return create_vector_store(get_settings())


@lru_cache
def get_bm25_index() -> BM25KeywordIndex:
    return BM25KeywordIndex()


@lru_cache
def get_reranker() -> CrossEncoderReranker:
    reranker = CrossEncoderReranker(get_settings())
    reranker.load_model()
    return reranker


def get_indexing_pipeline(
    settings: Settings = Depends(get_settings),
    vector_store: VectorStore = Depends(get_vector_store),
    embedding: EmbeddingService = Depends(get_embedding_service),
    bm25: BM25KeywordIndex = Depends(get_bm25_index),
    storage: SQLiteStorageService = Depends(get_storage_service),
) -> IndexingPipeline:
    return IndexingPipeline(
        vector_store=vector_store,
        embedding_service=embedding,
        bm25_index=bm25,
        storage=storage,
        vector_backend=VectorBackend(settings.vector_backend),
    )


def get_retrieval_service(
    settings: Settings = Depends(get_settings),
    vector_store: VectorStore = Depends(get_vector_store),
    bm25: BM25KeywordIndex = Depends(get_bm25_index),
    embedding: EmbeddingService = Depends(get_embedding_service),
    storage: SQLiteStorageService = Depends(get_storage_service),
) -> RetrievalService:
    return RetrievalService(
        vector_store=vector_store,
        bm25_index=bm25,
        embedding_service=embedding,
        storage=storage,
        vector_weight=settings.vector_weight,
        keyword_weight=settings.keyword_weight,
    )


def require_api_token(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    if settings.environment == "development":
        return
    if not settings.api_auth_token:
        raise AuthenticationError("Server has no API_AUTH_TOKEN configured.")
    expected = f"Bearer {settings.api_auth_token}"
    if authorization != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
