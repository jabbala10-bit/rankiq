"""Health and readiness routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import get_bm25_index, get_vector_store
from src.config.settings import Settings, get_settings
from src.services.retrieval.bm25_index import BM25KeywordIndex
from src.services.vectorstore.base import VectorStore

router = APIRouter(tags=["health"])


@router.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict:
    return {
        "status": "ok",
        "app": settings.app_name,
        "environment": settings.environment,
        "vector_backend": settings.vector_backend,
    }


@router.get("/health/ready")
def readiness(
    vector_store: VectorStore = Depends(get_vector_store),
    bm25: BM25KeywordIndex = Depends(get_bm25_index),
) -> dict:
    """
    Readiness reflects whether the catalog has actually been indexed yet
    — an empty index isn't a failure (fresh deployment), but it's useful
    operational signal distinct from "service is up."
    """
    try:
        vector_count = vector_store.count()
        vector_ok = True
    except Exception:  # noqa: BLE001
        vector_count = 0
        vector_ok = False

    return {
        "status": "ok" if vector_ok else "degraded",
        "vector_store_reachable": vector_ok,
        "vector_count": vector_count,
        "bm25_built": bm25.is_built,
        "bm25_count": bm25.count(),
    }
