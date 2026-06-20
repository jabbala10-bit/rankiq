"""Search routes — the main read-path entrypoint for shoppers/clients."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import get_retrieval_service
from src.domain.schemas import SearchQuery, SearchResult
from src.services.retrieval.retrieval_service import RetrievalService

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResult)
def search(query: SearchQuery, retrieval: RetrievalService = Depends(get_retrieval_service)) -> SearchResult:
    """
    Executes a hybrid vector + keyword search. `fusion_strategy` controls
    how the two candidate lists are combined (rrf|weighted_sum|vector_only|
    keyword_only); `rerank=true` adds a cross-encoder pass over the top
    fused candidates before truncating to top_k.
    """
    return retrieval.search(query)
