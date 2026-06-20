"""Catalog management and indexing routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_indexing_pipeline, get_storage_service, require_api_token
from src.domain.exceptions import RankIQError
from src.domain.schemas import IndexingJob, Product
from src.services.indexing.pipeline import IndexingPipeline
from src.services.storage.sqlite_service import SQLiteStorageService

router = APIRouter(prefix="/catalog", tags=["catalog"], dependencies=[Depends(require_api_token)])


@router.post("/products", response_model=Product, status_code=201)
def upsert_product(product: Product, pipeline: IndexingPipeline = Depends(get_indexing_pipeline)) -> Product:
    """Adds or updates a single product — incremental indexing, no full BM25 rebuild wait."""
    try:
        pipeline.upsert_product(product)
    except RankIQError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return product


@router.get("/products/{product_id}", response_model=Product)
def get_product(product_id: str, storage: SQLiteStorageService = Depends(get_storage_service)) -> Product:
    product = storage.get_product(product_id)
    if product is None:
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
    return product


@router.delete("/products/{product_id}", status_code=204)
def delete_product(product_id: str, pipeline: IndexingPipeline = Depends(get_indexing_pipeline)) -> None:
    pipeline.delete_product(product_id)


@router.post("/index", response_model=IndexingJob, status_code=201)
def index_catalog(
    products: list[Product], pipeline: IndexingPipeline = Depends(get_indexing_pipeline)
) -> IndexingJob:
    """Full (re)indexing of a product list — persists, embeds, upserts the vector store, rebuilds BM25."""
    try:
        return pipeline.index_catalog(products)
    except RankIQError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/index/{job_id}", response_model=IndexingJob)
def get_indexing_job(job_id: str, storage: SQLiteStorageService = Depends(get_storage_service)) -> IndexingJob:
    job = storage.get_indexing_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Indexing job {job_id} not found")
    return job


@router.get("/stats")
def catalog_stats(storage: SQLiteStorageService = Depends(get_storage_service)) -> dict:
    return {"product_count": storage.count_products()}
