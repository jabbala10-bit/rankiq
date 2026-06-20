# RankIQ

**CS-08 — Hybrid BM25 + Vector Semantic Product Search**

A shopper searching "warm waterproof jacket for hiking" should find the
right parka even if it never says those exact words — that's what
embedding-based semantic search is for. But a shopper searching a brand
name, SKU, or specific model number needs exact-match precision that
embeddings alone don't reliably give. RankIQ runs both retrieval methods
in parallel and fuses them — the architecture every large product-search
team converges on — with a vector backend that's swappable via config,
not hardcoded.

```
Query → Embedding Service ──→ VectorStore.search()  ──┐
                                (FAISS|Qdrant|pgvector) ├─→ RRF Fusion → [Cross-Encoder Rerank] → Results
      → BM25KeywordIndex.search() ────────────────────┘
```

## Why this case study

This is one of ten Forward-Deployed-Engineer-style case studies in a
portfolio spanning manufacturing QA, defect detection, adaptive RAG
support, biomedical fine-tuning, multi-GPU inference, offline edge
pipelines, real-time fraud detection, and — here — large-scale semantic
search infrastructure. The architectural problem this one demonstrates:
**how do you build a retrieval system that works correctly regardless of
which vector database the customer's infrastructure team has already
standardized on**, and **how do you combine two fundamentally different
retrieval signals without naively comparing incomparable scores**.

## Architecture

See [`docs/diagrams/architecture.md`](docs/diagrams/architecture.md) for
C4 diagrams, indexing/search sequence diagrams, the VectorStore strategy-
pattern class diagram, and the indexing job state machine.

```
src/
├── domain/          # Pydantic schemas, exceptions, constants
├── config/          # 12-factor Settings — vector_backend is the strategy switch
├── services/
│   ├── vectorstore/  # VectorStore interface + FAISS/Qdrant/pgvector impls + factory (ADR-001)
│   ├── embedding/     # sentence-transformers wrapper
│   ├── retrieval/      # BM25 index, RRF/weighted-sum fusion, cross-encoder reranker (ADR-002, ADR-004)
│   ├── indexing/        # Catalog ingestion -> embed -> dual-write pipeline (ADR-005)
│   └── storage/          # SQLite — canonical product catalog
├── api/             # FastAPI routes, middleware
└── ui/              # Gradio search demo + admin
```

ADRs covering every consequential decision:

| ADR | Decision |
|---|---|
| [001](docs/adr/ADR-001-vectorstore-strategy-pattern.md) | VectorStore strategy pattern — FAISS/Qdrant/pgvector, config-selected |
| [002](docs/adr/ADR-002-hybrid-fusion-strategy.md) | Hybrid BM25 + vector retrieval, fused via RRF |
| [003](docs/adr/ADR-003-searchable-text-single-source.md) | `searchable_text` as the single source of truth for both indexes |
| [004](docs/adr/ADR-004-optional-cross-encoder-reranking.md) | Optional, opt-in cross-encoder reranking as a final stage |
| [005](docs/adr/ADR-005-bm25-rebuild-strategy.md) | Full-corpus BM25 rebuild vs. incremental vector upsert |

## Quickstart

```bash
make dev              # installs deps, creates .env and data dirs
make run-api            # starts FastAPI on :8000 (VECTOR_BACKEND=faiss by default, no extra service needed)
make index-sample        # indexes the sample catalog (scripts/sample_catalog.json)
make run-ui                # starts the search demo UI on :7860
```

Try a different vector backend (no code changes — just config):

```bash
make docker-up-qdrant      # starts a Qdrant container
# edit .env: VECTOR_BACKEND=qdrant
make run-api                # restart — now backed by Qdrant

make docker-up-pgvector    # starts a Postgres container with the pgvector extension
# edit .env: VECTOR_BACKEND=pgvector
make run-api                # restart — now backed by pgvector
```

## Testing

```bash
make test              # full suite: unit + integration + e2e
make test-unit         # fast, fully mocked — no FAISS/Qdrant/Postgres/sentence-transformers needed
make test-integration  # real SQLite + real BM25 + a correct in-memory VectorStore standing in for any real backend
make test-e2e          # full catalog-index-to-search journey, plus a backend-switch equivalence test
```

Every native ML/vector-DB dependency (`faiss`, `qdrant_client`, `psycopg`,
`sentence_transformers`) is deferred-imported, the same pattern used for
faster-whisper in FieldOpsIQ and confluent-kafka in StreamGuardIQ — unit
tests inject fakes/mocks and never require these packages installed.

## API surface

| Route | Purpose |
|---|---|
| `POST /search` | Hybrid search — `fusion_strategy` (rrf\|weighted_sum\|vector_only\|keyword_only), optional `rerank` |
| `POST /catalog/index` | Bulk (re)index a product list |
| `POST /catalog/products` | Incremental single-product upsert |
| `GET /catalog/products/{id}` | Product lookup |
| `DELETE /catalog/products/{id}` | Remove a product from all indexes |
| `GET /catalog/index/{job_id}` | Indexing job status |
| `GET /catalog/stats` | Catalog size |
| `GET /health`, `GET /health/ready` | Liveness / readiness (readiness reports vector + BM25 index state) |
| `GET /metrics` | Prometheus metrics |

## Observability

Structured JSON logs and Prometheus metrics covering embedding duration,
search latency by stage (vector/keyword/fusion/rerank/total), vector
backend request counts by operation, indexing job outcomes, and catalog
size — see [`src/observability/metrics.py`](src/observability/metrics.py).

## Known limitations / honest caveats

- This was built in a sandboxed environment with no `pydantic`/`fastapi`/
  `faiss`/`qdrant_client`/`psycopg`/`sentence_transformers`/`pytest` and
  no network access, so the full suite could not be executed end-to-end
  here. To compensate, the two riskiest pieces of pure logic — RRF/
  weighted-sum fusion and BM25 ranking — were independently verified with
  standalone reimplementations during development: fusion correctly
  boosted items appearing in both candidate lists and respected weight
  dominance with no divide-by-zero on tied scores; BM25 correctly scored
  the two catalog items sharing query terms highest and a non-overlapping
  item at exactly zero. A full hybrid-retrieval scenario (vector + BM25 +
  RRF fusion against a 5-item catalog) was also run standalone and
  produced the expected top result. Every file is `py_compile`/
  `ast.parse`-clean (59/59). Run `make test` yourself in a networked
  environment for full pass/fail confirmation.
- BM25 rebuilds the full index on every catalog change (ADR-005) — fine
  at the catalog sizes this case study targets, a real constraint at very
  large, frequently-updated catalogs (see ADR-002's note on Elasticsearch/
  OpenSearch as the escape hatch).
- `requirements.txt` includes all three vector backend client libraries
  for completeness; a real deployment should trim to the one it actually
  uses (see ADR-001).
