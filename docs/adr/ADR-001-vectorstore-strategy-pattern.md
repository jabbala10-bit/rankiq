# ADR-001: VectorStore Strategy Pattern — FAISS / Qdrant / pgvector, Config-Selected

## Status
Accepted

## Context
Three credible vector backends exist for product search, each with a
different operational profile:

| Backend | Model | Pros | Cons |
|---|---|---|---|
| **FAISS** | In-process library | No server to run; fastest for small-to-medium catalogs; simplest local dev | No native persistence/filtering; single-process only; manual ID management |
| **Qdrant** | Purpose-built vector DB server | Native filtering, payload storage, horizontal scaling, REST/gRPC API | Another service to run/operate/monitor |
| **pgvector** | Postgres extension | Zero new infrastructure if Postgres is already in the stack; familiar ops (backups, replication via existing Postgres tooling) | HNSW index performance trails purpose-built vector DBs at very large scale; vector ops via raw SQL, no high-level client |

No single choice is right for every deployment — a startup with a 10K-SKU
catalog and Postgres already running has a very different optimal answer
than an enterprise with a 50M-SKU catalog and dedicated infra budget.

## Decision
Define one abstract `VectorStore` interface (`upsert`, `search`, `delete`,
`count`, `clear`) and implement **all three backends as real, complete
implementations** of it (`FAISSVectorStore`, `QdrantVectorStore`,
`PgVectorStore`). A factory function (`create_vector_store()`) reads
`Settings.vector_backend` and constructs the selected implementation;
every other module (`IndexingPipeline`, `RetrievalService`, the API layer)
depends only on the `VectorStore` interface, never on a concrete backend
class.

Rationale:
- **This is precisely the kind of decision a Forward Deployed Engineer
  has to make differently for every customer.** One customer's
  infrastructure team already runs Postgres and doesn't want to operate
  a new vector DB; another is at a scale where FAISS's single-process
  model is a non-starter; another has standardized on Qdrant
  company-wide. Building a strategy-pattern abstraction rather than
  hardcoding one backend is what makes the same RankIQ codebase
  deployable across all three situations without a rewrite.
- All three implementations going through `tests/unit/test_vectorstore_factory.py`'s
  backend-selection tests, plus the e2e `TestBackendSwitchEquivalence`
  test, demonstrates the abstraction is real (not just structurally
  present but behaviorally untested) — swapping backends through the
  identical `RetrievalService` code path produces consistent results for
  the same query and catalog.
- Each implementation handles its own connection/ID-management quirks
  internally: FAISS needs a stable int64-ID derivation scheme (FAISS has
  no native string-ID support) and explicit save/load since it has no
  server-side persistence; Qdrant and pgvector both accept string IDs
  natively and persist automatically via their respective servers. These
  differences are real and stay encapsulated inside each backend's class
  — `VectorStore` callers never see them.

## Consequences
- `requirements.txt` lists all three backend client libraries
  (`faiss-cpu`, `qdrant-client`, `psycopg`) even though a given deployment
  only needs one — flagged explicitly in the README as "install only what
  you need in production." A leaner deployment could split these into
  optional extras (`pip install rankiq[faiss]`) as a follow-up; not done
  here to keep the single-requirements-file pattern consistent with the
  rest of this portfolio.
- Score scales differ meaningfully across backends (FAISS inner-product
  on normalized vectors ≈ cosine similarity in [-1, 1]; Qdrant configured
  for cosine distance returns a similarity score; pgvector's `<=>`
  operator returns a *distance*, which `PgVectorStore.search()` explicitly
  converts to a similarity via `1 - distance` before returning). This
  conversion is each backend's responsibility internally, so callers can
  treat "higher score = more similar" as a uniform contract regardless of
  backend — but it's a detail worth knowing if extending any of these
  implementations.
- Choosing FAISS as the *default* (`VECTOR_BACKEND=faiss` in
  `.env.example`) reflects that it's the lowest-friction option for local
  development and this case study's demo — not a claim that it's the
  best production choice in general. ADR-006 in StreamGuardIQ made a
  similar "good default for the reference implementation, not necessarily
  the production answer" distinction for SQLite, and the same spirit
  applies here.
