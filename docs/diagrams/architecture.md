# Architecture Diagrams

## C4 Level 1 — System Context

```mermaid
C4Context
    title RankIQ — System Context

    Person(shopper, "Shopper", "Searches the product catalog with natural-language queries")
    Person(admin, "Catalog Admin", "Indexes and updates the product catalog")
    System(rankiq, "RankIQ", "Hybrid BM25 + vector semantic product search")
    System_Ext(vectordb, "Vector Backend", "FAISS (in-process) | Qdrant | pgvector — config-selected")

    Rel(shopper, rankiq, "Searches products", "Gradio UI / API")
    Rel(admin, rankiq, "Indexes catalog", "Gradio UI / API")
    Rel(rankiq, vectordb, "Upserts/queries embeddings", "Backend-specific protocol")
```

## C4 Level 2 — Containers

```mermaid
C4Container
    title RankIQ — Containers

    Person(shopper, "Shopper")
    Person(admin, "Catalog Admin")

    Container_Boundary(rankiq, "RankIQ") {
        Container(ui, "Gradio UI", "Python/Gradio", "Search demo + catalog admin")
        Container(api, "FastAPI Service", "Python/FastAPI", "Search, catalog, indexing endpoints")
        Container(embed, "Embedding Service", "sentence-transformers", "Query + product text -> dense vectors")
        Container(bm25, "BM25 Index", "rank_bm25, in-process", "Keyword candidate retrieval")
        Container(rerank, "Cross-Encoder Reranker", "sentence-transformers", "Optional final relevance pass")
        ContainerDb(sqlite, "SQLite (WAL)", "File-based DB", "Canonical product catalog, indexing jobs")
    }

    System_Ext(vectorstore, "VectorStore", "FAISS | Qdrant | pgvector (strategy-selected)")

    Rel(shopper, ui, "Searches")
    Rel(admin, ui, "Indexes catalog")
    Rel(ui, api, "Calls")
    Rel(api, embed, "Generates embeddings")
    Rel(api, bm25, "Keyword search")
    Rel(api, vectorstore, "Vector search (via VectorStore interface)")
    Rel(api, rerank, "Optional rerank")
    Rel(api, sqlite, "Reads/writes")
```

## Indexing Sequence

```mermaid
sequenceDiagram
    participant Admin
    participant API as FastAPI /catalog/index
    participant Pipe as IndexingPipeline
    participant DB as SQLite
    participant Embed as EmbeddingService
    participant VS as VectorStore (FAISS/Qdrant/pgvector)
    participant BM25 as BM25KeywordIndex

    Admin->>API: POST /catalog/index [products]
    API->>Pipe: index_catalog(products)
    loop for each batch
        Pipe->>DB: save_products_batch(batch)
        Pipe->>Embed: embed_batch(batch)
        Embed-->>Pipe: ProductEmbedding[]
        Pipe->>VS: upsert(embeddings)
    end
    Pipe->>DB: list_all_products()
    Pipe->>BM25: build(all_products)
    Pipe-->>API: IndexingJob (completed)
    API-->>Admin: IndexingJob
```

## Hybrid Search Sequence

```mermaid
sequenceDiagram
    participant Shopper
    participant API as FastAPI /search
    participant Retr as RetrievalService
    participant Embed as EmbeddingService
    participant VS as VectorStore
    participant BM25 as BM25KeywordIndex
    participant Fusion as fuse_rrf()
    participant DB as SQLite
    participant Rerank as CrossEncoderReranker

    Shopper->>API: POST /search {query_text, fusion_strategy, rerank}
    API->>Retr: search(query)
    par vector retrieval
        Retr->>Embed: embed_text(query_text)
        Embed-->>Retr: query_vector
        Retr->>VS: search(query_vector, top_k=50)
        VS-->>Retr: vector_hits
    and keyword retrieval
        Retr->>BM25: search(query_text, top_k=50)
        BM25-->>Retr: keyword_hits
    end
    Retr->>Fusion: fuse_rrf(vector_ids, keyword_ids)
    Fusion-->>Retr: fused (product_id, score)[]
    Retr->>DB: get_products_by_ids(candidate_ids)
    DB-->>Retr: Product[]
    opt rerank=true
        Retr->>Rerank: rerank(query_text, top_candidates)
        Rerank-->>Retr: reordered candidates
    end
    Retr-->>API: SearchResult
    API-->>Shopper: ranked results
```

## VectorStore Strategy Pattern

```mermaid
classDiagram
    class VectorStore {
        <<abstract>>
        +upsert(embeddings)
        +search(query_vector, top_k)
        +delete(product_ids)
        +count()
        +clear()
    }
    class FAISSVectorStore {
        +upsert(embeddings)
        +search(query_vector, top_k)
        +save(directory)
        +load(directory)
    }
    class QdrantVectorStore {
        +upsert(embeddings)
        +search(query_vector, top_k)
    }
    class PgVectorStore {
        +upsert(embeddings)
        +search(query_vector, top_k)
    }
    class IndexingPipeline
    class RetrievalService

    VectorStore <|-- FAISSVectorStore
    VectorStore <|-- QdrantVectorStore
    VectorStore <|-- PgVectorStore
    IndexingPipeline --> VectorStore : depends on interface
    RetrievalService --> VectorStore : depends on interface
    create_vector_store --> VectorStore : returns selected impl
```

## Indexing Job State Machine

```mermaid
stateDiagram-v2
    [*] --> PENDING
    PENDING --> RUNNING
    RUNNING --> COMPLETED
    RUNNING --> FAILED
    COMPLETED --> [*]
    FAILED --> [*]
```
