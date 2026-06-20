# ADR-002: Hybrid BM25 + Vector Retrieval, Fused via RRF

## Status
Accepted

## Context
Pure vector (embedding) search excels at *semantic* intent — "warm
waterproof jacket for hiking" correctly matches a parka even without
exact word overlap — but is weak on *exact-match* signals: brand names,
SKU codes, model numbers, and specific nouns a shopper typed deliberately
("TrailPeak", "JKT-001") can get diluted in a dense embedding's averaged
semantics. Pure keyword (BM25) search is the opposite: excellent exact-
match precision, no understanding of synonyms or intent ("warm jacket"
won't match "insulated parka" without shared vocabulary). Production
e-commerce search at Google/Amazon scale runs both and combines them —
this is not a novel insight, but implementing it well requires care in
how the two signals get combined.

## Decision
Run **both retrieval methods in parallel** for every query — a BM25
search over `BM25KeywordIndex` and an ANN search over the configured
`VectorStore` — each returning a candidate pool
(`DEFAULT_VECTOR_CANDIDATES`/`DEFAULT_KEYWORD_CANDIDATES` = 50), then
**fuse with Reciprocal Rank Fusion (RRF) by default**, with weighted-sum
fusion offered as a configurable alternative.

Rationale for RRF as the default:
- **RRF requires no score normalization between fundamentally
  incomparable scales.** BM25 scores are unbounded and corpus-dependent;
  vector similarity scores live in [-1, 1] or [0, 1] depending on metric.
  Naively averaging or summing these raw scores is close to meaningless
  without careful normalization — RRF sidesteps the entire problem by
  only ever looking at *rank position* within each list, never the raw
  score value. This was directly verified during development: a
  standalone harness fusing a fake vector ranking and a real BM25-lite
  ranking for "waterproof hiking jacket" correctly surfaced the hiking
  jacket and waterproof backpack as the top two fused results, matching
  both individual signals' independent top picks.
- **RRF is simple, parameter-light, and well-studied** (Cormack, Clarke &
  Buettcher, 2009) — the only tunable is the damping constant `k`
  (default 60, the standard value from the literature), versus weighted-
  sum fusion's two weights that need active tuning per deployment.
- Weighted-sum fusion remains available
  (`FusionStrategy.WEIGHTED_SUM`, configurable `vector_weight`/
  `keyword_weight`) for deployments that want explicit, interpretable
  control — e.g. "I want vector search to dominate for this category" —
  at the cost of needing real tuning effort and accepting that min-max
  normalization is sensitive to outlier scores in either candidate list.

## Alternative considered: a dedicated search-engine server (Elasticsearch/OpenSearch)
The keyword half of this hybrid design uses an in-process `rank_bm25`
index rather than a separate Elasticsearch or OpenSearch cluster.
Elasticsearch/OpenSearch would be the better choice when: the catalog is
large enough that in-process BM25 (which holds the full tokenized corpus
in memory and rebuilds entirely on any catalog change — see this index's
own implementation notes) becomes a memory or rebuild-latency problem;
when more sophisticated text analysis (stemming, synonym expansion,
multi-field boosting, fuzzy matching) is needed beyond plain BM25; or
when the keyword search needs its own independent scaling/sharding story
separate from the vector search infrastructure. For this case study's
scope and the catalog sizes it's meant to demonstrate, in-process BM25
keeps the stack simpler without sacrificing the core hybrid-retrieval
lesson — but a real large-catalog deployment should treat "BM25 index
choice" as its own follow-up decision, parallel to how ADR-001 treats
vector backend choice.

## Consequences
- Both candidate pools are fetched on every query (`DEFAULT_VECTOR_CANDIDATES`
  + `DEFAULT_KEYWORD_CANDIDATES` = up to 100 candidates examined per
  search), which is more compute than a single-method search — an
  accepted cost for meaningfully better relevance, and bounded by the
  candidate pool size constants rather than scanning the full catalog.
- `RetrievalService._retrieve_candidates()` only calls the methods
  actually needed for the selected `FusionStrategy` (e.g.
  `FusionStrategy.VECTOR_ONLY` skips the BM25 call entirely) — fusion
  strategy selection is also an escape hatch for callers who want a
  single-method search without paying the dual-retrieval cost.
- BM25's full-corpus-rebuild-on-change behavior (see
  `BM25KeywordIndex.build()`'s docstring) means `upsert_product()`'s
  incremental path is cheap for the vector store but always pays a full
  BM25 rebuild — acceptable for moderate catalog sizes and update
  frequencies, a real scaling constraint to revisit (likely via the
  Elasticsearch/OpenSearch alternative above) for very large, frequently-
  updated catalogs.
