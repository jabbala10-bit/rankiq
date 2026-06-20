# ADR-003: `searchable_text` as the Single Source of Truth for Indexing

## Status
Accepted

## Context
A product has many fields (title, description, brand, category,
attributes). Both the embedding model and the BM25 index need *some*
text representation of a product, and that representation needs to be
identical at index time and at any point it's recomputed — if embedding
generation and BM25 tokenization used separately-assembled text (e.g. one
includes attributes, the other doesn't), the two retrieval methods would
silently be searching different representations of "the same" product,
making relevance debugging far harder and introducing exactly the kind
of train/serve-skew bug this portfolio's other case studies have
explicitly guarded against (FieldOpsIQ's domain-constant centralization,
StreamGuardIQ's shared `_extract_features()` function for the Isolation
Forest).

## Decision
`Product.searchable_text` is a single computed property — title, brand,
category, description, and attribute values concatenated in a fixed
order — and it is the **only** text representation ever passed to either
`EmbeddingService.embed_product()` or `BM25KeywordIndex.build()`/`upsert()`.
Neither the embedding service nor the BM25 index ever independently
reconstructs product text from raw fields.

Rationale:
- A single property, defined once on the domain model itself (not in a
  service, not duplicated in two places), is the only way to guarantee
  by construction that both retrieval methods see the same text — there's
  no code path where they could drift apart, because there's only one
  function that produces this text at all.
- Putting it on `Product` (the domain model) rather than in, say,
  `EmbeddingService` keeps this decision colocated with the data it
  describes, and makes it visible to anyone reading the domain layer
  without needing to trace through service code to find "what text
  actually gets searched."
- The field concatenation order (title first, then brand/category, then
  description, then attributes) is a deliberate relevance choice, not
  arbitrary — title terms appearing earliest gives them implicitly higher
  weight in BM25's term-frequency calculation for a given document length,
  and most embedding models are sensitive to term position/proximity in
  similar ways.

## Consequences
- Changing what counts as "searchable" for a product (e.g. deciding SKU
  codes should also be indexed) is a one-line change to
  `Product.searchable_text`, automatically propagating to both the vector
  and keyword indexes on the next reindex — no risk of updating one
  indexing path and forgetting the other.
- This does mean a single, fixed text representation serves both very
  different retrieval methods — a hypothetical future optimization
  (e.g. weighting title matches higher in BM25 specifically, independent
  of vector embedding text) would require diverging from this single-
  property design, a tradeoff explicitly accepted here in favor of
  simplicity and skew-prevention over per-method tuning flexibility.
- Query-time embedding (`RetrievalService._retrieve_candidates()` calling
  `embed_text("query", query.query_text)`) deliberately does *not* go
  through `searchable_text` (queries aren't `Product` objects) — but it
  uses the exact same `EmbeddingService.embed_text()` code path that
  `embed_product()` delegates to internally, so the query and the indexed
  product vectors are still guaranteed to live in the same embedding
  space, which is the property that actually matters for retrieval
  correctness.
