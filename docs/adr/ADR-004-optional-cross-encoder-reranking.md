# ADR-004: Optional Cross-Encoder Reranking as a Final, Opt-In Stage

## Status
Accepted

## Context
Fused vector+keyword results are ranked using independently-computed
signals — the query and each product were never directly compared to
each other in a single model pass. A cross-encoder (a model that takes
the *pair* (query, product_text) as joint input and outputs a relevance
score) is generally more accurate than fusing independent scores, because
it can model interactions between the specific query and the specific
candidate, at the cost of being far too slow to run over an entire
catalog — it must be restricted to a small candidate set.

## Decision
Cross-encoder reranking (`CrossEncoderReranker`) is implemented as an
**optional final stage**, applied only to the top
`DEFAULT_RERANK_CANDIDATES` (20) fused results, and only when the caller
explicitly requests it (`SearchQuery.rerank=True`). It is never on by
default (`Settings.reranker_enabled_by_default = False`).

Rationale:
- **Reranking only makes sense after fusion has already narrowed the
  field.** Running a cross-encoder over the full candidate pool (up to
  100 items from vector+keyword retrieval) on every query would add
  meaningful latency for a relevance gain that's mostly redundant with
  what RRF/weighted-sum fusion already captures for items outside the
  top ~20 — the cross-encoder's marginal value is highest for refining
  the order *within* an already-relevant top set, not for surfacing
  items fusion missed entirely.
- **Opt-in by default** because the latency cost is real and not every
  caller wants to pay it — a typeahead/autocomplete use case wants the
  fastest possible response and can tolerate fusion-only ranking; a
  full search-results-page render can afford the extra latency for
  better top-of-page relevance. Exposing this as a per-request flag
  (rather than a global server setting only) lets a single deployment
  serve both use cases from the same service.
- The reranker reuses `Product.searchable_text` (ADR-003) for the
  product side of each (query, product_text) pair — consistent with the
  rest of the pipeline's "one text representation per product" rule,
  even though the cross-encoder's joint-attention mechanism could in
  principle benefit from a differently-formatted representation; kept
  consistent here to avoid a second place product text could drift from
  the embedding/BM25 representation.

## Consequences
- `CrossEncoderReranker` follows the same deferred-import-for-load_model
  pattern as `EmbeddingService` and every other heavy-ML-dependency
  service in this portfolio — it can be constructed and unit-tested
  without `sentence-transformers`/`torch` installed, using a fake model
  injected directly onto `self._model`.
- Because reranking only touches the top `DEFAULT_RERANK_CANDIDATES`,
  any true positive ranked below that cutoff by the fusion stage can
  never be promoted by reranking — fusion quality still bounds overall
  result quality; reranking can only improve ordering *within* what
  fusion already surfaced. This is a deliberate, accepted limitation of
  cascading retrieval architectures generally, not specific to this
  implementation.
- `RetrievalService._apply_rerank()` preserves any results beyond the
  reranked top slice unchanged (`updated + scored[DEFAULT_RERANK_CANDIDATES:]`)
  rather than dropping them, so `top_k` truncation still behaves
  correctly even when `top_k` exceeds the reranking window.
