# ADR-005: Full-Catalog BM25 Rebuild vs. Incremental Vector Upsert

## Status
Accepted

## Context
The two indexes underlying hybrid search have fundamentally different
update models. A `VectorStore` supports true incremental upsert — adding
or updating one product's vector doesn't require touching any other
product's vector. BM25, by contrast, computes corpus-wide statistics
(inverse document frequency depends on how many documents in the *entire*
corpus contain a given term) — there is no correct way to incrementally
patch a BM25 index for one new document without invalidating IDF for
every term that document introduces or removes.

## Decision
`IndexingPipeline.index_catalog()` (bulk reindex) rebuilds the BM25 index
once, after all batches have been embedded and upserted into the vector
store. `IndexingPipeline.upsert_product()`/`delete_product()` (single-
product incremental operations) call `BM25KeywordIndex.upsert()`/`delete()`,
which themselves perform a full rebuild internally — the incremental-
looking API exists for caller convenience, but the underlying cost is
always "rebuild from the current full corpus."

Rationale:
- This is the only *correct* option, not merely the simplest one — a
  BM25 index that doesn't rebuild IDF statistics on corpus change isn't
  approximately right, it's actively wrong in a way that gets worse as
  the corpus drifts further from what the stale statistics assumed.
- Making the rebuild cost explicit in the method names' behavior (rather
  than hiding it behind a misleadingly "incremental"-sounding API) was
  considered, but ultimately the simpler `upsert_product()`/
  `delete_product()` naming was kept because the *caller-facing contract*
  (add/update/remove one product, index reflects it) is accurately
  incremental even if the BM25 internals aren't — the cost characteristic
  is documented in the BM25 index's own docstrings and in this ADR, which
  is where an engineer investigating performance would look.
- For the bulk `index_catalog()` path specifically, deferring the BM25
  rebuild to *after* all batches are embedded/upserted (rather than
  rebuilding once per batch) avoids paying the full-corpus rebuild cost
  `INDEXING_BATCH_SIZE`-many times during a large catalog load — it's
  paid exactly once per `index_catalog()` call regardless of catalog
  size.

## Consequences
- Single-product `upsert_product()`/`delete_product()` calls have an
  effective cost of O(catalog size) due to the BM25 rebuild, not O(1) —
  a deployment doing frequent single-product updates against a large
  catalog will feel this. ADR-002's note on Elasticsearch/OpenSearch as
  an alternative keyword backend is the relevant escape hatch if this
  becomes a real bottleneck — a proper inverted-index search engine
  supports genuinely incremental document updates.
- The vector store side of the same operations *is* genuinely O(1)
  (or O(log n) depending on the backend's index structure) — so at scale,
  BM25 rebuild cost will dominate single-product update latency long
  before vector store upsert cost does. This asymmetry is worth knowing
  when capacity planning for a catalog with frequent price/stock updates
  (which call `upsert_product()` on every change).
- This ADR's reasoning generalizes a pattern already established
  elsewhere in this portfolio: StreamGuardIQ's ADR-002 made a similar
  "O(1) real-time path, accept a different cost model for the
  population-wide statistic" tradeoff for its EWMA scorer vs. the batch
  ML re-scorer — the specific mechanism differs, but the underlying
  principle (some statistics are inherently corpus-wide, not
  per-document) recurs.
