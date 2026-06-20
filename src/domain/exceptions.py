"""Typed domain exceptions for RankIQ."""


class RankIQError(Exception):
    """Base class for all domain-level errors."""


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------

class EmbeddingError(RankIQError):
    """Raised when embedding generation fails."""


class ModelNotLoadedError(EmbeddingError):
    """Raised when the embedding model is invoked before being loaded."""


# --------------------------------------------------------------------------
# Vector store
# --------------------------------------------------------------------------

class VectorStoreError(RankIQError):
    """Base class for vector store backend errors."""


class VectorStoreUnavailableError(VectorStoreError):
    """Raised when the configured backend (Qdrant, pgvector) is unreachable."""


class IndexNotFoundError(VectorStoreError):
    """Raised when a query/upsert targets an index that doesn't exist yet."""


class DimensionMismatchError(VectorStoreError):
    """Raised when a vector's dimensionality doesn't match the index's configured dimension."""


# --------------------------------------------------------------------------
# Keyword index
# --------------------------------------------------------------------------

class KeywordIndexError(RankIQError):
    """Raised on BM25 index build/query failures."""


# --------------------------------------------------------------------------
# Retrieval / fusion / reranking
# --------------------------------------------------------------------------

class RetrievalError(RankIQError):
    """Raised when the retrieval pipeline fails."""


class FusionError(RetrievalError):
    """Raised when result fusion fails (e.g. mismatched candidate sets)."""


class RerankingError(RetrievalError):
    """Raised when cross-encoder reranking fails."""


# --------------------------------------------------------------------------
# Indexing pipeline
# --------------------------------------------------------------------------

class IndexingError(RankIQError):
    """Raised when the catalog indexing pipeline fails."""


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

class StorageError(RankIQError):
    """Raised on catalog/job persistence failures."""


# --------------------------------------------------------------------------
# Config / auth
# --------------------------------------------------------------------------

class ConfigurationError(RankIQError):
    """Raised when required configuration/secrets are missing at startup."""


class AuthenticationError(RankIQError):
    """Raised when an API request fails authentication."""


class RateLimitExceededError(RankIQError):
    """Raised when a client exceeds the configured rate limit."""
