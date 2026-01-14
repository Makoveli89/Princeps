"""
Vector Search Utilities - Low-level vector search functions for RetrieverAgent.

This module provides foundational vector search capabilities:
- embed_text(): Generate embeddings for text using sentence-transformers
- query_vector_index(): Search vector indices with filters
- batch_embed(): Efficient batch embedding generation
- similarity_search(): Cosine similarity computations
- Support for multiple backends (pgvector, ChromaDB, in-memory)

The utilities integrate with the Brain Layer's existing vector stores
while providing a clean interface for the RetrieverAgent.
"""

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Configuration
# =============================================================================


class VectorBackend(Enum):
    """Supported vector search backends."""

    PGVECTOR = "pgvector"
    CHROMADB = "chromadb"
    IN_MEMORY = "in_memory"
    FAISS = "faiss"


class DistanceMetric(Enum):
    """Distance metrics for similarity computation."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


class EmbeddingModel(Enum):
    """Supported embedding models."""

    MINILM_L6 = "all-MiniLM-L6-v2"  # 384 dimensions, fast
    MPNET_BASE = "all-mpnet-base-v2"  # 768 dimensions, better quality
    BGE_SMALL = "BAAI/bge-small-en-v1.5"  # 384 dimensions, good quality
    BGE_BASE = "BAAI/bge-base-en-v1.5"  # 768 dimensions, high quality
    INSTRUCTOR = "hkunlp/instructor-large"  # 768 dimensions, instruction-tuned


@dataclass
class VectorSearchConfig:
    """Configuration for vector search operations."""

    # Embedding settings
    model_name: str = EmbeddingModel.MINILM_L6.value
    embedding_dimension: int = 384
    normalize_embeddings: bool = True

    # Search settings
    default_top_k: int = 10
    min_similarity: float = 0.0
    distance_metric: DistanceMetric = DistanceMetric.COSINE

    # Caching
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600
    max_cache_size: int = 10000

    # Backend settings
    backend: VectorBackend = VectorBackend.PGVECTOR
    fallback_backends: list[VectorBackend] = field(
        default_factory=lambda: [VectorBackend.CHROMADB, VectorBackend.IN_MEMORY]
    )

    # Performance
    batch_size: int = 32
    max_concurrent_requests: int = 5
    timeout_seconds: float = 30.0


@dataclass
class VectorSearchResult:
    """Result from a vector search query."""

    id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    source: str | None = None
    chunk_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "source": self.source,
            "chunk_index": self.chunk_index,
        }


@dataclass
class SearchFilter:
    """Filter criteria for vector search."""

    # Metadata filters
    metadata_match: dict[str, Any] = field(default_factory=dict)
    metadata_contains: dict[str, str] = field(default_factory=dict)

    # Source filters
    sources: list[str] | None = None
    exclude_sources: list[str] | None = None

    # Content filters
    content_contains: str | None = None
    min_content_length: int | None = None
    max_content_length: int | None = None

    # Tenant isolation
    tenant_id: str | None = None

    # Time-based filters
    created_after: float | None = None
    created_before: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {}
        if self.metadata_match:
            result["metadata_match"] = self.metadata_match
        if self.metadata_contains:
            result["metadata_contains"] = self.metadata_contains
        if self.sources:
            result["sources"] = self.sources
        if self.exclude_sources:
            result["exclude_sources"] = self.exclude_sources
        if self.content_contains:
            result["content_contains"] = self.content_contains
        if self.tenant_id:
            result["tenant_id"] = self.tenant_id
        return result


# =============================================================================
# Embedding Cache
# =============================================================================


class EmbeddingCache:
    """LRU cache for embeddings with TTL support."""

    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: int = 3600,
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[list[float], float]] = {}
        self._access_order: list[str] = []
        self._lock = asyncio.Lock()

    def _compute_key(self, text: str, model: str) -> str:
        """Compute cache key from text and model."""
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    async def get(
        self,
        text: str,
        model: str,
    ) -> list[float] | None:
        """Get cached embedding if available and not expired."""
        key = self._compute_key(text, model)

        async with self._lock:
            if key not in self._cache:
                return None

            embedding, timestamp = self._cache[key]

            # Check TTL
            if time.time() - timestamp > self.ttl_seconds:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return None

            # Update access order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

            return embedding

    async def set(
        self,
        text: str,
        model: str,
        embedding: list[float],
    ) -> None:
        """Cache an embedding."""
        key = self._compute_key(text, model)

        async with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self.max_size and self._access_order:
                oldest_key = self._access_order.pop(0)
                self._cache.pop(oldest_key, None)

            self._cache[key] = (embedding, time.time())
            self._access_order.append(key)

    async def clear(self) -> None:
        """Clear the cache."""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
        }


# =============================================================================
# Embedding Service
# =============================================================================


class EmbeddingService:
    """Service for generating text embeddings."""

    def __init__(
        self,
        config: VectorSearchConfig | None = None,
    ):
        self.config = config or VectorSearchConfig()
        self._model = None
        self._model_name = self.config.model_name
        self._cache = (
            EmbeddingCache(
                max_size=self.config.max_cache_size,
                ttl_seconds=self.config.cache_ttl_seconds,
            )
            if self.config.enable_cache
            else None
        )
        self._lock = asyncio.Lock()

    async def _ensure_model(self) -> None:
        """Lazy load the embedding model."""
        if self._model is not None:
            return

        async with self._lock:
            if self._model is not None:
                return

            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading embedding model: {self._model_name}")
                self._model = SentenceTransformer(self._model_name)
                logger.info(
                    f"Loaded embedding model with dimension: {self._model.get_sentence_embedding_dimension()}"
                )

            except ImportError:
                logger.warning("sentence-transformers not available, using fallback")
                self._model = None
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                self._model = None

    async def embed_text(
        self,
        text: str,
        use_cache: bool = True,
    ) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            use_cache: Whether to use caching

        Returns:
            List of floats representing the embedding vector
        """
        # Check cache
        if use_cache and self._cache:
            cached = await self._cache.get(text, self._model_name)
            if cached is not None:
                return cached

        # Generate embedding
        embedding = await self._generate_embedding(text)

        # Cache result
        if use_cache and self._cache:
            await self._cache.set(text, self._model_name, embedding)

        return embedding

    async def _generate_embedding(self, text: str) -> list[float]:
        """Generate embedding using the model."""
        await self._ensure_model()

        if self._model is None:
            # Fallback: generate deterministic pseudo-embedding
            return self._fallback_embedding(text)

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                lambda: self._model.encode(
                    text,
                    normalize_embeddings=self.config.normalize_embeddings,
                    show_progress_bar=False,
                ),
            )
            return embedding.tolist()

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return self._fallback_embedding(text)

    def _fallback_embedding(self, text: str) -> list[float]:
        """Generate a deterministic fallback embedding."""
        # Use hash-based deterministic embedding
        hash_bytes = hashlib.sha256(text.encode()).digest()

        # Expand hash to embedding dimension
        embedding = []
        for i in range(self.config.embedding_dimension):
            # Use different hash segments
            byte_idx = i % len(hash_bytes)
            val = (hash_bytes[byte_idx] + i) / 255.0
            embedding.append(val * 2 - 1)  # Scale to [-1, 1]

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = [v / norm for v in embedding]

        return embedding

    async def batch_embed(
        self,
        texts: list[str],
        use_cache: bool = True,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed
            use_cache: Whether to use caching
            show_progress: Whether to show progress

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        texts_to_embed: list[tuple[int, str]] = []

        # Check cache for each text
        if use_cache and self._cache:
            for i, text in enumerate(texts):
                cached = await self._cache.get(text, self._model_name)
                if cached is not None:
                    results[i] = cached
                else:
                    texts_to_embed.append((i, text))
        else:
            texts_to_embed = list(enumerate(texts))

        # Generate embeddings for uncached texts
        if texts_to_embed:
            await self._ensure_model()

            if self._model is not None:
                # Batch process with model
                uncached_texts = [t for _, t in texts_to_embed]

                loop = asyncio.get_event_loop()
                embeddings = await loop.run_in_executor(
                    None,
                    lambda: self._model.encode(
                        uncached_texts,
                        normalize_embeddings=self.config.normalize_embeddings,
                        show_progress_bar=show_progress,
                        batch_size=self.config.batch_size,
                    ),
                )

                for (idx, text), embedding in zip(texts_to_embed, embeddings):
                    emb_list = embedding.tolist()
                    results[idx] = emb_list
                    if use_cache and self._cache:
                        await self._cache.set(text, self._model_name, emb_list)
            else:
                # Fallback for each
                for idx, text in texts_to_embed:
                    emb = self._fallback_embedding(text)
                    results[idx] = emb
                    if use_cache and self._cache:
                        await self._cache.set(text, self._model_name, emb)

        return [r for r in results if r is not None]

    def cache_stats(self) -> dict[str, Any] | None:
        """Get cache statistics."""
        return self._cache.stats() if self._cache else None


# =============================================================================
# Similarity Functions
# =============================================================================


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)

    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot / (norm_a * norm_b))


def euclidean_distance(a: list[float], b: list[float]) -> float:
    """Compute Euclidean distance between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.linalg.norm(a_arr - b_arr))


def dot_product(a: list[float], b: list[float]) -> float:
    """Compute dot product of two vectors."""
    return float(np.dot(a, b))


def manhattan_distance(a: list[float], b: list[float]) -> float:
    """Compute Manhattan distance between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.sum(np.abs(a_arr - b_arr)))


def compute_similarity(
    a: list[float],
    b: list[float],
    metric: DistanceMetric = DistanceMetric.COSINE,
) -> float:
    """
    Compute similarity between two vectors using specified metric.

    For distance metrics (Euclidean, Manhattan), returns 1/(1+distance)
    to convert to similarity score.
    """
    if metric == DistanceMetric.COSINE:
        return cosine_similarity(a, b)
    elif metric == DistanceMetric.DOT_PRODUCT:
        return dot_product(a, b)
    elif metric == DistanceMetric.EUCLIDEAN:
        distance = euclidean_distance(a, b)
        return 1.0 / (1.0 + distance)
    elif metric == DistanceMetric.MANHATTAN:
        distance = manhattan_distance(a, b)
        return 1.0 / (1.0 + distance)
    else:
        return cosine_similarity(a, b)


# =============================================================================
# Vector Index Interface
# =============================================================================


class VectorIndex(ABC):
    """Abstract base class for vector indices."""

    @abstractmethod
    async def add(
        self,
        id: str,
        embedding: list[float],
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a vector to the index."""
        pass

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: SearchFilter | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete a vector from the index."""
        pass

    @abstractmethod
    async def count(self) -> int:
        """Get the number of vectors in the index."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Release resources."""
        pass


class InMemoryVectorIndex(VectorIndex):
    """In-memory vector index for testing and fallback."""

    def __init__(
        self,
        metric: DistanceMetric = DistanceMetric.COSINE,
    ):
        self.metric = metric
        self._vectors: dict[str, tuple[list[float], str, dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    async def add(
        self,
        id: str,
        embedding: list[float],
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a vector to the index."""
        async with self._lock:
            self._vectors[id] = (embedding, content, metadata or {})

    async def add_batch(
        self,
        items: list[tuple[str, list[float], str, dict[str, Any]]],
    ) -> None:
        """Add multiple vectors to the index."""
        async with self._lock:
            for id, embedding, content, metadata in items:
                self._vectors[id] = (embedding, content, metadata)

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: SearchFilter | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors."""
        results: list[VectorSearchResult] = []

        async with self._lock:
            for id, (embedding, content, metadata) in self._vectors.items():
                # Apply filters
                if filters:
                    if not self._matches_filter(content, metadata, filters):
                        continue

                score = compute_similarity(query_vector, embedding, self.metric)

                results.append(
                    VectorSearchResult(
                        id=id,
                        content=content,
                        score=score,
                        metadata=metadata,
                        source=metadata.get("source"),
                        chunk_index=metadata.get("chunk_index"),
                    )
                )

        # Sort by score descending and return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _matches_filter(
        self,
        content: str,
        metadata: dict[str, Any],
        filters: SearchFilter,
    ) -> bool:
        """Check if a document matches the filter criteria."""
        # Tenant isolation
        if filters.tenant_id and metadata.get("tenant_id") != filters.tenant_id:
            return False

        # Metadata match
        for key, value in filters.metadata_match.items():
            if metadata.get(key) != value:
                return False

        # Metadata contains
        for key, substring in filters.metadata_contains.items():
            meta_val = metadata.get(key, "")
            if isinstance(meta_val, str) and substring not in meta_val:
                return False

        # Source filters
        if filters.sources:
            source = metadata.get("source", "")
            if source not in filters.sources:
                return False

        if filters.exclude_sources:
            source = metadata.get("source", "")
            if source in filters.exclude_sources:
                return False

        # Content filters
        if filters.content_contains:
            if filters.content_contains.lower() not in content.lower():
                return False

        if filters.min_content_length and len(content) < filters.min_content_length:
            return False

        if filters.max_content_length and len(content) > filters.max_content_length:
            return False

        return True

    async def delete(self, id: str) -> bool:
        """Delete a vector from the index."""
        async with self._lock:
            if id in self._vectors:
                del self._vectors[id]
                return True
            return False

    async def count(self) -> int:
        """Get the number of vectors in the index."""
        return len(self._vectors)

    async def clear(self) -> None:
        """Clear all vectors from the index."""
        async with self._lock:
            self._vectors.clear()

    async def close(self) -> None:
        """Release resources."""
        await self.clear()


# =============================================================================
# PgVector Index Adapter
# =============================================================================


class PgVectorIndex(VectorIndex):
    """Vector index using pgvector (PostgreSQL)."""

    def __init__(
        self,
        connection_string: str | None = None,
        table_name: str = "embeddings",
        dimension: int = 384,
    ):
        self.connection_string = connection_string
        self.table_name = table_name
        self.dimension = dimension
        self._pool = None

    async def _ensure_pool(self) -> None:
        """Ensure database connection pool is available."""
        if self._pool is not None:
            return

        try:
            import asyncpg

            self._pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=2,
                max_size=10,
            )
        except ImportError:
            raise RuntimeError("asyncpg not installed for pgvector support")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to PostgreSQL: {e}")

    async def add(
        self,
        id: str,
        embedding: list[float],
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a vector to the index."""
        await self._ensure_pool()

        import json

        query = f"""
            INSERT INTO {self.table_name} (id, embedding, content, metadata)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                content = EXCLUDED.content,
                metadata = EXCLUDED.metadata
        """

        async with self._pool.acquire() as conn:
            await conn.execute(
                query,
                id,
                embedding,
                content,
                json.dumps(metadata or {}),
            )

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: SearchFilter | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors using cosine distance."""
        await self._ensure_pool()

        import json

        # Build filter clause
        where_clauses = []
        params = [query_vector, top_k]
        param_idx = 3

        if filters:
            if filters.tenant_id:
                where_clauses.append(f"metadata->>'tenant_id' = ${param_idx}")
                params.append(filters.tenant_id)
                param_idx += 1

            if filters.content_contains:
                where_clauses.append(f"content ILIKE ${param_idx}")
                params.append(f"%{filters.content_contains}%")
                param_idx += 1

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        query = f"""
            SELECT id, content, metadata,
                   1 - (embedding <=> $1) as similarity
            FROM {self.table_name}
            {where_sql}
            ORDER BY embedding <=> $1
            LIMIT $2
        """

        results = []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

            for row in rows:
                metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                results.append(
                    VectorSearchResult(
                        id=row["id"],
                        content=row["content"],
                        score=float(row["similarity"]),
                        metadata=metadata,
                        source=metadata.get("source"),
                        chunk_index=metadata.get("chunk_index"),
                    )
                )

        return results

    async def delete(self, id: str) -> bool:
        """Delete a vector from the index."""
        await self._ensure_pool()

        query = f"DELETE FROM {self.table_name} WHERE id = $1"

        async with self._pool.acquire() as conn:
            result = await conn.execute(query, id)
            return "DELETE 1" in result

    async def count(self) -> int:
        """Get the number of vectors in the index."""
        await self._ensure_pool()

        query = f"SELECT COUNT(*) FROM {self.table_name}"

        async with self._pool.acquire() as conn:
            result = await conn.fetchval(query)
            return int(result)

    async def close(self) -> None:
        """Release resources."""
        if self._pool:
            await self._pool.close()
            self._pool = None


# =============================================================================
# ChromaDB Index Adapter
# =============================================================================


class ChromaDBIndex(VectorIndex):
    """Vector index using ChromaDB."""

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str | None = None,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._client = None
        self._collection = None

    async def _ensure_collection(self) -> None:
        """Ensure ChromaDB collection is available."""
        if self._collection is not None:
            return

        try:
            import chromadb
            from chromadb.config import Settings

            if self.persist_directory:
                self._client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(anonymized_telemetry=False),
                )
            else:
                self._client = chromadb.Client(Settings(anonymized_telemetry=False))

            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        except ImportError:
            raise RuntimeError("chromadb not installed")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChromaDB: {e}")

    async def add(
        self,
        id: str,
        embedding: list[float],
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a vector to the index."""
        await self._ensure_collection()

        # ChromaDB doesn't support nested metadata
        flat_metadata = self._flatten_metadata(metadata or {})

        self._collection.upsert(
            ids=[id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[flat_metadata],
        )

    def _flatten_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Flatten nested metadata for ChromaDB compatibility."""
        flat = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                flat[key] = value
            elif isinstance(value, list):
                flat[key] = str(value)
            elif isinstance(value, dict):
                flat[key] = str(value)
            else:
                flat[key] = str(value)
        return flat

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: SearchFilter | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors."""
        await self._ensure_collection()

        # Build where clause for ChromaDB
        where = None
        if filters and filters.tenant_id:
            where = {"tenant_id": filters.tenant_id}

        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []

        if results["ids"] and results["ids"][0]:
            for i, id in enumerate(results["ids"][0]):
                # Convert distance to similarity (ChromaDB returns distance)
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = 1.0 - distance

                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                content = results["documents"][0][i] if results["documents"] else ""

                search_results.append(
                    VectorSearchResult(
                        id=id,
                        content=content,
                        score=similarity,
                        metadata=metadata,
                        source=metadata.get("source"),
                        chunk_index=metadata.get("chunk_index"),
                    )
                )

        return search_results

    async def delete(self, id: str) -> bool:
        """Delete a vector from the index."""
        await self._ensure_collection()

        try:
            self._collection.delete(ids=[id])
            return True
        except Exception:
            return False

    async def count(self) -> int:
        """Get the number of vectors in the index."""
        await self._ensure_collection()
        return self._collection.count()

    async def close(self) -> None:
        """Release resources."""
        # ChromaDB client doesn't explicitly require closing in this version,
        # but we can clear references.
        self._client = None
        self._collection = None


# =============================================================================
# SQLite Index Adapter (Fallback)
# =============================================================================


class SQLiteVectorIndex(VectorIndex):
    """
    Vector index fallback for SQLite.
    Performs full table scan and in-memory cosine similarity.
    Suitable for small datasets or development environments.
    """

    def __init__(
        self,
        connection_string: str | None = None,
        table_name: str = "doc_chunks",
    ):
        self.connection_string = connection_string
        self.table_name = table_name
        self._engine = None

    async def _ensure_engine(self) -> None:
        if self._engine is not None:
            return

        from sqlalchemy import create_engine

        self._engine = create_engine(self.connection_string)

    async def add(
        self,
        id: str,
        embedding: list[float],
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a vector to the index (Handled by main DB insert logic typically)."""
        # In this architecture, insertion happens via ORM elsewhere.
        # This method is here to satisfy interface if used directly.
        pass

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: SearchFilter | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors using in-memory cosine similarity."""
        await self._ensure_engine()

        import json

        from sqlalchemy import text

        # Query all chunks (potentially expensive, but necessary for SQLite fallback)
        # Note: In a real app, we might limit by tenant_id first to reduce load

        filter_clauses = []
        params = {}

        if filters:
            if filters.tenant_id:
                filter_clauses.append("tenant_id = :tenant_id")
                params["tenant_id"] = filters.tenant_id

        where_sql = ""
        if filter_clauses:
            where_sql = "WHERE " + " AND ".join(filter_clauses)

        query = text(f"SELECT id, content, embedding, metadata FROM {self.table_name} {where_sql}")

        results = []

        with self._engine.connect() as conn:
            rows = conn.execute(query, params)

            for row in rows:
                row_id = row.id
                content = row.content
                embedding_json = row.embedding
                metadata = row.metadata

                if not embedding_json:
                    continue

                # Deserialize embedding
                if isinstance(embedding_json, str):
                    try:
                        embedding = json.loads(embedding_json)
                    except:
                        continue
                elif isinstance(embedding_json, list):
                    embedding = embedding_json
                else:
                    continue

                # Compute similarity
                score = compute_similarity(query_vector, embedding)

                # Normalize metadata
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}
                elif metadata is None:
                    metadata = {}

                results.append(
                    VectorSearchResult(
                        id=str(row_id),
                        content=content,
                        score=score,
                        metadata=metadata,
                        source=metadata.get("source"),
                        chunk_index=metadata.get("chunk_index"),
                    )
                )

        # Sort by score descending and return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    async def delete(self, id: str) -> bool:
        return False  # Not implemented for fallback

    async def count(self) -> int:
        await self._ensure_engine()
        from sqlalchemy import text

        with self._engine.connect() as conn:
            return conn.execute(text(f"SELECT COUNT(*) FROM {self.table_name}")).scalar()

    async def close(self) -> None:
        """Release resources."""
        if self._engine:
            self._engine.dispose()
            self._engine = None


# =============================================================================
# Unified Query Function
# =============================================================================


async def query_vector_index(
    query_vector: list[float],
    index: VectorIndex,
    top_k: int = 10,
    filters: SearchFilter | None = None,
    min_similarity: float = 0.0,
) -> list[VectorSearchResult]:
    """
    Query a vector index with filtering and minimum similarity threshold.

    Args:
        query_vector: The query embedding vector
        index: The vector index to search
        top_k: Maximum number of results to return
        filters: Optional search filters
        min_similarity: Minimum similarity score (0.0-1.0)

    Returns:
        List of VectorSearchResult ordered by similarity (descending)
    """
    results = await index.search(
        query_vector=query_vector,
        top_k=top_k,
        filters=filters,
    )

    # Apply minimum similarity filter
    if min_similarity > 0:
        results = [r for r in results if r.score >= min_similarity]

    return results


async def hybrid_search(
    query: str,
    embedding_service: EmbeddingService,
    indices: list[VectorIndex],
    top_k: int = 10,
    filters: SearchFilter | None = None,
    merge_strategy: str = "interleave",
) -> list[VectorSearchResult]:
    """
    Perform hybrid search across multiple indices.

    Args:
        query: Natural language query
        embedding_service: Service to generate query embedding
        indices: List of vector indices to search
        top_k: Maximum results per index
        filters: Optional search filters
        merge_strategy: How to merge results ("interleave" or "concatenate")

    Returns:
        Merged list of search results
    """
    # Generate query embedding
    query_vector = await embedding_service.embed_text(query)

    # Search all indices concurrently
    search_tasks = [query_vector_index(query_vector, index, top_k, filters) for index in indices]

    all_results = await asyncio.gather(*search_tasks, return_exceptions=True)

    # Flatten results
    merged: list[VectorSearchResult] = []
    for results in all_results:
        if isinstance(results, Exception):
            logger.error(f"Index search failed: {results}")
            continue
        merged.extend(results)

    # Deduplicate by ID, keeping highest score
    seen: dict[str, VectorSearchResult] = {}
    for result in merged:
        if result.id not in seen or result.score > seen[result.id].score:
            seen[result.id] = result

    # Sort by score
    final_results = sorted(seen.values(), key=lambda x: x.score, reverse=True)

    return final_results[:top_k]


# =============================================================================
# Convenience Functions
# =============================================================================

# Global embedding service instance
_embedding_service: EmbeddingService | None = None


def get_embedding_service(config: VectorSearchConfig | None = None) -> EmbeddingService:
    """Get or create the global embedding service."""
    global _embedding_service

    if _embedding_service is None:
        _embedding_service = EmbeddingService(config)

    return _embedding_service


async def embed_text(
    text: str,
    config: VectorSearchConfig | None = None,
) -> list[float]:
    """
    Convenience function to embed text using the global service.

    Args:
        text: Text to embed
        config: Optional configuration

    Returns:
        Embedding vector as list of floats
    """
    service = get_embedding_service(config)
    return await service.embed_text(text)


async def batch_embed_texts(
    texts: list[str],
    config: VectorSearchConfig | None = None,
) -> list[list[float]]:
    """
    Convenience function to batch embed texts using the global service.

    Args:
        texts: List of texts to embed
        config: Optional configuration

    Returns:
        List of embedding vectors
    """
    service = get_embedding_service(config)
    return await service.batch_embed(texts)


def create_in_memory_index(
    metric: DistanceMetric = DistanceMetric.COSINE,
) -> InMemoryVectorIndex:
    """Create an in-memory vector index."""
    return InMemoryVectorIndex(metric=metric)


def create_pgvector_index(
    connection_string: str,
    table_name: str = "embeddings",
    dimension: int = 384,
) -> PgVectorIndex:
    """Create a pgvector index."""
    return PgVectorIndex(
        connection_string=connection_string,
        table_name=table_name,
        dimension=dimension,
    )


def create_chromadb_index(
    collection_name: str = "documents",
    persist_directory: str | None = None,
) -> ChromaDBIndex:
    """Create a ChromaDB index."""
    return ChromaDBIndex(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )


def create_sqlite_index(
    connection_string: str,
    table_name: str = "doc_chunks",
) -> SQLiteVectorIndex:
    """Create a SQLite vector index."""
    return SQLiteVectorIndex(
        connection_string=connection_string,
        table_name=table_name,
    )
