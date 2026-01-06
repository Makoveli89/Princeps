"""
RetrieverAgent - Knowledge Retrieval Agent with Vector Search

This module implements the RetrieverAgent that provides intelligent knowledge retrieval:
- Vector search using pgvector, ChromaDB, or in-memory indices
- Semantic search with embedding-based similarity
- Fallback chain: pgvector → ChromaDB → TF-IDF → Heuristic
- Query understanding and expansion using LLM
- Result reranking and filtering
- Multi-source aggregation
- Brain Layer integration for logging retrieval operations

The RetrieverAgent acts as the knowledge interface between agents and the Brain Layer's
persistent knowledge stores. It abstracts away the complexity of vector search while
providing intelligent query processing and result curation.

Strategic Intent:
When other agents need to retrieve relevant context from the knowledge base, they call
the RetrieverAgent. It takes a natural language query, converts it to an embedding,
searches the vector store, and returns the most relevant chunks. It can optionally
expand or rephrase the query using an LLM to improve recall.

Adapted from patterns in:
- brain_layer/05_retrieval_systems/unified_retriever.py (fallback chain)
- brain_layer/05_retrieval_systems/vector_store.py (pgvector integration)
- brain_layer/05_retrieval_systems/embedding_retriever.py (embedding patterns)
- brain_layer/05_retrieval_systems/tfidf_librarian.py (TF-IDF fallback)
"""

import asyncio
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from framework.agents.base_agent import (
    AgentConfig,
    AgentContext,
    AgentResponse,
    AgentTask,
    BaseAgent,
    LLMProvider,
    TaskStatus,
)
from framework.agents.brain_logger import BrainLogger, get_brain_logger
from framework.agents.schemas.agent_run import (
    AgentRunCreate,
    AgentRunUpdate,
    RunStatus,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Configuration
# =============================================================================


class RetrievalMode(Enum):
    """Mode of retrieval operation."""

    SEMANTIC = "semantic"  # Pure vector similarity search
    KEYWORD = "keyword"  # TF-IDF / BM25 keyword matching
    HYBRID = "hybrid"  # Combination of semantic + keyword
    EXPANDED = "expanded"  # LLM-expanded query
    MULTI_QUERY = "multi_query"  # Generate multiple queries for recall


class RetrievalSource(Enum):
    """Source backends for retrieval."""

    PGVECTOR = "pgvector"
    CHROMADB = "chromadb"
    TFIDF = "tfidf"
    HEURISTIC = "heuristic"
    IN_MEMORY = "in_memory"


class RerankerType(Enum):
    """Type of reranking to apply."""

    NONE = "none"
    CROSS_ENCODER = "cross_encoder"  # Use cross-encoder model
    LLM = "llm"  # Use LLM for reranking
    MMR = "mmr"  # Maximal Marginal Relevance
    RECIPROCAL_RANK = "reciprocal_rank"  # Reciprocal rank fusion


@dataclass
class RetrieverConfig:
    """Configuration for the RetrieverAgent."""

    # Search settings
    default_top_k: int = 10
    max_top_k: int = 100
    min_similarity: float = 0.3
    default_mode: RetrievalMode = RetrievalMode.HYBRID

    # Source priority (fallback chain order)
    source_priority: list[RetrievalSource] = field(
        default_factory=lambda: [
            RetrievalSource.PGVECTOR,
            RetrievalSource.CHROMADB,
            RetrievalSource.TFIDF,
            RetrievalSource.HEURISTIC,
        ]
    )

    # Query expansion settings
    enable_query_expansion: bool = True
    expansion_model: LLMProvider = LLMProvider.ANTHROPIC
    max_expanded_queries: int = 3

    # Reranking settings
    reranker: RerankerType = RerankerType.MMR
    mmr_lambda: float = 0.5  # Diversity vs relevance tradeoff (0=diverse, 1=relevant)

    # Caching
    enable_cache: bool = True
    cache_ttl_seconds: int = 300
    max_cache_size: int = 1000

    # Performance
    search_timeout_seconds: float = 10.0
    max_concurrent_searches: int = 3

    # Embedding settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Brain Layer integration
    enable_brain_logging: bool = True


@dataclass
class RetrievalFilter:
    """Filters for retrieval queries."""

    # Metadata filters
    metadata_match: dict[str, Any] = field(default_factory=dict)
    metadata_contains: dict[str, str] = field(default_factory=dict)

    # Source filters
    sources: list[str] | None = None
    exclude_sources: list[str] | None = None
    document_types: list[str] | None = None

    # Content filters
    content_contains: str | None = None
    min_content_length: int | None = None
    max_content_length: int | None = None

    # Time filters
    created_after: datetime | None = None
    created_before: datetime | None = None

    # Tenant isolation
    tenant_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = {}
        if self.metadata_match:
            result["metadata_match"] = self.metadata_match
        if self.sources:
            result["sources"] = self.sources
        if self.document_types:
            result["document_types"] = self.document_types
        if self.tenant_id:
            result["tenant_id"] = self.tenant_id
        return result


@dataclass
class RetrievalResult:
    """Single result from framework.retrieval."""

    id: str
    content: str
    score: float
    source: RetrievalSource
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_index: int | None = None
    document_id: str | None = None
    document_title: str | None = None
    highlight: str | None = None  # Highlighted snippet

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "source": self.source.value,
            "metadata": self.metadata,
            "chunk_index": self.chunk_index,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "highlight": self.highlight,
        }


@dataclass
class RetrievalResponse:
    """Response from a retrieval operation."""

    query: str
    results: list[RetrievalResult]
    total_found: int
    sources_searched: list[RetrievalSource]
    mode_used: RetrievalMode
    expanded_queries: list[str] | None = None
    search_time_ms: float = 0.0
    cache_hit: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "total_found": self.total_found,
            "sources_searched": [s.value for s in self.sources_searched],
            "mode_used": self.mode_used.value,
            "expanded_queries": self.expanded_queries,
            "search_time_ms": self.search_time_ms,
            "cache_hit": self.cache_hit,
            "metadata": self.metadata,
        }


# =============================================================================
# Query Cache
# =============================================================================


class QueryCache:
    """LRU cache for retrieval results with TTL."""

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 300,
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[RetrievalResponse, float]] = {}
        self._access_order: list[str] = []
        self._lock = asyncio.Lock()

    def _compute_key(
        self,
        query: str,
        filters: RetrievalFilter | None,
        top_k: int,
        mode: RetrievalMode,
    ) -> str:
        """Compute cache key from query parameters."""
        filter_str = str(filters.to_dict()) if filters else ""
        content = f"{query}:{filter_str}:{top_k}:{mode.value}"
        return hashlib.sha256(content.encode()).hexdigest()

    async def get(
        self,
        query: str,
        filters: RetrievalFilter | None,
        top_k: int,
        mode: RetrievalMode,
    ) -> RetrievalResponse | None:
        """Get cached result if available and not expired."""
        key = self._compute_key(query, filters, top_k, mode)

        async with self._lock:
            if key not in self._cache:
                return None

            response, timestamp = self._cache[key]

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

            return response

    async def set(
        self,
        query: str,
        filters: RetrievalFilter | None,
        top_k: int,
        mode: RetrievalMode,
        response: RetrievalResponse,
    ) -> None:
        """Cache a retrieval response."""
        key = self._compute_key(query, filters, top_k, mode)

        async with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self.max_size and self._access_order:
                oldest_key = self._access_order.pop(0)
                self._cache.pop(oldest_key, None)

            self._cache[key] = (response, time.time())
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
# RetrieverAgent Implementation
# =============================================================================


class RetrieverAgent(BaseAgent):
    """
    Knowledge Retrieval Agent with vector search and fallback chain.

    The RetrieverAgent provides:
    - Semantic search using embeddings
    - Multi-source retrieval with fallback
    - Query expansion for better recall
    - Result reranking for quality
    - Caching for performance

    Usage:
        retriever = RetrieverAgent(config=config)
        results = await retriever.retrieve(
            query="How do I implement authentication?",
            filters=RetrievalFilter(tenant_id="tenant_123"),
            top_k=5,
        )
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        retriever_config: RetrieverConfig | None = None,
        context: AgentContext | None = None,
        brain_logger: BrainLogger | None = None,
    ):
        # Initialize base agent
        super().__init__(config=config, context=context)

        self.retriever_config = retriever_config or RetrieverConfig()
        self.brain_logger = brain_logger or get_brain_logger()

        # Initialize cache
        self._cache = (
            QueryCache(
                max_size=self.retriever_config.max_cache_size,
                ttl_seconds=self.retriever_config.cache_ttl_seconds,
            )
            if self.retriever_config.enable_cache
            else None
        )

        # Initialize embedding service (lazy loaded)
        self._embedding_service = None

        # Source adapters (lazy loaded)
        self._source_adapters: dict[RetrievalSource, Any] = {}

        # Statistics
        self._stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "source_usage": {s.value: 0 for s in RetrievalSource},
            "average_latency_ms": 0.0,
        }

    async def _get_embedding_service(self):
        """Lazy load embedding service."""
        if self._embedding_service is None:
            try:
                from framework.retrieval.vector_search import EmbeddingService, VectorSearchConfig

                config = VectorSearchConfig(
                    model_name=self.retriever_config.embedding_model,
                    embedding_dimension=self.retriever_config.embedding_dimension,
                    enable_cache=True,
                )
                self._embedding_service = EmbeddingService(config)
            except ImportError:
                logger.warning("vector_search module not available, using fallback")
                self._embedding_service = None

        return self._embedding_service

    async def _get_source_adapter(self, source: RetrievalSource):
        """Get or create a source adapter."""
        if source in self._source_adapters:
            return self._source_adapters[source]

        adapter = await self._create_source_adapter(source)
        if adapter:
            self._source_adapters[source] = adapter

        return adapter

    async def _create_source_adapter(self, source: RetrievalSource):
        """Create a source adapter for the given source type."""
        try:
            if source == RetrievalSource.PGVECTOR:
                return await self._create_pgvector_adapter()
            elif source == RetrievalSource.CHROMADB:
                return await self._create_chromadb_adapter()
            elif source == RetrievalSource.TFIDF:
                return await self._create_tfidf_adapter()
            elif source == RetrievalSource.IN_MEMORY:
                return await self._create_inmemory_adapter()
            elif source == RetrievalSource.HEURISTIC:
                return self._create_heuristic_adapter()
        except Exception as e:
            logger.warning(f"Failed to create adapter for {source.value}: {e}")
            return None

    async def _create_pgvector_adapter(self):
        """Create pgvector adapter using Brain Layer components."""
        try:
            # Try to import from Brain Layer (dynamic import for numbered module)
            import importlib

            vector_store_module = importlib.import_module(
                "brain_layer.05_retrieval_systems.vector_store"
            )
            VectorStore = getattr(vector_store_module, "VectorStore")
            return VectorStore()
        except (ImportError, AttributeError):
            try:
                import os

                from framework.retrieval.vector_search import PgVectorIndex

                conn_string = os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL")
                if conn_string:
                    return PgVectorIndex(
                        connection_string=conn_string,
                        dimension=self.retriever_config.embedding_dimension,
                    )
            except ImportError:
                pass
        return None

    async def _create_chromadb_adapter(self):
        """Create ChromaDB adapter."""
        try:
            from framework.retrieval.vector_search import ChromaDBIndex

            return ChromaDBIndex(
                collection_name="princeps_knowledge",
                persist_directory="./data/chromadb",
            )
        except ImportError:
            return None

    async def _create_tfidf_adapter(self):
        """Create TF-IDF adapter using Brain Layer components."""
        try:
            import importlib

            tfidf_module = importlib.import_module(
                "brain_layer.05_retrieval_systems.tfidf_librarian"
            )
            TFIDFLibrarian = getattr(tfidf_module, "TFIDFLibrarian")
            return TFIDFLibrarian()
        except (ImportError, AttributeError):
            return None

    async def _create_inmemory_adapter(self):
        """Create in-memory vector index."""
        try:
            from framework.retrieval.vector_search import InMemoryVectorIndex

            return InMemoryVectorIndex()
        except ImportError:
            return None

    def _create_heuristic_adapter(self):
        """Create heuristic search adapter (simple substring matching)."""
        return HeuristicSearchAdapter()

    # =========================================================================
    # Core Retrieval Interface
    # =========================================================================

    async def retrieve(
        self,
        query: str,
        filters: RetrievalFilter | None = None,
        top_k: int | None = None,
        mode: RetrievalMode | None = None,
        rerank: bool = True,
    ) -> RetrievalResponse:
        """
        Retrieve relevant knowledge for a query.

        Args:
            query: Natural language query
            filters: Optional filters to apply
            top_k: Maximum number of results (default from config)
            mode: Retrieval mode (default from config)
            rerank: Whether to rerank results

        Returns:
            RetrievalResponse with results and metadata
        """
        start_time = time.time()

        # Apply defaults
        top_k = min(top_k or self.retriever_config.default_top_k, self.retriever_config.max_top_k)
        mode = mode or self.retriever_config.default_mode

        # Apply tenant isolation from context if not in filters
        if filters is None:
            filters = RetrievalFilter()
        if filters.tenant_id is None and self.context and self.context.tenant_id:
            filters.tenant_id = self.context.tenant_id

        # Check cache
        if self._cache:
            cached = await self._cache.get(query, filters, top_k, mode)
            if cached:
                self._stats["cache_hits"] += 1
                cached.cache_hit = True
                return cached

        # Log retrieval start
        run_id = str(uuid.uuid4())
        if self.retriever_config.enable_brain_logging and self.brain_logger:
            await self._log_retrieval_start(run_id, query, filters, mode)

        try:
            # Execute retrieval based on mode
            if mode == RetrievalMode.EXPANDED:
                results = await self._retrieve_with_expansion(query, filters, top_k)
            elif mode == RetrievalMode.MULTI_QUERY:
                results = await self._retrieve_multi_query(query, filters, top_k)
            elif mode == RetrievalMode.HYBRID:
                results = await self._retrieve_hybrid(query, filters, top_k)
            elif mode == RetrievalMode.KEYWORD:
                results = await self._retrieve_keyword(query, filters, top_k)
            else:  # SEMANTIC
                results = await self._retrieve_semantic(query, filters, top_k)

            # Rerank if enabled
            if rerank and results:
                results = await self._rerank_results(query, results)

            # Apply minimum similarity filter
            results = [r for r in results if r.score >= self.retriever_config.min_similarity]

            # Limit to top_k
            results = results[:top_k]

            # Build response
            search_time_ms = (time.time() - start_time) * 1000

            response = RetrievalResponse(
                query=query,
                results=results,
                total_found=len(results),
                sources_searched=list(set(r.source for r in results)) if results else [],
                mode_used=mode,
                search_time_ms=search_time_ms,
                cache_hit=False,
            )

            # Update cache
            if self._cache:
                await self._cache.set(query, filters, top_k, mode, response)

            # Update statistics
            self._update_stats(response)

            # Log retrieval completion
            if self.retriever_config.enable_brain_logging and self.brain_logger:
                await self._log_retrieval_complete(run_id, response)

            return response

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            if self.retriever_config.enable_brain_logging and self.brain_logger:
                await self._log_retrieval_error(run_id, str(e))

            # Return empty response on error
            return RetrievalResponse(
                query=query,
                results=[],
                total_found=0,
                sources_searched=[],
                mode_used=mode,
                search_time_ms=(time.time() - start_time) * 1000,
                metadata={"error": str(e)},
            )

    async def _retrieve_semantic(
        self,
        query: str,
        filters: RetrievalFilter | None,
        top_k: int,
    ) -> list[RetrievalResult]:
        """Perform semantic search using embeddings."""
        embedding_service = await self._get_embedding_service()
        if embedding_service is None:
            logger.warning("No embedding service available, falling back to keyword search")
            return await self._retrieve_keyword(query, filters, top_k)

        # Generate query embedding
        query_embedding = await embedding_service.embed_text(query)

        # Try sources in priority order
        for source in self.retriever_config.source_priority:
            if source in [RetrievalSource.TFIDF, RetrievalSource.HEURISTIC]:
                continue  # These don't support vector search

            adapter = await self._get_source_adapter(source)
            if adapter is None:
                continue

            try:
                results = await self._search_source(
                    adapter=adapter,
                    source=source,
                    query_embedding=query_embedding,
                    filters=filters,
                    top_k=top_k,
                )

                if results:
                    self._stats["source_usage"][source.value] += 1
                    return results

            except Exception as e:
                logger.warning(f"Search failed for {source.value}: {e}")
                continue

        # Fallback to keyword search if all vector sources fail
        return await self._retrieve_keyword(query, filters, top_k)

    async def _retrieve_keyword(
        self,
        query: str,
        filters: RetrievalFilter | None,
        top_k: int,
    ) -> list[RetrievalResult]:
        """Perform keyword-based search using TF-IDF or heuristic."""
        for source in [RetrievalSource.TFIDF, RetrievalSource.HEURISTIC]:
            adapter = await self._get_source_adapter(source)
            if adapter is None:
                continue

            try:
                results = await self._search_keyword_source(
                    adapter=adapter,
                    source=source,
                    query=query,
                    filters=filters,
                    top_k=top_k,
                )

                if results:
                    self._stats["source_usage"][source.value] += 1
                    return results

            except Exception as e:
                logger.warning(f"Keyword search failed for {source.value}: {e}")
                continue

        return []

    async def _retrieve_hybrid(
        self,
        query: str,
        filters: RetrievalFilter | None,
        top_k: int,
    ) -> list[RetrievalResult]:
        """Perform hybrid search combining semantic and keyword."""
        # Run both in parallel
        semantic_task = asyncio.create_task(self._retrieve_semantic(query, filters, top_k))
        keyword_task = asyncio.create_task(self._retrieve_keyword(query, filters, top_k))

        semantic_results, keyword_results = await asyncio.gather(
            semantic_task,
            keyword_task,
            return_exceptions=True,
        )

        # Handle exceptions
        if isinstance(semantic_results, Exception):
            semantic_results = []
        if isinstance(keyword_results, Exception):
            keyword_results = []

        # Merge results using reciprocal rank fusion
        return self._reciprocal_rank_fusion(
            [semantic_results, keyword_results],
            top_k=top_k,
        )

    async def _retrieve_with_expansion(
        self,
        query: str,
        filters: RetrievalFilter | None,
        top_k: int,
    ) -> list[RetrievalResult]:
        """Retrieve with LLM-expanded query."""
        # Expand query using LLM
        expanded_queries = await self._expand_query(query)

        # Search with all queries
        all_results: list[list[RetrievalResult]] = []

        for expanded_query in [query] + expanded_queries:
            results = await self._retrieve_semantic(expanded_query, filters, top_k)
            all_results.append(results)

        # Merge using reciprocal rank fusion
        return self._reciprocal_rank_fusion(all_results, top_k=top_k)

    async def _retrieve_multi_query(
        self,
        query: str,
        filters: RetrievalFilter | None,
        top_k: int,
    ) -> list[RetrievalResult]:
        """Generate multiple query variations and merge results."""
        # Generate query variations
        variations = await self._generate_query_variations(query)

        # Search all variations concurrently
        tasks = [self._retrieve_semantic(q, filters, top_k) for q in [query] + variations]

        all_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = [r for r in all_results if isinstance(r, list)]

        return self._reciprocal_rank_fusion(valid_results, top_k=top_k)

    # =========================================================================
    # Query Expansion
    # =========================================================================

    async def _expand_query(self, query: str) -> list[str]:
        """Expand query using LLM to improve recall."""
        if not self.retriever_config.enable_query_expansion:
            return []

        try:
            prompt = f"""Given the search query below, generate {self.retriever_config.max_expanded_queries} alternative phrasings that capture the same intent but might match different documents.

Query: {query}

Return only the alternative queries, one per line. Do not include the original query."""

            # Use base agent's LLM capabilities
            response = await self._call_llm(prompt)

            if response and response.output:
                # Parse response into list of queries
                expanded = [
                    line.strip()
                    for line in response.output.strip().split("\n")
                    if line.strip() and not line.startswith("-")
                ]
                return expanded[: self.retriever_config.max_expanded_queries]

        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")

        return []

    async def _generate_query_variations(self, query: str) -> list[str]:
        """Generate query variations for multi-query retrieval."""
        try:
            prompt = f"""Generate 3 different ways to ask the following question. Each should capture the same intent but use different words or perspectives.

Original: {query}

Return only the variations, one per line."""

            response = await self._call_llm(prompt)

            if response and response.output:
                variations = [
                    line.strip() for line in response.output.strip().split("\n") if line.strip()
                ]
                return variations[:3]

        except Exception as e:
            logger.warning(f"Query variation generation failed: {e}")

        return []

    # =========================================================================
    # Search Helpers
    # =========================================================================

    async def _search_source(
        self,
        adapter: Any,
        source: RetrievalSource,
        query_embedding: list[float],
        filters: RetrievalFilter | None,
        top_k: int,
    ) -> list[RetrievalResult]:
        """Search a specific source with vector embedding."""
        try:
            # Convert filters to source-specific format
            search_filters = self._convert_filters(filters, source)

            # Execute search based on adapter type
            if hasattr(adapter, "search"):
                raw_results = await adapter.search(
                    query_vector=query_embedding,
                    top_k=top_k,
                    filters=search_filters,
                )
            elif hasattr(adapter, "query"):
                raw_results = await adapter.query(
                    embedding=query_embedding,
                    k=top_k,
                    **search_filters,
                )
            else:
                logger.warning(f"Unknown adapter interface for {source.value}")
                return []

            # Convert to RetrievalResult
            return self._convert_results(raw_results, source)

        except asyncio.TimeoutError:
            logger.warning(f"Search timeout for {source.value}")
            return []
        except Exception as e:
            logger.error(f"Search error for {source.value}: {e}")
            return []

    async def _search_keyword_source(
        self,
        adapter: Any,
        source: RetrievalSource,
        query: str,
        filters: RetrievalFilter | None,
        top_k: int,
    ) -> list[RetrievalResult]:
        """Search a keyword-based source."""
        try:
            if hasattr(adapter, "search_text"):
                raw_results = await adapter.search_text(query, top_k)
            elif hasattr(adapter, "search"):
                raw_results = adapter.search(query, top_k)
            else:
                return []

            return self._convert_results(raw_results, source)

        except Exception as e:
            logger.error(f"Keyword search error for {source.value}: {e}")
            return []

    def _convert_filters(
        self,
        filters: RetrievalFilter | None,
        source: RetrievalSource,
    ) -> dict[str, Any]:
        """Convert RetrievalFilter to source-specific format."""
        if filters is None:
            return {}

        result = {}

        if filters.tenant_id:
            result["tenant_id"] = filters.tenant_id

        if filters.metadata_match:
            result["metadata_match"] = filters.metadata_match

        if filters.sources:
            result["sources"] = filters.sources

        return result

    def _convert_results(
        self,
        raw_results: Any,
        source: RetrievalSource,
    ) -> list[RetrievalResult]:
        """Convert source-specific results to RetrievalResult."""
        results = []

        if raw_results is None:
            return results

        # Handle list of results
        for item in raw_results:
            try:
                if hasattr(item, "id"):
                    # VectorSearchResult-like object
                    results.append(
                        RetrievalResult(
                            id=str(item.id),
                            content=getattr(item, "content", ""),
                            score=float(getattr(item, "score", 0.0)),
                            source=source,
                            metadata=getattr(item, "metadata", {}),
                            chunk_index=getattr(item, "chunk_index", None),
                            document_id=getattr(item, "document_id", None),
                        )
                    )
                elif isinstance(item, dict):
                    results.append(
                        RetrievalResult(
                            id=str(item.get("id", uuid.uuid4())),
                            content=item.get("content", item.get("text", "")),
                            score=float(item.get("score", item.get("similarity", 0.0))),
                            source=source,
                            metadata=item.get("metadata", {}),
                            chunk_index=item.get("chunk_index"),
                            document_id=item.get("document_id"),
                        )
                    )
                elif isinstance(item, tuple) and len(item) >= 2:
                    # (content, score) tuple
                    results.append(
                        RetrievalResult(
                            id=str(uuid.uuid4()),
                            content=str(item[0]),
                            score=float(item[1]),
                            source=source,
                        )
                    )

            except Exception as e:
                logger.warning(f"Failed to convert result: {e}")
                continue

        return results

    # =========================================================================
    # Reranking
    # =========================================================================

    async def _rerank_results(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Rerank results based on configured strategy."""
        if not results:
            return results

        reranker = self.retriever_config.reranker

        if reranker == RerankerType.NONE:
            return results
        elif reranker == RerankerType.MMR:
            return await self._mmr_rerank(query, results)
        elif reranker == RerankerType.RECIPROCAL_RANK:
            # Already in correct order from search
            return results
        elif reranker == RerankerType.LLM:
            return await self._llm_rerank(query, results)
        elif reranker == RerankerType.CROSS_ENCODER:
            return await self._cross_encoder_rerank(query, results)

        return results

    async def _mmr_rerank(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Maximal Marginal Relevance reranking for diversity."""
        if len(results) <= 1:
            return results

        embedding_service = await self._get_embedding_service()
        if embedding_service is None:
            return results

        try:
            # Get embeddings for all results
            contents = [r.content for r in results]
            embeddings = await embedding_service.batch_embed(contents)

            # Get query embedding
            query_embedding = await embedding_service.embed_text(query)

            # MMR selection
            lambda_param = self.retriever_config.mmr_lambda
            selected_indices = []
            remaining_indices = list(range(len(results)))

            while remaining_indices and len(selected_indices) < len(results):
                mmr_scores = []

                for idx in remaining_indices:
                    # Relevance to query
                    from framework.retrieval.vector_search import cosine_similarity

                    relevance = cosine_similarity(query_embedding, embeddings[idx])

                    # Max similarity to already selected
                    max_sim = 0.0
                    for sel_idx in selected_indices:
                        sim = cosine_similarity(embeddings[idx], embeddings[sel_idx])
                        max_sim = max(max_sim, sim)

                    # MMR score
                    mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                    mmr_scores.append((idx, mmr))

                # Select highest MMR score
                best_idx, _ = max(mmr_scores, key=lambda x: x[1])
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)

            return [results[i] for i in selected_indices]

        except Exception as e:
            logger.warning(f"MMR reranking failed: {e}")
            return results

    async def _llm_rerank(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Use LLM to rerank results by relevance."""
        if len(results) <= 1:
            return results

        try:
            # Build prompt with results
            results_text = "\n\n".join(
                [
                    f"Document {i + 1}:\n{r.content[:500]}..."
                    for i, r in enumerate(results[:10])  # Limit to 10 for context
                ]
            )

            prompt = f"""Rank the following documents by relevance to the query.
Query: {query}

{results_text}

Return the document numbers in order of relevance (most relevant first), separated by commas.
Example: 3, 1, 5, 2, 4"""

            response = await self._call_llm(prompt)

            if response and response.output:
                # Parse ranking
                ranking = []
                for num in response.output.strip().split(","):
                    try:
                        idx = int(num.strip()) - 1
                        if 0 <= idx < len(results):
                            ranking.append(idx)
                    except ValueError:
                        continue

                # Reorder results
                reranked = [results[i] for i in ranking]
                # Add any missing results at the end
                for i, r in enumerate(results):
                    if i not in ranking:
                        reranked.append(r)

                return reranked

        except Exception as e:
            logger.warning(f"LLM reranking failed: {e}")

        return results

    async def _cross_encoder_rerank(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Use cross-encoder model for reranking."""
        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

            # Score each result
            pairs = [(query, r.content) for r in results]
            scores = model.predict(pairs)

            # Sort by score
            scored_results = list(zip(results, scores))
            scored_results.sort(key=lambda x: x[1], reverse=True)

            return [r for r, _ in scored_results]

        except ImportError:
            logger.warning("Cross-encoder not available, skipping reranking")
            return results
        except Exception as e:
            logger.warning(f"Cross-encoder reranking failed: {e}")
            return results

    def _reciprocal_rank_fusion(
        self,
        result_lists: list[list[RetrievalResult]],
        k: int = 60,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Merge results using Reciprocal Rank Fusion."""
        scores: dict[str, float] = {}
        result_map: dict[str, RetrievalResult] = {}

        for results in result_lists:
            for rank, result in enumerate(results):
                if result.id not in result_map:
                    result_map[result.id] = result
                    scores[result.id] = 0.0

                # RRF score
                scores[result.id] += 1.0 / (k + rank + 1)

        # Sort by RRF score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Update scores and return
        merged = []
        for id in sorted_ids[:top_k]:
            result = result_map[id]
            result.score = scores[id]
            merged.append(result)

        return merged

    # =========================================================================
    # Logging
    # =========================================================================

    async def _log_retrieval_start(
        self,
        run_id: str,
        query: str,
        filters: RetrievalFilter | None,
        mode: RetrievalMode,
    ) -> None:
        """Log retrieval start to Brain Layer."""
        try:
            run_create = AgentRunCreate(
                agent_id=f"retriever_{self.agent_id}",
                agent_type="retriever",
                tenant_id=self.context.tenant_id if self.context else None,
                input_summary=query[:200],
                metadata={
                    "mode": mode.value,
                    "filters": filters.to_dict() if filters else {},
                },
            )

            await self.brain_logger.log_run_start(run_create)

        except Exception as e:
            logger.warning(f"Failed to log retrieval start: {e}")

    async def _log_retrieval_complete(
        self,
        run_id: str,
        response: RetrievalResponse,
    ) -> None:
        """Log retrieval completion to Brain Layer."""
        try:
            run_update = AgentRunUpdate(
                status=RunStatus.COMPLETED,
                output_summary=f"Found {response.total_found} results",
                metrics={
                    "results_count": response.total_found,
                    "search_time_ms": response.search_time_ms,
                    "cache_hit": response.cache_hit,
                    "sources": [s.value for s in response.sources_searched],
                },
            )

            await self.brain_logger.log_run_update(run_id, run_update)

        except Exception as e:
            logger.warning(f"Failed to log retrieval completion: {e}")

    async def _log_retrieval_error(
        self,
        run_id: str,
        error: str,
    ) -> None:
        """Log retrieval error to Brain Layer."""
        try:
            run_update = AgentRunUpdate(
                status=RunStatus.FAILED,
                error_message=error,
            )

            await self.brain_logger.log_run_update(run_id, run_update)

        except Exception as e:
            logger.warning(f"Failed to log retrieval error: {e}")

    def _update_stats(self, response: RetrievalResponse) -> None:
        """Update internal statistics."""
        self._stats["total_queries"] += 1

        # Update running average latency
        n = self._stats["total_queries"]
        old_avg = self._stats["average_latency_ms"]
        self._stats["average_latency_ms"] = old_avg + (response.search_time_ms - old_avg) / n

    # =========================================================================
    # BaseAgent Interface Implementation
    # =========================================================================

    async def _execute_task(self, task: AgentTask) -> AgentResponse:
        """Execute a retrieval task."""
        query = task.input_data.get("query", "")
        top_k = task.input_data.get("top_k", self.retriever_config.default_top_k)
        mode_str = task.input_data.get("mode", self.retriever_config.default_mode.value)

        try:
            mode = RetrievalMode(mode_str)
        except ValueError:
            mode = self.retriever_config.default_mode

        # Build filters from task
        filters = None
        if "filters" in task.input_data:
            filter_data = task.input_data["filters"]
            filters = RetrievalFilter(
                metadata_match=filter_data.get("metadata_match", {}),
                sources=filter_data.get("sources"),
                tenant_id=filter_data.get("tenant_id"),
            )

        response = await self.retrieve(
            query=query,
            filters=filters,
            top_k=top_k,
            mode=mode,
        )

        return AgentResponse(
            task_id=task.task_id,
            status=TaskStatus.COMPLETED,
            output=response.to_dict(),
            confidence=1.0 if response.results else 0.5,
            metadata={
                "search_time_ms": response.search_time_ms,
                "results_count": len(response.results),
            },
        )

    def get_stats(self) -> dict[str, Any]:
        """Get retriever statistics."""
        return {
            **self._stats,
            "cache_stats": self._cache.stats() if self._cache else None,
        }

    async def clear_cache(self) -> None:
        """Clear the query cache."""
        if self._cache:
            await self._cache.clear()


# =============================================================================
# Heuristic Search Adapter
# =============================================================================


class HeuristicSearchAdapter:
    """Simple substring-based search as last resort fallback."""

    def __init__(self):
        self._documents: dict[str, tuple[str, dict[str, Any]]] = {}

    def add_document(
        self,
        id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a document to the index."""
        self._documents[id] = (content, metadata or {})

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Search documents using substring matching."""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        results = []

        for id, (content, metadata) in self._documents.items():
            content_lower = content.lower()

            # Score based on word matches
            content_words = set(content_lower.split())
            common_words = query_words & content_words
            score = len(common_words) / max(len(query_words), 1)

            # Boost for exact substring match
            if query_lower in content_lower:
                score += 0.5

            if score > 0:
                results.append(
                    {
                        "id": id,
                        "content": content,
                        "score": min(score, 1.0),
                        "metadata": metadata,
                    }
                )

        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]


# =============================================================================
# Factory Function
# =============================================================================


def create_retriever_agent(
    config: AgentConfig | None = None,
    retriever_config: RetrieverConfig | None = None,
    context: AgentContext | None = None,
) -> RetrieverAgent:
    """
    Factory function to create a RetrieverAgent.

    Args:
        config: Base agent configuration
        retriever_config: Retriever-specific configuration
        context: Agent execution context

    Returns:
        Configured RetrieverAgent instance
    """
    return RetrieverAgent(
        config=config,
        retriever_config=retriever_config,
        context=context,
    )
