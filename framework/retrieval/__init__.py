"""
Retrieval Module - Vector Search and Knowledge Retrieval Utilities

This module provides the retrieval infrastructure for the Princeps system:
- Vector search with multiple backends (pgvector, ChromaDB, in-memory)
- Embedding generation using sentence-transformers
- Similarity computation and distance metrics
- Caching for embeddings and search results

Usage:
    from retrieval import (
        embed_text,
        query_vector_index,
        create_in_memory_index,
        VectorSearchConfig,
    )

    # Embed text
    embedding = await embed_text("How do I implement auth?")

    # Create and search an index
    index = create_in_memory_index()
    await index.add("doc1", embedding, "Document content", {"source": "docs"})
    results = await query_vector_index(embedding, index, top_k=5)
"""

from framework.retrieval.vector_search import (
    ChromaDBIndex,
    DistanceMetric,
    EmbeddingCache,
    EmbeddingModel,
    # Embedding
    EmbeddingService,
    InMemoryVectorIndex,
    PgVectorIndex,
    SearchFilter,
    # Enums
    VectorBackend,
    # Index classes
    VectorIndex,
    # Configuration
    VectorSearchConfig,
    # Data classes
    VectorSearchResult,
    batch_embed_texts,
    compute_similarity,
    # Similarity functions
    cosine_similarity,
    create_chromadb_index,
    # Factory functions
    create_in_memory_index,
    create_pgvector_index,
    dot_product,
    embed_text,
    euclidean_distance,
    get_embedding_service,
    hybrid_search,
    manhattan_distance,
    # Query functions
    query_vector_index,
)

__all__ = [
    # Enums
    "VectorBackend",
    "DistanceMetric",
    "EmbeddingModel",
    # Configuration
    "VectorSearchConfig",
    # Data classes
    "VectorSearchResult",
    "SearchFilter",
    # Embedding
    "EmbeddingService",
    "EmbeddingCache",
    "get_embedding_service",
    "embed_text",
    "batch_embed_texts",
    # Similarity functions
    "cosine_similarity",
    "euclidean_distance",
    "dot_product",
    "manhattan_distance",
    "compute_similarity",
    # Index classes
    "VectorIndex",
    "InMemoryVectorIndex",
    "PgVectorIndex",
    "ChromaDBIndex",
    # Query functions
    "query_vector_index",
    "hybrid_search",
    # Factory functions
    "create_in_memory_index",
    "create_pgvector_index",
    "create_chromadb_index",
]
