"""
Retrieval Systems for Princeps Brain Layer
===========================================

Multi-backend document retrieval with automatic fallback:

1. **pgvector** (primary) - PostgreSQL native vector search
2. **ChromaDB** - Standalone vector store with SBERT
3. **TF-IDF** - Lightweight keyword-based retrieval
4. **Heuristic** - Token overlap as last resort

Quick Start:
    from retrieval_systems import create_retriever
    
    # Auto-select best available backend
    retriever = create_retriever()
    
    # With Postgres session (enables pgvector)
    retriever = create_retriever(session=db_session)
    
    # Search
    results = retriever.search("machine learning", top_k=5)
    
    # Add document
    retriever.add_document("doc_1", "Content here...", {"source": "api"})

Modules:
    - vector_store: ChromaDB wrapper with SBERT/hashing embeddings
    - embedding_retriever: Multi-backend embeddings (SBERT, TF-IDF, heuristic)
    - tfidf_librarian: Lightweight TF-IDF index with persistence
    - unified_retriever: Single interface for all backends
"""

from .vector_store import (
    ChromaVectorStore,
    VectorResult,
    BaseEmbedding,
    SbertEmbedding,
    HashingEmbedding,
    build_embedding,
)

from .embedding_retriever import (
    EmbeddingRetriever,
    SimpleTfidfRetriever,
    HeuristicRetriever,
)

from .tfidf_librarian import (
    TfidfLibrarian,
    Chunk,
    chunk_text,
    tokenize,
)

from .unified_retriever import (
    UnifiedRetriever,
    RetrievalResult,
    create_retriever,
    PgvectorBackend,
    ChromaBackend,
    TfidfBackend,
    HeuristicBackend,
)

__all__ = [
    # Vector Store
    "ChromaVectorStore",
    "VectorResult",
    "BaseEmbedding",
    "SbertEmbedding",
    "HashingEmbedding",
    "build_embedding",
    # Embedding Retriever
    "EmbeddingRetriever",
    "SimpleTfidfRetriever",
    "HeuristicRetriever",
    # TF-IDF Librarian
    "TfidfLibrarian",
    "Chunk",
    "chunk_text",
    "tokenize",
    # Unified Retriever
    "UnifiedRetriever",
    "RetrievalResult",
    "create_retriever",
    "PgvectorBackend",
    "ChromaBackend",
    "TfidfBackend",
    "HeuristicBackend",
]
