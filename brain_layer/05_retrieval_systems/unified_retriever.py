"""
Unified Retrieval System for Princeps Brain Layer
==================================================

Provides a single interface for document retrieval with multiple backends:
1. pgvector (primary) - Postgres-native vector similarity search
2. ChromaDB - Chroma vector store with SBERT embeddings
3. TF-IDF - Lightweight keyword-based fallback
4. Heuristic - Token overlap as last resort

Integrates with SQLAlchemy models from 04_data_models/models.py.

Usage:
    from unified_retriever import UnifiedRetriever
    
    # With Postgres/pgvector
    retriever = UnifiedRetriever(session=db_session, backend="pgvector")
    
    # With Chroma fallback
    retriever = UnifiedRetriever(backend="chroma", persist_dir=".chroma")
    
    # Query
    results = retriever.search("machine learning pipeline", top_k=5)
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Backend availability flags
PGVECTOR_AVAILABLE = False
CHROMA_AVAILABLE = False
SBERT_AVAILABLE = False

try:
    from pgvector.sqlalchemy import Vector as _Vector  # noqa: F401
    PGVECTOR_AVAILABLE = True
except ImportError:
    pass

try:
    import chromadb as _chromadb  # noqa: F401
    CHROMA_AVAILABLE = True
except ImportError:
    pass

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    pass


@dataclass
class RetrievalResult:
    """Unified result format across all backends."""
    id: str
    content: str
    score: float
    source: str
    metadata: Dict[str, Any]
    backend: str  # Which backend produced this result
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "source": self.source,
            "metadata": self.metadata,
            "backend": self.backend,
        }


class BaseRetrieverBackend:
    """Abstract base for retrieval backends."""
    
    name: str = "base"
    
    def search(self, query: str, top_k: int = 5, **filters) -> List[RetrievalResult]:
        raise NotImplementedError
    
    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None) -> bool:
        raise NotImplementedError
    
    def is_available(self) -> bool:
        return True


class PgvectorBackend(BaseRetrieverBackend):
    """PostgreSQL with pgvector extension for native vector search."""
    
    name = "pgvector"
    
    def __init__(self, session, model_name: str = "all-MiniLM-L6-v2"):
        if not PGVECTOR_AVAILABLE:
            raise RuntimeError("pgvector not installed. Run: pip install pgvector")
        if not SBERT_AVAILABLE:
            raise RuntimeError("sentence-transformers not installed")
        
        self.session = session
        self.model = SentenceTransformer(model_name)
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        
    def _embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        return self.model.encode([text], show_progress_bar=False)[0].tolist()
    
    def search(self, query: str, top_k: int = 5, **filters) -> List[RetrievalResult]:
        """Search using pgvector cosine similarity."""
        from sqlalchemy import text
        
        query_embedding = self._embed(query)
        
        # Build filter conditions
        where_clauses = []
        params = {"embedding": str(query_embedding), "limit": top_k}
        
        if filters.get("source"):
            where_clauses.append("d.source = :source")
            params["source"] = filters["source"]
        
        if filters.get("category"):
            where_clauses.append("d.category = :category")
            params["category"] = filters["category"]
        
        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        # Query chunks with vector similarity
        sql = text(f"""
            SELECT 
                c.id,
                c.content,
                c.document_id,
                d.title,
                d.source,
                d.metadata,
                1 - (c.embedding <=> :embedding::vector) as score
            FROM doc_chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.embedding IS NOT NULL AND {where_sql}
            ORDER BY c.embedding <=> :embedding::vector
            LIMIT :limit
        """)
        
        results = []
        try:
            rows = self.session.execute(sql, params).fetchall()
            for row in rows:
                results.append(RetrievalResult(
                    id=row.id,
                    content=row.content,
                    score=float(row.score),
                    source=row.source,
                    metadata=row.metadata or {},
                    backend=self.name,
                ))
        except Exception as e:
            logger.error(f"pgvector search failed: {e}")
        
        return results
    
    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None) -> bool:
        """Add document chunk with embedding."""
        from sqlalchemy import text
        
        embedding = self._embed(content)
        
        sql = text("""
            INSERT INTO doc_chunks (id, document_id, content, embedding, metadata, created_at)
            VALUES (:id, :doc_id, :content, :embedding::vector, :metadata, :created_at)
            ON CONFLICT (id) DO UPDATE SET
                content = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata
        """)
        
        try:
            self.session.execute(sql, {
                "id": f"{doc_id}_chunk_0",
                "doc_id": doc_id,
                "content": content[:50000],  # Truncate very long content
                "embedding": str(embedding),
                "metadata": metadata or {},
                "created_at": datetime.utcnow(),
            })
            self.session.commit()
            return True
        except Exception as e:
            logger.error(f"pgvector add failed: {e}")
            self.session.rollback()
            return False
    
    def is_available(self) -> bool:
        return PGVECTOR_AVAILABLE and SBERT_AVAILABLE


class ChromaBackend(BaseRetrieverBackend):
    """ChromaDB vector store backend."""
    
    name = "chroma"
    
    def __init__(self, persist_dir: str = ".chroma", collection_name: str = "brain_layer"):
        if not CHROMA_AVAILABLE:
            raise RuntimeError("chromadb not installed. Run: pip install chromadb")
        
        # Import local vector_store module
        from .vector_store import ChromaVectorStore
        
        self.store = ChromaVectorStore(
            persist_dir=persist_dir,
            collection_name=collection_name,
            prefer_sbert=True,
        )
    
    def search(self, query: str, top_k: int = 5, **filters) -> List[RetrievalResult]:
        """Search using Chroma vector store."""
        results = []
        try:
            chroma_results = self.store.query(query, top_k=top_k)
            for r in chroma_results:
                results.append(RetrievalResult(
                    id=r.id,
                    content=r.document,
                    score=r.score,
                    source=r.metadata.get("source", "unknown"),
                    metadata=r.metadata,
                    backend=self.name,
                ))
        except Exception as e:
            logger.error(f"Chroma search failed: {e}")
        
        return results
    
    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None) -> bool:
        """Add document to Chroma store."""
        return self.store.upsert_learning(
            learning_id=doc_id,
            task=metadata.get("title", "Document"),
            feedback=content,
            solution=None,
            metadata=metadata,
        )
    
    def is_available(self) -> bool:
        return self.store.is_available()


class TfidfBackend(BaseRetrieverBackend):
    """TF-IDF based retrieval (no external deps)."""
    
    name = "tfidf"
    
    def __init__(self, storage_dir: str = "./tfidf_index"):
        from .tfidf_librarian import TfidfLibrarian
        
        self.librarian = TfidfLibrarian(
            project_root=".",
            storage_dir=storage_dir,
        )
    
    def search(self, query: str, top_k: int = 5, **filters) -> List[RetrievalResult]:
        """Search using TF-IDF similarity."""
        results = []
        try:
            tfidf_results = self.librarian.query(query, top_k=top_k)
            for r in tfidf_results:
                results.append(RetrievalResult(
                    id=r["id"],
                    content=r["text"],
                    score=r["score"],
                    source=r["source"],
                    metadata={"start": r["start"]},
                    backend=self.name,
                ))
        except Exception as e:
            logger.error(f"TF-IDF search failed: {e}")
        
        return results
    
    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None) -> bool:
        """Add document to TF-IDF index."""
        try:
            result = self.librarian.ingest_texts([{
                "id": doc_id,
                "text": content,
            }])
            return result.get("chunks_added", 0) > 0
        except Exception as e:
            logger.error(f"TF-IDF add failed: {e}")
            return False
    
    def is_available(self) -> bool:
        return True  # No external deps


class HeuristicBackend(BaseRetrieverBackend):
    """Simple token overlap retrieval (last resort)."""
    
    name = "heuristic"
    
    def __init__(self):
        from .embedding_retriever import HeuristicRetriever
        
        self.retriever = HeuristicRetriever()
        self.documents: List[Dict[str, Any]] = []
    
    def search(self, query: str, top_k: int = 5, **filters) -> List[RetrievalResult]:
        """Search using Jaccard similarity."""
        results = []
        try:
            heuristic_results = self.retriever.retrieve(query, self.documents, top_k=top_k)
            for r in heuristic_results:
                score = self.retriever.score(query, r.get("text", ""))
                results.append(RetrievalResult(
                    id=r.get("id", "unknown"),
                    content=r.get("text", ""),
                    score=score,
                    source=r.get("source", "unknown"),
                    metadata=r.get("metadata", {}),
                    backend=self.name,
                ))
        except Exception as e:
            logger.error(f"Heuristic search failed: {e}")
        
        return results
    
    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None) -> bool:
        """Add document to in-memory store."""
        self.documents.append({
            "id": doc_id,
            "text": content,
            "source": metadata.get("source", "unknown") if metadata else "unknown",
            "metadata": metadata or {},
        })
        return True
    
    def is_available(self) -> bool:
        return True  # Always available


class UnifiedRetriever:
    """
    Unified retrieval interface with automatic fallback.
    
    Backend priority (configurable):
    1. pgvector - Best quality, requires Postgres + pgvector
    2. chroma - Good quality, requires chromadb
    3. tfidf - Lightweight, no deps
    4. heuristic - Last resort, token overlap
    """
    
    def __init__(
        self,
        backend: str = "auto",
        session = None,  # SQLAlchemy session for pgvector
        persist_dir: str = ".chroma",
        tfidf_dir: str = "./tfidf_index",
        fallback_chain: List[str] = None,
    ):
        """
        Initialize unified retriever.
        
        Args:
            backend: Preferred backend ("auto", "pgvector", "chroma", "tfidf", "heuristic")
            session: SQLAlchemy session (required for pgvector)
            persist_dir: Directory for Chroma persistence
            tfidf_dir: Directory for TF-IDF index
            fallback_chain: Custom fallback order
        """
        self.fallback_chain = fallback_chain or ["pgvector", "chroma", "tfidf", "heuristic"]
        self.backends: Dict[str, BaseRetrieverBackend] = {}
        self.active_backend: Optional[BaseRetrieverBackend] = None
        
        # Initialize available backends
        self._init_backends(session, persist_dir, tfidf_dir)
        
        # Select backend
        if backend == "auto":
            self._select_best_backend()
        else:
            self._select_backend(backend)
    
    def _init_backends(self, session, persist_dir: str, tfidf_dir: str):
        """Initialize all available backends."""
        
        # pgvector
        if session and PGVECTOR_AVAILABLE and SBERT_AVAILABLE:
            try:
                self.backends["pgvector"] = PgvectorBackend(session)
                logger.info("pgvector backend initialized")
            except Exception as e:
                logger.warning(f"pgvector init failed: {e}")
        
        # Chroma
        if CHROMA_AVAILABLE:
            try:
                self.backends["chroma"] = ChromaBackend(persist_dir)
                logger.info("Chroma backend initialized")
            except Exception as e:
                logger.warning(f"Chroma init failed: {e}")
        
        # TF-IDF (always available)
        try:
            self.backends["tfidf"] = TfidfBackend(tfidf_dir)
            logger.info("TF-IDF backend initialized")
        except Exception as e:
            logger.warning(f"TF-IDF init failed: {e}")
        
        # Heuristic (always available)
        self.backends["heuristic"] = HeuristicBackend()
        logger.info("Heuristic backend initialized")
    
    def _select_best_backend(self):
        """Select the best available backend from the fallback chain."""
        for name in self.fallback_chain:
            if name in self.backends and self.backends[name].is_available():
                self.active_backend = self.backends[name]
                logger.info(f"Selected backend: {name}")
                return
        
        # Should never happen since heuristic is always available
        raise RuntimeError("No retrieval backend available")
    
    def _select_backend(self, name: str):
        """Select a specific backend."""
        if name not in self.backends:
            raise ValueError(f"Backend '{name}' not available. Options: {list(self.backends.keys())}")
        
        self.active_backend = self.backends[name]
        logger.info(f"Selected backend: {name}")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        with_fallback: bool = True,
        min_results: int = 1,
        **filters,
    ) -> List[RetrievalResult]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Maximum results to return
            with_fallback: Try fallback backends if primary fails
            min_results: Minimum results before trying fallback
            **filters: Backend-specific filters (source, category, etc.)
        
        Returns:
            List of RetrievalResult objects
        """
        results = []
        
        # Try primary backend
        if self.active_backend:
            try:
                results = self.active_backend.search(query, top_k=top_k, **filters)
            except Exception as e:
                logger.warning(f"Primary backend failed: {e}")
        
        # Fallback if needed
        if with_fallback and len(results) < min_results:
            for name in self.fallback_chain:
                if name in self.backends and self.backends[name] != self.active_backend:
                    try:
                        fallback_results = self.backends[name].search(query, top_k=top_k, **filters)
                        results.extend(fallback_results)
                        if len(results) >= min_results:
                            break
                    except Exception as e:
                        logger.warning(f"Fallback {name} failed: {e}")
        
        # Deduplicate and sort by score
        seen_ids = set()
        unique_results = []
        for r in sorted(results, key=lambda x: x.score, reverse=True):
            if r.id not in seen_ids:
                seen_ids.add(r.id)
                unique_results.append(r)
        
        return unique_results[:top_k]
    
    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Dict[str, Any] = None,
        to_all: bool = False,
    ) -> Dict[str, bool]:
        """
        Add document to retrieval index.
        
        Args:
            doc_id: Document identifier
            content: Document content
            metadata: Additional metadata
            to_all: Add to all backends (for redundancy)
        
        Returns:
            Dict of backend_name -> success status
        """
        results = {}
        
        if to_all:
            for name, backend in self.backends.items():
                results[name] = backend.add_document(doc_id, content, metadata)
        else:
            if self.active_backend:
                results[self.active_backend.name] = self.active_backend.add_document(
                    doc_id, content, metadata
                )
        
        return results
    
    def get_backend_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all backends."""
        status = {}
        for name, backend in self.backends.items():
            status[name] = {
                "available": backend.is_available(),
                "active": backend == self.active_backend,
            }
        return status
    
    @property
    def active_backend_name(self) -> str:
        """Get name of active backend."""
        return self.active_backend.name if self.active_backend else "none"


# Convenience function
def create_retriever(
    session = None,
    backend: str = "auto",
    **kwargs,
) -> UnifiedRetriever:
    """
    Create a unified retriever with sensible defaults.
    
    Args:
        session: SQLAlchemy session (enables pgvector)
        backend: Preferred backend
        **kwargs: Additional options
    
    Returns:
        Configured UnifiedRetriever instance
    """
    return UnifiedRetriever(
        session=session,
        backend=backend,
        **kwargs,
    )
