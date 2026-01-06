"""
Princeps Brain Layer - Database Utilities
==========================================

Section 1 Deliverable: ORM Integration

Database connection management, session handling, and utility functions
for working with the brain layer PostgreSQL database.

Usage:
    from db import get_engine, get_session, init_db

    # Initialize database
    engine = get_engine()
    init_db(engine)

    # Use session
    with get_session() as session:
        repo = Repository(name="test", url="https://...")
        session.add(repo)
        session.commit()
"""

import hashlib
import json
import os
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

# Import models
from .models import Base, Operation, OperationStatusEnum, OperationTypeEnum

# =============================================================================
# CONFIGURATION
# =============================================================================


def get_database_url() -> str:
    """
    Get database URL from environment or default.

    Environment variables (in order of precedence):
    - DATABASE_URL: Full connection string
    - POSTGRES_* variables: Individual connection parameters
    """
    if url := os.getenv("DATABASE_URL"):
        return url

    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "")
    database = os.getenv("POSTGRES_DB", "princeps_brain")

    if password:
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"
    return f"postgresql://{user}@{host}:{port}/{database}"


# =============================================================================
# ENGINE AND SESSION MANAGEMENT
# =============================================================================

_engine: Engine | None = None
_SessionFactory: sessionmaker | None = None


def get_engine(
    url: str | None = None,
    pool_size: int = 5,
    max_overflow: int = 10,
    echo: bool = False,
) -> Engine:
    """
    Get or create the SQLAlchemy engine.

    Args:
        url: Database URL (uses get_database_url() if not provided)
        pool_size: Number of connections in the pool
        max_overflow: Max connections above pool_size
        echo: Enable SQL logging

    Returns:
        SQLAlchemy Engine instance
    """
    global _engine

    if _engine is None:
        db_url = url or get_database_url()
        is_sqlite = db_url.startswith("sqlite")

        engine_kwargs = {"echo": echo}

        if is_sqlite:
            # SQLite-specific settings
            engine_kwargs["connect_args"] = {"check_same_thread": False}
        else:
            # PostgreSQL-specific settings
            engine_kwargs["poolclass"] = QueuePool
            engine_kwargs["pool_size"] = pool_size
            engine_kwargs["max_overflow"] = max_overflow
            engine_kwargs["pool_pre_ping"] = True
            engine_kwargs["connect_args"] = {"prepare_threshold": None}

        _engine = create_engine(db_url, **engine_kwargs)

    return _engine


def get_session_factory(engine: Engine | None = None) -> sessionmaker:
    """Get or create session factory."""
    global _SessionFactory

    if _SessionFactory is None:
        _SessionFactory = sessionmaker(
            bind=engine or get_engine(),
            autocommit=False,
            autoflush=False,
        )

    return _SessionFactory


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.

    Usage:
        with get_session() as session:
            results = session.query(Document).all()
    """
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# =============================================================================
# DATABASE INITIALIZATION
# =============================================================================


def init_db(engine: Engine | None = None) -> None:
    """
    Initialize the database schema.

    Creates all tables and enables required extensions.
    For production, use Alembic migrations instead.
    """
    engine = engine or get_engine()

    # Check if using PostgreSQL (extensions only available there)
    is_postgres = str(engine.url).startswith("postgresql")

    if is_postgres:
        # Enable PostgreSQL extensions
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
            conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
            conn.commit()

    # Create all tables
    Base.metadata.create_all(engine)


def drop_all_tables(engine: Engine | None = None) -> None:
    """
    Drop all tables. USE WITH CAUTION!

    Only for development/testing.
    """
    engine = engine or get_engine()
    Base.metadata.drop_all(engine)


# =============================================================================
# IDEMPOTENCY HELPERS
# =============================================================================


def compute_input_hash(op_type: str, inputs: dict[str, Any]) -> str:
    """
    Compute deterministic hash of operation inputs for idempotency.

    This is a simple implementation for basic use cases. For advanced
    features like path normalization and field exclusion, use
    brain.resilience.idempotency_service.compute_input_hash instead.

    Args:
        op_type: The operation type string
        inputs: Dictionary of input parameters

    Returns:
        SHA-256 hash (64 char hex string)
    """
    # Sort keys for deterministic ordering
    normalized = json.dumps({"op_type": op_type, "inputs": inputs}, sort_keys=True, default=str)
    return hashlib.sha256(normalized.encode()).hexdigest()


def get_or_create_operation(
    session: Session, tenant_id: str, op_type: OperationTypeEnum, inputs: dict[str, Any], **kwargs
) -> tuple[Operation, bool]:
    """
    Get existing operation or create new one (idempotency pattern).

    Args:
        session: Database session
        tenant_id: Tenant UUID
        op_type: Operation type enum
        inputs: Operation input parameters
        **kwargs: Additional operation fields

    Returns:
        Tuple of (Operation, created) where created is True if new
    """
    input_hash = compute_input_hash(op_type.value, inputs)

    # Try to find existing operation
    existing = (
        session.query(Operation)
        .filter(
            Operation.op_type == op_type,
            Operation.input_hash == input_hash,
        )
        .first()
    )

    if existing:
        return existing, False

    # Create new operation
    operation = Operation(
        tenant_id=tenant_id,
        op_type=op_type,
        input_hash=input_hash,
        inputs=inputs,
        status=OperationStatusEnum.PENDING,
        **kwargs,
    )
    session.add(operation)
    session.flush()  # Get the ID

    return operation, True


def mark_operation_started(session: Session, operation_id: str) -> None:
    """Mark an operation as in progress."""
    session.query(Operation).filter(Operation.id == operation_id).update(
        {
            "status": OperationStatusEnum.IN_PROGRESS,
            "started_at": datetime.utcnow(),
        }
    )


def mark_operation_success(
    session: Session, operation_id: str, outputs: dict | None = None
) -> None:
    """Mark an operation as successful."""
    now = datetime.utcnow()
    updates = {
        "status": OperationStatusEnum.SUCCESS,
        "completed_at": now,
        "outputs": outputs,
    }

    # Calculate duration if started_at exists
    op = session.query(Operation).filter(Operation.id == operation_id).first()
    if op and op.started_at:
        updates["duration_ms"] = int((now - op.started_at).total_seconds() * 1000)

    session.query(Operation).filter(Operation.id == operation_id).update(updates)


def mark_operation_failed(
    session: Session, operation_id: str, error_message: str, traceback: str | None = None
) -> None:
    """Mark an operation as failed."""
    now = datetime.utcnow()
    updates = {
        "status": OperationStatusEnum.FAILED,
        "completed_at": now,
        "error_message": error_message,
        "error_traceback": traceback,
    }

    op = session.query(Operation).filter(Operation.id == operation_id).first()
    if op:
        updates["retry_count"] = op.retry_count + 1
        if op.started_at:
            updates["duration_ms"] = int((now - op.started_at).total_seconds() * 1000)

    session.query(Operation).filter(Operation.id == operation_id).update(updates)


# =============================================================================
# TENANT HELPERS
# =============================================================================


def get_default_tenant_id(session: Session) -> str:
    """Get or create the default tenant ID."""
    from .models import Tenant

    tenant = session.query(Tenant).filter(Tenant.name == "default").first()
    if tenant:
        return str(tenant.id)

    # Create default tenant
    tenant = Tenant(name="default", description="Default tenant")
    session.add(tenant)
    session.flush()
    return str(tenant.id)


# =============================================================================
# CONTENT HASH HELPERS
# =============================================================================


def compute_content_hash(content: str) -> str:
    """Compute SHA-256 hash of content for deduplication."""
    return hashlib.sha256(content.encode()).hexdigest()


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file's contents."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


# =============================================================================
# VECTOR SEARCH HELPERS
# =============================================================================


def similarity_search_chunks(
    session: Session,
    query_embedding: list[float],
    tenant_id: str | None = None,
    document_id: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """
    Search for similar document chunks using pgvector.

    Args:
        session: Database session
        query_embedding: 384-dim embedding vector
        tenant_id: Optional tenant filter
        document_id: Optional document filter
        limit: Max results

    Returns:
        List of dicts with id, document_id, content, similarity
    """
    # Convert embedding to PostgreSQL array format
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    query = text("""
        SELECT id, document_id, content,
               1 - (embedding <=> :embedding::vector) as similarity
        FROM doc_chunks
        WHERE embedding IS NOT NULL
          AND (:tenant_id IS NULL OR tenant_id = :tenant_id::uuid)
          AND (:document_id IS NULL OR document_id = :document_id::uuid)
        ORDER BY embedding <=> :embedding::vector
        LIMIT :limit
    """)

    result = session.execute(
        query,
        {
            "embedding": embedding_str,
            "tenant_id": tenant_id,
            "document_id": document_id,
            "limit": limit,
        },
    )

    return [
        {
            "id": str(row.id),
            "document_id": str(row.document_id),
            "content": row.content,
            "similarity": row.similarity,
        }
        for row in result
    ]


# =============================================================================
# QUERY HELPERS
# =============================================================================


def list_repositories(
    session: Session,
    tenant_id: str,
    include_inactive: bool = False,
) -> list:
    """List all repositories for a tenant."""
    from .models import Repository

    query = session.query(Repository).filter(Repository.tenant_id == tenant_id)
    if not include_inactive:
        query = query.filter(Repository.is_active)
    return query.all()


def list_documents(
    session: Session,
    tenant_id: str,
    doc_type: str | None = None,
    source: str | None = None,
    limit: int = 100,
) -> list:
    """List documents with optional filters."""
    from .models import Document

    query = session.query(Document).filter(Document.tenant_id == tenant_id)
    if doc_type:
        query = query.filter(Document.doc_type == doc_type)
    if source:
        query = query.filter(Document.source == source)
    return query.limit(limit).all()


def list_operations(
    session: Session,
    tenant_id: str,
    status: OperationStatusEnum | None = None,
    op_type: OperationTypeEnum | None = None,
    limit: int = 100,
) -> list:
    """List operations with optional filters."""
    query = session.query(Operation).filter(Operation.tenant_id == tenant_id)
    if status:
        query = query.filter(Operation.status == status)
    if op_type:
        query = query.filter(Operation.op_type == op_type)
    return query.order_by(Operation.created_at.desc()).limit(limit).all()


# =============================================================================
# TESTING UTILITIES
# =============================================================================


def create_test_engine(echo: bool = False) -> Engine:
    """Create an in-memory SQLite engine for testing."""
    engine = create_engine("sqlite:///:memory:", echo=echo)
    Base.metadata.create_all(engine)
    return engine


def seed_test_data(session: Session) -> dict:
    """Seed database with test data. Returns dict of created objects."""
    from .models import DocChunk, Document, Repository, Tenant

    # Create tenant
    tenant = Tenant(name="test_tenant", description="Test tenant")
    session.add(tenant)
    session.flush()

    # Create repository
    repo = Repository(
        tenant_id=tenant.id,
        name="test-repo",
        url="https://github.com/test/repo",
    )
    session.add(repo)
    session.flush()

    # Create document
    doc = Document(
        tenant_id=tenant.id,
        title="Test Document",
        content="This is test content for the document.",
        content_hash=compute_content_hash("This is test content for the document."),
        source="test",
    )
    session.add(doc)
    session.flush()

    # Create chunk
    chunk = DocChunk(
        tenant_id=tenant.id,
        document_id=doc.id,
        content="This is test content",
        chunk_index=0,
        token_count=5,
    )
    session.add(chunk)
    session.flush()

    return {
        "tenant": tenant,
        "repository": repo,
        "document": doc,
        "chunk": chunk,
    }
