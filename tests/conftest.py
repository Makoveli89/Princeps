"""Pytest configuration and fixtures for Princeps Brain Layer tests."""
import os
import sys
import json
import pytest
from datetime import datetime
from uuid import uuid4

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, event, Text, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy import TypeDecorator

# Define SQLite-compatible type adapters BEFORE importing models
# These will be used to create a separate Base for testing

class SQLiteJSONB(TypeDecorator):
    """SQLite-compatible JSONB (stores as JSON text)."""
    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value)
        return None

    def process_result_value(self, value, dialect):
        if value is not None:
            return json.loads(value)
        return None


class SQLiteArray(TypeDecorator):
    """SQLite-compatible ARRAY (stores as JSON list)."""
    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value)
        return None

    def process_result_value(self, value, dialect):
        if value is not None:
            return json.loads(value)
        return None


class SQLiteUUID(TypeDecorator):
    """SQLite-compatible UUID (stores as string)."""
    impl = String(36)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return str(value)
        return None

    def process_result_value(self, value, dialect):
        if value is not None:
            from uuid import UUID as UUIDType
            if isinstance(value, UUIDType):
                return value
            return UUIDType(value)
        return None


# Monkey-patch postgresql dialect types with our SQLite-compatible versions
# This must happen BEFORE models.py is imported
import sqlalchemy.dialects.postgresql as pg_dialect

# Store originals
_orig_JSONB = pg_dialect.JSONB
_orig_ARRAY = pg_dialect.ARRAY
_orig_UUID = pg_dialect.UUID

# Replace with SQLite-compatible versions
pg_dialect.JSONB = SQLiteJSONB
pg_dialect.ARRAY = lambda *args, **kwargs: SQLiteArray()
pg_dialect.UUID = lambda *args, **kwargs: SQLiteUUID()

# Also patch the vector type if present
try:
    import pgvector.sqlalchemy as pgv
    # Replace Vector with a Text column for SQLite
    class SQLiteVector(TypeDecorator):
        impl = Text
        cache_ok = True

        def __init__(self, dim=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.dim = dim

        def process_bind_param(self, value, dialect):
            if value is not None:
                return json.dumps(value)
            return None

        def process_result_value(self, value, dialect):
            if value is not None:
                return json.loads(value)
            return None

    pgv.Vector = SQLiteVector
except ImportError:
    pass

# Now import models (after patching)
from brain.core.models import Base, Tenant, Document, DocChunk, Operation, OperationTypeEnum, OperationStatusEnum


@pytest.fixture(scope="session")
def engine():
    """Create an in-memory SQLite engine for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)

    # Enable foreign key support in SQLite
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    return engine


@pytest.fixture(scope="function")
def session(engine):
    """Create a new database session for each test."""
    connection = engine.connect()
    transaction = connection.begin()
    Session = sessionmaker(bind=connection)
    session = Session()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def tenant(session):
    """Create a test tenant."""
    t = Tenant(
        id=uuid4(),
        name="test_tenant",
        description="Test tenant for unit tests",
    )
    session.add(t)
    session.commit()
    return t


@pytest.fixture
def sample_document(session, tenant):
    """Create a sample document for testing."""
    from brain.core.db import compute_content_hash

    content = "This is sample document content for testing purposes. It contains multiple sentences for chunking."
    doc = Document(
        id=uuid4(),
        tenant_id=tenant.id,
        title="Test Document",
        content=content,
        content_hash=compute_content_hash(content),
        source="test",
    )
    session.add(doc)
    session.commit()
    return doc


@pytest.fixture
def sample_chunks(session, tenant, sample_document):
    """Create sample document chunks."""
    chunks = []
    for i, text in enumerate(["Chunk one content.", "Chunk two content.", "Chunk three content."]):
        chunk = DocChunk(
            id=uuid4(),
            tenant_id=tenant.id,
            document_id=sample_document.id,
            content=text,
            chunk_index=i,
            token_count=len(text.split()),
        )
        session.add(chunk)
        chunks.append(chunk)
    session.commit()
    return chunks
