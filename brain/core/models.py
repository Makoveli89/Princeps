"""
Princeps Brain Layer - Complete SQLAlchemy Data Models
=======================================================

Section 1 Deliverable: Core Data Schema & Migration

This module defines the complete PostgreSQL schema for the Brain's knowledge
storage system. All knowledge entities (repositories, resources, operations, etc.)
are stored with proper relationships establishing a single source of truth.

Tables Defined:
- Repository: Git repository metadata and tracking
- Resource: Files/assets within repositories or standalone
- Document: Primary document/knowledge entries
- DocChunk: Document chunks for embedding-based retrieval
- Operation: Idempotent operation tracking with input hashing
- Artifact: Generated outputs from operations
- Decision: Agent decision logging
- Analysis tables: Summaries, Entities, Topics, Concepts
- Knowledge Network: Nodes, Edges, Flows

Usage:
    from models import Base, Repository, Resource, Document, Operation
    from sqlalchemy import create_engine
    
    engine = create_engine("postgresql://user:pass@localhost/princeps")
    Base.metadata.create_all(engine)
"""

import enum
import hashlib
import json
from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

# pgvector support - install with: pip install pgvector sqlalchemy
try:
    from pgvector.sqlalchemy import Vector
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False
    # Create a placeholder that acts like a Column type but stores as Text
    # This allows code to run without pgvector, just without vector operations
    from sqlalchemy import Text as _TextFallback
    
    class _VectorFallback:
        """Fallback for Vector type when pgvector is not available."""
        def __init__(self, dim: int = 384):
            self.dim = dim
        
        def __call__(self, *args, **kwargs):
            return _TextFallback()
    
    Vector = _VectorFallback  # type: ignore

Base = declarative_base()


# =============================================================================
# ENUMS - Type classifications for various entities
# =============================================================================

class OperationTypeEnum(enum.Enum):
    """Types of operations for idempotency tracking."""
    INGEST_REPO = "ingest_repo"
    INGEST_DOCUMENT = "ingest_document"
    CHUNK_DOCUMENT = "chunk_document"
    GENERATE_EMBEDDING = "generate_embedding"
    EXTRACT_ENTITIES = "extract_entities"
    EXTRACT_TOPICS = "extract_topics"
    EXTRACT_CONCEPTS = "extract_concepts"
    GENERATE_SUMMARY = "generate_summary"
    BUILD_DEPENDENCY_GRAPH = "build_dependency_graph"
    AGENT_TASK = "agent_task"
    ANALYSIS = "analysis"
    RETRIEVAL = "retrieval"


class OperationStatusEnum(enum.Enum):
    """Status of an operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"  # For idempotent duplicate detection


class ResourceTypeEnum(enum.Enum):
    """Types of resources/files."""
    CODE_FILE = "code_file"
    DOCUMENT = "document"
    IMAGE = "image"
    CONFIG = "config"
    DATA = "data"
    BINARY = "binary"
    DIRECTORY = "directory"


class KnowledgeTypeEnum(enum.Enum):
    """Types of knowledge entries."""
    AGENT_OUTPUT = "agent_output"
    TASK_RESULT = "task_result"
    DATASET = "dataset"
    CODE_SNIPPET = "code_snippet"
    DOCUMENTATION = "documentation"
    ERROR_SOLUTION = "error_solution"
    BEST_PRACTICE = "best_practice"
    RESEARCH_PAPER = "research_paper"
    CONVERSATION = "conversation"
    EXPERIMENT = "experiment"


class SecurityLevelEnum(enum.Enum):
    """Security classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class NodeKnowledgeTypeEnum(enum.Enum):
    """Cross-agent knowledge types."""
    SOLUTION = "solution"
    PATTERN = "pattern"
    INSIGHT = "insight"
    BEST_PRACTICE = "best_practice"
    FAILURE = "failure"
    OPTIMIZATION = "optimization"


class ShareScopeEnum(enum.Enum):
    """Knowledge sharing scope."""
    PRIVATE = "private"
    TEAM = "team"
    PUBLIC = "public"


class ArtifactTypeEnum(enum.Enum):
    """Types of generated artifacts."""
    EMBEDDING_INDEX = "embedding_index"
    SUMMARY = "summary"
    ENTITY_LIST = "entity_list"
    DEPENDENCY_GRAPH = "dependency_graph"
    ANALYSIS_REPORT = "analysis_report"
    CODE_ANALYSIS = "code_analysis"
    MODEL_OUTPUT = "model_output"


class DecisionTypeEnum(enum.Enum):
    """Types of agent decisions."""
    TASK_SELECTION = "task_selection"
    TOOL_CHOICE = "tool_choice"
    ROUTING = "routing"
    PRIORITIZATION = "prioritization"
    ESCALATION = "escalation"
    RETRY = "retry"
    ABORT = "abort"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_uuid() -> str:
    """Generate a UUID string."""
    return str(uuid4())


def compute_input_hash(op_type: str, inputs: dict[str, Any]) -> str:
    """
    Compute deterministic hash of operation inputs for idempotency.
    
    Note: For advanced idempotency with path normalization and config options,
    prefer using brain.resilience.idempotency_service.compute_input_hash.
    
    Args:
        op_type: The operation type string
        inputs: Dictionary of input parameters
        
    Returns:
        SHA-256 hash of normalized inputs
    """
    # Sort keys for deterministic ordering
    normalized = json.dumps({
        "op_type": op_type,
        "inputs": inputs
    }, sort_keys=True, default=str)
    return hashlib.sha256(normalized.encode()).hexdigest()


# =============================================================================
# TENANT & MULTI-TENANCY SUPPORT
# =============================================================================

class Tenant(Base):
    """
    Tenant/project for multi-tenancy support.
    
    Every record in core tables includes a tenant_id for data isolation.
    """
    __tablename__ = "tenants"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=True)

    # Settings
    settings = Column(JSONB, default={})
    is_active = Column(Boolean, default=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<Tenant(id={self.id}, name='{self.name}')>"


# =============================================================================
# REPOSITORY - Git repository tracking
# =============================================================================

class Repository(Base):
    """
    Git repository metadata and tracking.
    
    Stores information about ingested repositories including
    commit history tracking for incremental updates.
    """
    __tablename__ = "repositories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)

    # Repository identification
    name = Column(String(255), nullable=False, index=True)
    url = Column(String(1000), nullable=False)
    clone_path = Column(String(1000), nullable=True, comment="Local clone path if applicable")

    # Git metadata
    default_branch = Column(String(100), default="main")
    last_commit_sha = Column(String(64), nullable=True, comment="Latest ingested commit")
    last_commit_date = Column(DateTime(timezone=True), nullable=True)
    last_commit_message = Column(Text, nullable=True)

    # Statistics
    file_count = Column(Integer, default=0)
    total_lines = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    languages = Column(ARRAY(String), default=[])

    # Status
    is_active = Column(Boolean, default=True)
    last_ingested_at = Column(DateTime(timezone=True), nullable=True)
    ingest_frequency = Column(String(50), default="manual", comment="manual, daily, weekly")

    # Metadata
    description = Column(Text, nullable=True)
    tags = Column(ARRAY(String), default=[])
    extra_data = Column("metadata", JSONB, default={})

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    tenant = relationship("Tenant", backref="repositories")
    resources = relationship("Resource", back_populates="repository", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("tenant_id", "url", name="uq_tenant_repo_url"),
        Index("idx_repo_name", "name"),
        Index("idx_repo_tenant", "tenant_id"),
    )

    def __repr__(self):
        return f"<Repository(id={self.id}, name='{self.name}', url='{self.url[:50]}...')>"


# =============================================================================
# RESOURCE - Files and assets
# =============================================================================

class Resource(Base):
    """
    Files/assets within repositories or standalone documents.
    
    Represents individual files with their metadata, content hashes,
    and relationships to repositories and documents.
    """
    __tablename__ = "resources"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)
    repository_id = Column(UUID(as_uuid=True), ForeignKey("repositories.id", ondelete="CASCADE"), nullable=True, index=True)

    # File identification
    file_path = Column(String(1000), nullable=False)
    file_name = Column(String(255), nullable=False, index=True)
    file_extension = Column(String(50), nullable=True, index=True)
    resource_type = Column(Enum(ResourceTypeEnum), nullable=False, default=ResourceTypeEnum.CODE_FILE)

    # Content tracking
    content_hash = Column(String(64), nullable=False, index=True, comment="SHA-256 of file content")
    size_bytes = Column(Integer, nullable=True)
    line_count = Column(Integer, nullable=True)
    token_count = Column(Integer, nullable=True)

    # Git metadata (if from repository)
    last_commit_sha = Column(String(64), nullable=True)
    last_modified_at = Column(DateTime(timezone=True), nullable=True)

    # Programming language (for code files)
    language = Column(String(50), nullable=True, index=True)

    # Security flags
    has_pii = Column(Boolean, default=False, index=True, comment="Contains PII/sensitive data")
    has_secrets = Column(Boolean, default=False, index=True, comment="Contains API keys/secrets")
    security_flags = Column(ARRAY(String), default=[], comment="Specific security concerns found")

    # Status
    is_active = Column(Boolean, default=True)
    is_parsed = Column(Boolean, default=False, comment="Whether content has been parsed/chunked")
    parse_error = Column(Text, nullable=True)

    # Metadata
    extra_data = Column("metadata", JSONB, default={})

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    tenant = relationship("Tenant", backref="resources")
    repository = relationship("Repository", back_populates="resources")
    documents = relationship("Document", back_populates="source_resource")
    dependencies = relationship(
        "ResourceDependency",
        foreign_keys="ResourceDependency.source_id",
        back_populates="source_resource",
        cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("tenant_id", "repository_id", "file_path", name="uq_tenant_repo_path"),
        UniqueConstraint("tenant_id", "content_hash", name="uq_tenant_content_hash"),
        Index("idx_resource_tenant", "tenant_id"),
        Index("idx_resource_repo", "repository_id"),
        Index("idx_resource_type", "resource_type"),
        Index("idx_resource_language", "language"),
    )

    def __repr__(self):
        return f"<Resource(id={self.id}, path='{self.file_path}', type={self.resource_type.value})>"


class ResourceDependency(Base):
    """
    Dependency relationships between resources (imports, includes, etc.).
    
    Captures the dependency graph for code analysis and impact assessment.
    """
    __tablename__ = "resource_dependencies"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)

    source_id = Column(UUID(as_uuid=True), ForeignKey("resources.id", ondelete="CASCADE"), nullable=False, index=True)
    target_id = Column(UUID(as_uuid=True), ForeignKey("resources.id", ondelete="CASCADE"), nullable=False, index=True)

    # Dependency details
    dependency_type = Column(String(50), nullable=False, comment="import, include, extends, uses")
    import_path = Column(String(500), nullable=True, comment="The actual import statement")
    line_number = Column(Integer, nullable=True)

    # Metadata
    extra_data = Column("metadata", JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    tenant = relationship("Tenant", backref="resource_dependencies")
    source_resource = relationship("Resource", foreign_keys=[source_id], back_populates="dependencies")
    target_resource = relationship("Resource", foreign_keys=[target_id])

    __table_args__ = (
        UniqueConstraint("source_id", "target_id", "dependency_type", name="uq_resource_dependency"),
        Index("idx_dep_source", "source_id"),
        Index("idx_dep_target", "target_id"),
    )

    def __repr__(self):
        return f"<ResourceDependency({self.source_id} --{self.dependency_type}--> {self.target_id})>"


# =============================================================================
# OPERATION - Idempotent operation tracking
# =============================================================================

class Operation(Base):
    """
    Idempotent operation tracking with input hashing.
    
    Every operation (ingests, analyses, retrievals) is recorded with a
    hash of its inputs. This enables:
    - Idempotency: Same inputs = skip re-execution
    - Audit trail: Full history of all operations
    - Retry logic: Failed operations can be retried
    - Performance: Track operation duration and resource usage
    
    The unique constraint on (op_type, input_hash) enforces idempotency
    at the database level.
    """
    __tablename__ = "operations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)

    # Operation identification
    op_type = Column(Enum(OperationTypeEnum), nullable=False, index=True)
    input_hash = Column(String(64), nullable=False, index=True, comment="SHA-256 of normalized inputs")

    # Input/Output
    inputs = Column(JSONB, nullable=False, comment="Full input parameters")
    outputs = Column(JSONB, nullable=True, comment="Operation results/outputs")

    # Status tracking
    status = Column(
        Enum(OperationStatusEnum),
        nullable=False,
        default=OperationStatusEnum.PENDING,
        index=True
    )
    error_message = Column(Text, nullable=True)
    error_traceback = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)

    # Timing
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    duration_ms = Column(Integer, nullable=True)

    # Resource references (optional, for context)
    repository_id = Column(UUID(as_uuid=True), ForeignKey("repositories.id", ondelete="SET NULL"), nullable=True)
    resource_id = Column(UUID(as_uuid=True), ForeignKey("resources.id", ondelete="SET NULL"), nullable=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="SET NULL"), nullable=True)

    # Correlation for tracing
    correlation_id = Column(String(64), nullable=True, index=True, comment="For tracing related operations")
    parent_operation_id = Column(UUID(as_uuid=True), ForeignKey("operations.id"), nullable=True)

    # Agent context
    agent_id = Column(String(100), nullable=True, index=True)

    # Metadata
    extra_data = Column("metadata", JSONB, default={})

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    tenant = relationship("Tenant", backref="operations")
    repository = relationship("Repository", backref="operations")
    resource = relationship("Resource", backref="operations")
    document = relationship("Document", back_populates="operations")
    parent_operation = relationship("Operation", remote_side=[id], backref="child_operations")
    artifacts = relationship("Artifact", back_populates="operation", cascade="all, delete-orphan")
    decisions = relationship("Decision", back_populates="operation", cascade="all, delete-orphan")

    __table_args__ = (
        # CRITICAL: This constraint enforces idempotency
        UniqueConstraint("op_type", "input_hash", name="uq_op_type_input_hash"),
        Index("idx_op_tenant", "tenant_id"),
        Index("idx_op_status", "status"),
        Index("idx_op_type_status", "op_type", "status"),
        Index("idx_op_correlation", "correlation_id"),
        Index("idx_op_created", "created_at"),
    )

    @classmethod
    def compute_hash(cls, op_type: OperationTypeEnum, inputs: dict[str, Any]) -> str:
        """Compute deterministic hash for idempotency check."""
        return compute_input_hash(op_type.value, inputs)

    def mark_started(self):
        """Mark operation as started."""
        self.status = OperationStatusEnum.IN_PROGRESS
        self.started_at = datetime.utcnow()

    def mark_success(self, outputs: dict | None = None):
        """Mark operation as successful."""
        self.status = OperationStatusEnum.SUCCESS
        self.completed_at = datetime.utcnow()
        if self.started_at:
            self.duration_ms = int((self.completed_at - self.started_at).total_seconds() * 1000)
        if outputs:
            self.outputs = outputs

    def mark_failed(self, error_message: str, traceback: str | None = None):
        """Mark operation as failed."""
        self.status = OperationStatusEnum.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error_message
        self.error_traceback = traceback
        if self.started_at:
            self.duration_ms = int((self.completed_at - self.started_at).total_seconds() * 1000)

    def can_retry(self) -> bool:
        """Check if operation can be retried."""
        return self.retry_count < self.max_retries and self.status == OperationStatusEnum.FAILED

    def __repr__(self):
        return f"<Operation(id={self.id}, type={self.op_type.value}, status={self.status.value})>"


# =============================================================================
# DOCUMENT & CHUNKS - Core content storage
# =============================================================================

class Document(Base):
    """
    Primary document/knowledge entry storage.
    
    Stores document metadata and full content. For large documents,
    content is chunked into DocChunk records for embedding search.
    """
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)

    # Source reference
    source_resource_id = Column(UUID(as_uuid=True), ForeignKey("resources.id", ondelete="SET NULL"), nullable=True, index=True)

    # Document identification
    title = Column(String(500), nullable=False, index=True)
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False, index=True, comment="SHA-256 for deduplication")

    # Classification
    doc_type = Column(Enum(KnowledgeTypeEnum), nullable=False, default=KnowledgeTypeEnum.DOCUMENTATION)
    source = Column(String(255), nullable=False, index=True, comment="Origin (agent, user, system)")
    category = Column(String(100), nullable=True, index=True)

    # Content metrics
    word_count = Column(Integer, nullable=True)
    token_count = Column(Integer, nullable=True)
    page_count = Column(Integer, nullable=True)

    # Security
    security_level = Column(Enum(SecurityLevelEnum), default=SecurityLevelEnum.INTERNAL)
    has_pii = Column(Boolean, default=False, index=True)

    # Tags and metadata
    tags = Column(ARRAY(String), default=[])
    extra_data = Column("metadata", JSONB, default={})

    # Versioning
    version = Column(Integer, default=1)
    parent_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True)

    # Processing status
    is_chunked = Column(Boolean, default=False)
    is_embedded = Column(Boolean, default=False)
    is_analyzed = Column(Boolean, default=False)

    # Usage tracking
    access_count = Column(Integer, default=0)
    last_accessed_at = Column(DateTime(timezone=True), nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    tenant = relationship("Tenant", backref="documents")
    source_resource = relationship("Resource", back_populates="documents")
    parent = relationship("Document", remote_side=[id], backref="versions")
    chunks = relationship("DocChunk", back_populates="document", cascade="all, delete-orphan")
    summaries = relationship("DocumentSummary", back_populates="document", cascade="all, delete-orphan")
    entities = relationship("DocumentEntity", back_populates="document", cascade="all, delete-orphan")
    topics = relationship("DocumentTopic", back_populates="document", cascade="all, delete-orphan")
    concepts = relationship("DocumentConcept", back_populates="document", cascade="all, delete-orphan")
    operations = relationship("Operation", back_populates="document")

    __table_args__ = (
        UniqueConstraint("tenant_id", "content_hash", name="uq_tenant_doc_hash"),
        Index("idx_doc_tenant", "tenant_id"),
        Index("idx_doc_type", "doc_type"),
        Index("idx_doc_source", "source"),
        Index("idx_doc_created", "created_at"),
        Index("idx_doc_tags", "tags", postgresql_using="gin"),
        Index("idx_doc_metadata", "metadata", postgresql_using="gin"),
    )

    def __repr__(self):
        return f"<Document(id={self.id}, title='{self.title[:30]}...', type={self.doc_type.value})>"


class DocChunk(Base):
    """
    Document chunks for embedding-based retrieval.
    
    Each chunk is ~800-1000 tokens with overlap for context preservation.
    Embeddings stored via pgvector for similarity search.
    """
    __tablename__ = "doc_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)

    # Content
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)

    # Position in original
    start_char = Column(Integer, nullable=True)
    end_char = Column(Integer, nullable=True)
    page_number = Column(Integer, nullable=True)

    # Token info
    token_count = Column(Integer, nullable=True)
    char_count = Column(Integer, nullable=True)

    # Embedding - pgvector column (384 dims for all-MiniLM-L6-v2, 1536 for OpenAI)
    if PGVECTOR_AVAILABLE:
        embedding = Column(Vector(384), nullable=True)
    else:
        # Fallback: store as JSONB array if pgvector not available
        embedding = Column(JSONB, nullable=True, comment="Embedding vector as JSON array")

    # Embedding metadata
    embedding_model = Column(String(100), nullable=True, comment="Model used for embedding")
    embedded_at = Column(DateTime(timezone=True), nullable=True)

    # Metadata
    extra_data = Column("metadata", JSONB, default={})

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    tenant = relationship("Tenant", backref="chunks")
    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_doc_chunk_index"),
        Index("idx_chunk_tenant", "tenant_id"),
        Index("idx_chunk_doc", "document_id"),
    )

    def __repr__(self):
        return f"<DocChunk(doc={self.document_id}, idx={self.chunk_index}, tokens={self.token_count})>"


# =============================================================================
# ARTIFACT - Generated outputs
# =============================================================================

class Artifact(Base):
    """
    Generated outputs from operations.
    
    Artifacts are the products of operations - embeddings, summaries,
    analysis reports, etc. They reference their source operation for
    provenance tracking.
    """
    __tablename__ = "artifacts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)
    operation_id = Column(UUID(as_uuid=True), ForeignKey("operations.id", ondelete="CASCADE"), nullable=False, index=True)

    # Artifact identification
    artifact_type = Column(Enum(ArtifactTypeEnum), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # Content (either inline or reference)
    content = Column(JSONB, nullable=True, comment="Artifact content if small enough")
    file_path = Column(String(1000), nullable=True, comment="Path to artifact file if stored externally")
    file_size_bytes = Column(Integer, nullable=True)
    content_hash = Column(String(64), nullable=True)

    # Source references
    source_document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="SET NULL"), nullable=True)
    source_resource_id = Column(UUID(as_uuid=True), ForeignKey("resources.id", ondelete="SET NULL"), nullable=True)

    # Quality metrics
    quality_score = Column(Float, nullable=True, comment="Quality assessment 0-1")
    confidence = Column(Float, nullable=True)

    # Usage
    usage_count = Column(Integer, default=0)
    last_used_at = Column(DateTime(timezone=True), nullable=True)

    # Metadata
    extra_data = Column("metadata", JSONB, default={})

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    tenant = relationship("Tenant", backref="artifacts")
    operation = relationship("Operation", back_populates="artifacts")
    source_document = relationship("Document", backref="artifacts")
    source_resource = relationship("Resource", backref="artifacts")

    __table_args__ = (
        Index("idx_artifact_tenant", "tenant_id"),
        Index("idx_artifact_operation", "operation_id"),
        Index("idx_artifact_type", "artifact_type"),
    )

    def __repr__(self):
        return f"<Artifact(id={self.id}, type={self.artifact_type.value}, name='{self.name}')>"


# =============================================================================
# DECISION - Agent decision logging
# =============================================================================

class Decision(Base):
    """
    Agent decision logging for observability and learning.
    
    Records every significant decision an agent makes, including
    the context, reasoning, and outcome. This enables:
    - Debugging agent behavior
    - Training data collection
    - Performance analysis
    """
    __tablename__ = "decisions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)
    operation_id = Column(UUID(as_uuid=True), ForeignKey("operations.id", ondelete="CASCADE"), nullable=True, index=True)

    # Decision identification
    agent_id = Column(String(100), nullable=False, index=True)
    decision_type = Column(Enum(DecisionTypeEnum), nullable=False, index=True)

    # Context
    context = Column(JSONB, nullable=False, comment="Input context for the decision")
    options_considered = Column(JSONB, nullable=True, comment="List of options evaluated")

    # Decision made
    decision = Column(JSONB, nullable=False, comment="The actual decision made")
    reasoning = Column(Text, nullable=True, comment="Explanation of why this decision")
    confidence = Column(Float, nullable=True, comment="Confidence in the decision 0-1")

    # Outcome (filled in later)
    outcome = Column(JSONB, nullable=True, comment="Result of the decision")
    outcome_score = Column(Float, nullable=True, comment="Quality of outcome 0-1")
    was_correct = Column(Boolean, nullable=True, comment="Whether decision was correct in hindsight")
    feedback = Column(Text, nullable=True)

    # Timing
    decided_at = Column(DateTime(timezone=True), server_default=func.now())
    outcome_recorded_at = Column(DateTime(timezone=True), nullable=True)
    decision_latency_ms = Column(Integer, nullable=True, comment="Time to make decision")

    # Metadata
    extra_data = Column("metadata", JSONB, default={})

    # Relationships
    tenant = relationship("Tenant", backref="decisions")
    operation = relationship("Operation", back_populates="decisions")

    __table_args__ = (
        Index("idx_decision_tenant", "tenant_id"),
        Index("idx_decision_agent", "agent_id"),
        Index("idx_decision_type", "decision_type"),
        Index("idx_decision_operation", "operation_id"),
        Index("idx_decision_time", "decided_at"),
    )

    def __repr__(self):
        return f"<Decision(id={self.id}, agent={self.agent_id}, type={self.decision_type.value})>"


# =============================================================================
# ANALYSIS TABLES - Knowledge atoms from distillation
# =============================================================================

class DocumentSummary(Base):
    """Generated summaries for documents."""
    __tablename__ = "document_summaries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)

    # Summary content
    one_sentence = Column(Text, nullable=True)
    executive = Column(Text, nullable=True)
    detailed = Column(Text, nullable=True)

    # Generation metadata
    model_used = Column(String(100), nullable=True)
    generation_params = Column(JSONB, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    tenant = relationship("Tenant", backref="document_summaries")
    document = relationship("Document", back_populates="summaries")

    __table_args__ = (
        UniqueConstraint("document_id", name="uq_doc_summary"),
        Index("idx_summary_tenant", "tenant_id"),
    )


class DocumentEntity(Base):
    """Named entities extracted from documents."""
    __tablename__ = "document_entities"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)

    # Entity details
    text = Column(String(500), nullable=False, index=True)
    label = Column(String(50), nullable=False, index=True, comment="PERSON, ORG, GPE, etc.")

    # Position
    start_char = Column(Integer, nullable=True)
    end_char = Column(Integer, nullable=True)

    # Metrics
    confidence = Column(Float, nullable=True)
    frequency = Column(Integer, default=1)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    tenant = relationship("Tenant", backref="document_entities")
    document = relationship("Document", back_populates="entities")

    __table_args__ = (
        Index("idx_entity_tenant", "tenant_id"),
        Index("idx_entity_doc", "document_id"),
        Index("idx_entity_label_text", "label", "text"),
    )


class DocumentTopic(Base):
    """Topics identified in documents."""
    __tablename__ = "document_topics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)

    # Topic details
    topic_id = Column(Integer, nullable=False, comment="Topic cluster ID from BERTopic")
    name = Column(String(255), nullable=True)
    keywords = Column(ARRAY(String), default=[])

    # Confidence
    probability = Column(Float, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    tenant = relationship("Tenant", backref="document_topics")
    document = relationship("Document", back_populates="topics")

    __table_args__ = (
        Index("idx_topic_tenant", "tenant_id"),
        Index("idx_topic_doc", "document_id"),
        Index("idx_topic_id", "topic_id"),
    )


class DocumentConcept(Base):
    """Key concepts extracted from documents."""
    __tablename__ = "document_concepts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)

    # Concept details
    concept = Column(String(255), nullable=False, index=True)
    relevance = Column(Float, nullable=False)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    tenant = relationship("Tenant", backref="document_concepts")
    document = relationship("Document", back_populates="concepts")

    __table_args__ = (
        Index("idx_concept_tenant", "tenant_id"),
        Index("idx_concept_doc", "document_id"),
        Index("idx_concept_relevance", "relevance"),
    )


# =============================================================================
# AGENT RUNS - Activity tracing
# =============================================================================

class AgentRun(Base):
    """Log of agent task executions and outcomes."""
    __tablename__ = "agent_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)

    # Task info
    agent_id = Column(String(100), nullable=False, index=True)
    task = Column(Text, nullable=False)
    task_hash = Column(String(64), nullable=True, index=True)
    iteration = Column(Integer, default=1)

    # Outcome
    success = Column(Boolean, nullable=False, index=True)
    score = Column(Float, nullable=True)
    solution = Column(JSONB, nullable=True)
    feedback = Column(Text, nullable=True)

    # Timing
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    duration_ms = Column(Integer, nullable=True)

    # Context
    context = Column(JSONB, nullable=True)
    tools_used = Column(ARRAY(String), default=[])
    model_version = Column(String(50), nullable=True)

    # Usage tracking
    relevance_count = Column(Integer, default=0)
    last_accessed_at = Column(DateTime(timezone=True), nullable=True)

    # Consolidation
    consolidated = Column(Boolean, default=False)
    parent_id = Column(UUID(as_uuid=True), ForeignKey("agent_runs.id"), nullable=True)

    # Metadata
    extra_data = Column("metadata", JSONB, default={})

    # Relationships
    tenant = relationship("Tenant", backref="agent_runs")
    parent = relationship("AgentRun", remote_side=[id], backref="children")

    __table_args__ = (
        Index("idx_run_tenant", "tenant_id"),
        Index("idx_run_agent_success", "agent_id", "success"),
        Index("idx_run_task_hash", "task_hash"),
        Index("idx_run_started", "started_at"),
    )


# =============================================================================
# KNOWLEDGE NETWORK - Cross-agent intelligence
# =============================================================================

class KnowledgeNode(Base):
    """Cross-agent knowledge nodes for shared intelligence."""
    __tablename__ = "knowledge_nodes"

    node_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)

    agent_id = Column(String(100), nullable=False, index=True)

    # Classification
    knowledge_type = Column(Enum(NodeKnowledgeTypeEnum), nullable=False)
    scope = Column(Enum(ShareScopeEnum), default=ShareScopeEnum.PUBLIC)

    # Content
    title = Column(String(500), nullable=False, index=True)
    content = Column(Text, nullable=False)
    tags = Column(ARRAY(String), default=[])
    problem_domain = Column(String(100), nullable=True, index=True)

    # Versioning
    version = Column(Integer, default=1)

    # Usage metrics
    access_count = Column(Integer, default=0)
    success_rate = Column(Float, default=0.0)
    relevance_score = Column(Float, default=1.0)

    # Sharing
    shared_with = Column(ARRAY(String), default=[])

    # Context
    prerequisites = Column(ARRAY(String), default=[])
    related_nodes = Column(ARRAY(String), default=[])

    # Embedding
    if PGVECTOR_AVAILABLE:
        embedding = Column(Vector(384), nullable=True)
    else:
        embedding = Column(JSONB, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    tenant = relationship("Tenant", backref="knowledge_nodes")
    outgoing_edges = relationship(
        "KnowledgeEdge",
        foreign_keys="KnowledgeEdge.from_node_id",
        back_populates="from_node",
        cascade="all, delete-orphan"
    )
    incoming_edges = relationship(
        "KnowledgeEdge",
        foreign_keys="KnowledgeEdge.to_node_id",
        back_populates="to_node",
        cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_knode_tenant", "tenant_id"),
        Index("idx_knode_type_scope", "knowledge_type", "scope"),
        Index("idx_knode_domain", "problem_domain"),
        Index("idx_knode_tags", "tags", postgresql_using="gin"),
    )


class KnowledgeEdge(Base):
    """Relationships between knowledge nodes."""
    __tablename__ = "knowledge_edges"

    edge_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True)

    from_node_id = Column(UUID(as_uuid=True), ForeignKey("knowledge_nodes.node_id", ondelete="CASCADE"), nullable=False, index=True)
    to_node_id = Column(UUID(as_uuid=True), ForeignKey("knowledge_nodes.node_id", ondelete="CASCADE"), nullable=False, index=True)

    # Relationship details
    edge_type = Column("relationship", String(50), nullable=False, index=True)
    strength = Column(Float, default=1.0)

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    created_by = Column(String(100), nullable=True)

    # Relationships
    tenant = relationship("Tenant", backref="knowledge_edges")
    from_node = relationship("KnowledgeNode", foreign_keys=[from_node_id], back_populates="outgoing_edges")
    to_node = relationship("KnowledgeNode", foreign_keys=[to_node_id], back_populates="incoming_edges")

    __table_args__ = (
        UniqueConstraint("from_node_id", "to_node_id", "relationship", name="uq_knowledge_edge"),
        Index("idx_kedge_tenant", "tenant_id"),
    )


# =============================================================================
# SCHEMA VERSION TRACKING
# =============================================================================

class SchemaVersion(Base):
    """Track schema migrations."""
    __tablename__ = "_schema_version"

    version = Column(Integer, primary_key=True)
    applied_at = Column(DateTime(timezone=True), server_default=func.now())
    description = Column(String(255))


# Current schema version
SCHEMA_VERSION = 2


# =============================================================================
# TABLE CREATION HELPERS
# =============================================================================

def create_all_tables(engine):
    """
    Create all tables in the database.
    
    Note: Run the following SQL first:
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE EXTENSION IF NOT EXISTS pg_trgm;
    """
    Base.metadata.create_all(engine)


def get_pgvector_index_sql() -> str:
    """
    Returns SQL to create pgvector indexes for embedding columns.
    Run after table creation for optimal similarity search performance.
    """
    return """
    -- Enable pgvector extension
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION IF NOT EXISTS pg_trgm;
    
    -- Create IVFFlat index on doc_chunks for approximate nearest neighbor
    -- Adjust 'lists' parameter based on data size (sqrt(n) is a good starting point)
    CREATE INDEX IF NOT EXISTS idx_chunk_embedding_ivfflat 
    ON doc_chunks 
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
    
    -- Create index on knowledge_nodes embeddings
    CREATE INDEX IF NOT EXISTS idx_knode_embedding_ivfflat 
    ON knowledge_nodes 
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
    
    -- Similarity search function for document chunks
    CREATE OR REPLACE FUNCTION match_chunks(
        query_embedding vector(384),
        match_count int DEFAULT 10,
        filter_tenant_id uuid DEFAULT NULL,
        filter_document_id uuid DEFAULT NULL
    )
    RETURNS TABLE (
        id uuid,
        document_id uuid,
        content text,
        similarity float
    )
    LANGUAGE plpgsql
    AS $$
    BEGIN
        RETURN QUERY
        SELECT 
            dc.id,
            dc.document_id,
            dc.content,
            1 - (dc.embedding <=> query_embedding) as similarity
        FROM doc_chunks dc
        WHERE 
            dc.embedding IS NOT NULL
            AND (filter_tenant_id IS NULL OR dc.tenant_id = filter_tenant_id)
            AND (filter_document_id IS NULL OR dc.document_id = filter_document_id)
        ORDER BY dc.embedding <=> query_embedding
        LIMIT match_count;
    END;
    $$;
    """
