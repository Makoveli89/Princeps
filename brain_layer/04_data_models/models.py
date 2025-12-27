"""
Princeps Brain Layer - SQLAlchemy Models for Postgres
======================================================

Comprehensive data models mapped from legacy SQLite schemas to Postgres,
with pgvector support for embeddings and JSONB for flexible metadata.

Source Patterns:
- §1 Document Ingestion: pdf_utils.py, librarian.py
- §2 Activity Tracing: umi_client.py, memory_system_improvements.py
- §3 Knowledge Distillation: librarian_agent.py analysis tables
- §4 Data Models: knowledge_network.py dataclasses

Usage:
    from models import Base, Document, DocChunk, AgentRun, KnowledgeNode
    
    # Create all tables
    Base.metadata.create_all(engine)
"""

import enum
from datetime import datetime

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
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# pgvector support - install with: pip install pgvector
try:
    from pgvector.sqlalchemy import Vector
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False
    Vector = None  # type: ignore

Base = declarative_base()


# =============================================================================
# ENUMS - Matching legacy code patterns
# =============================================================================

class KnowledgeTypeEnum(enum.Enum):
    """Types of knowledge entries (from librarian_agent.py)"""
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
    """Security classification (from librarian_agent.py)"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class NodeKnowledgeTypeEnum(enum.Enum):
    """Cross-agent knowledge types (from knowledge_network.py)"""
    SOLUTION = "solution"
    PATTERN = "pattern"
    INSIGHT = "insight"
    BEST_PRACTICE = "best_practice"
    FAILURE = "failure"
    OPTIMIZATION = "optimization"


class ShareScopeEnum(enum.Enum):
    """Knowledge sharing scope (from knowledge_network.py)"""
    PRIVATE = "private"
    TEAM = "team"
    PUBLIC = "public"


class EntityLabelEnum(enum.Enum):
    """NER entity types (from ner_agent.py / spaCy)"""
    PERSON = "PERSON"
    ORG = "ORG"
    GPE = "GPE"  # Geo-political entity
    LOC = "LOC"
    DATE = "DATE"
    TIME = "TIME"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    WORK_OF_ART = "WORK_OF_ART"
    LAW = "LAW"
    LANGUAGE = "LANGUAGE"
    QUANTITY = "QUANTITY"
    ORDINAL = "ORDINAL"
    CARDINAL = "CARDINAL"
    FAC = "FAC"  # Facility
    NORP = "NORP"  # Nationalities, religious/political groups


# =============================================================================
# DOCUMENTS & CHUNKS - Core content storage
# =============================================================================

class Document(Base):
    """
    Primary document/knowledge entry storage.
    
    Source: Librarian's `knowledge` table (librarian_agent.py)
    
    Stores document metadata and full content. For large documents,
    content is chunked into DocChunk records for embedding search.
    """
    __tablename__ = "documents"

    id = Column(String(64), primary_key=True, comment="MD5 hash-based ID")
    type = Column(
        Enum(KnowledgeTypeEnum),
        nullable=False,
        default=KnowledgeTypeEnum.DOCUMENTATION,
        comment="Knowledge type classification"
    )
    title = Column(String(500), nullable=False, index=True)
    content = Column(Text, nullable=False, comment="Full document content")
    source = Column(String(255), nullable=False, index=True, comment="Agent or system that created this")
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Classification
    tags = Column(ARRAY(String), default=[], comment="Tag array for filtering")
    metadata = Column(JSONB, default={}, comment="Flexible metadata storage")
    category = Column(String(100), index=True)
    security_level = Column(
        Enum(SecurityLevelEnum),
        default=SecurityLevelEnum.INTERNAL,
        nullable=False
    )
    
    # Versioning
    version = Column(Integer, default=1)
    parent_id = Column(String(64), ForeignKey("documents.id"), nullable=True)
    
    # Integrity
    checksum = Column(String(64), comment="SHA-256 hash of content")
    
    # Usage tracking
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, nullable=True)
    
    # Vector embedding reference (for non-chunked docs)
    embedding_id = Column(Integer, nullable=True, comment="FAISS/pgvector index position")

    # Relationships
    parent = relationship("Document", remote_side=[id], backref="versions")
    chunks = relationship("DocChunk", back_populates="document", cascade="all, delete-orphan")
    entities = relationship("DocumentEntity", back_populates="document", cascade="all, delete-orphan")
    topics = relationship("DocumentTopic", back_populates="document", cascade="all, delete-orphan")
    concepts = relationship("DocumentConcept", back_populates="document", cascade="all, delete-orphan")
    summaries = relationship("DocumentSummary", back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_doc_type_source", "type", "source"),
        Index("idx_doc_created", "created_at"),
        Index("idx_doc_tags", "tags", postgresql_using="gin"),
        Index("idx_doc_metadata", "metadata", postgresql_using="gin"),
    )

    def __repr__(self):
        return f"<Document(id={self.id[:8]}..., title='{self.title[:30]}...', source='{self.source}')>"


class DocChunk(Base):
    """
    Document chunks for embedding-based retrieval.
    
    Source: pdf_utils.py chunking logic, Lumina librarian.py
    
    Each chunk is ~800-1000 tokens with overlap for context preservation.
    Embeddings stored via pgvector for similarity search.
    """
    __tablename__ = "doc_chunks"

    id = Column(String(64), primary_key=True)
    document_id = Column(String(64), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Content
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False, comment="Position in original document")
    
    # Token info (from pdf_utils.py)
    token_count = Column(Integer, comment="tiktoken count")
    char_count = Column(Integer, comment="Character count fallback")
    
    # Embedding - pgvector column (384 dims for all-MiniLM-L6-v2)
    # If pgvector not available, store as NULL and use external index
    if PGVECTOR_AVAILABLE:
        embedding = Column(Vector(384), nullable=True, comment="Sentence-BERT embedding")
    else:
        embedding = Column(Text, nullable=True, comment="Serialized embedding (pgvector not installed)")
    
    # Metadata
    metadata = Column(JSONB, default={}, comment="Chunk-specific metadata (page number, etc.)")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_doc_chunk_index"),
        Index("idx_chunk_doc", "document_id"),
    )

    def __repr__(self):
        return f"<DocChunk(doc={self.document_id[:8]}..., idx={self.chunk_index}, tokens={self.token_count})>"


# =============================================================================
# ANALYSIS TABLES - Knowledge atoms from distillation
# =============================================================================

class DocumentEntity(Base):
    """
    Named entities extracted from documents.
    
    Source: librarian_agent.py `entities` table, NERAgent (spaCy)
    """
    __tablename__ = "document_entities"

    id = Column(String(64), primary_key=True)
    document_id = Column(String(64), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    
    text = Column(String(500), nullable=False, index=True, comment="Entity text")
    label = Column(String(50), nullable=False, index=True, comment="Entity type (PERSON, ORG, etc.)")
    
    # Character positions (optional, for highlighting)
    start_char = Column(Integer, nullable=True)
    end_char = Column(Integer, nullable=True)
    
    # Metadata
    confidence = Column(Float, nullable=True, comment="NER confidence score")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="entities")

    __table_args__ = (
        Index("idx_entity_label_text", "label", "text"),
    )

    def __repr__(self):
        return f"<DocumentEntity(text='{self.text}', label={self.label})>"


class DocumentTopic(Base):
    """
    Topics identified in documents.
    
    Source: librarian_agent.py `topics` table, TopicModelingAgent (BERTopic)
    """
    __tablename__ = "document_topics"

    id = Column(String(64), primary_key=True)
    document_id = Column(String(64), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    
    topic_id = Column(Integer, nullable=False, comment="Topic cluster ID from BERTopic")
    name = Column(String(255), nullable=True, comment="Human-readable topic name")
    keywords = Column(ARRAY(String), default=[], comment="Top keywords for this topic")
    
    # Confidence
    probability = Column(Float, nullable=True, comment="Topic assignment probability")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="topics")

    __table_args__ = (
        Index("idx_topic_id", "topic_id"),
    )

    def __repr__(self):
        return f"<DocumentTopic(topic_id={self.topic_id}, name='{self.name}')>"


class DocumentConcept(Base):
    """
    Key concepts extracted from documents.
    
    Source: librarian_agent.py `concepts` table, ConceptGraphAgent (KeyBERT)
    """
    __tablename__ = "document_concepts"

    id = Column(String(64), primary_key=True)
    document_id = Column(String(64), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    
    concept = Column(String(255), nullable=False, index=True, comment="Concept/keyword phrase")
    relevance = Column(Float, nullable=False, comment="Relevance score from KeyBERT (0-1)")
    
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="concepts")

    __table_args__ = (
        Index("idx_concept_relevance", "relevance"),
    )

    def __repr__(self):
        return f"<DocumentConcept(concept='{self.concept}', relevance={self.relevance:.2f})>"


class DocumentSummary(Base):
    """
    Generated summaries for documents.
    
    Source: librarian_agent.py `summaries` table, SummarizationAgent (BART)
    """
    __tablename__ = "document_summaries"

    id = Column(String(64), primary_key=True)
    document_id = Column(String(64), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, unique=True)
    
    one_sentence = Column(Text, nullable=True, comment="One-line summary")
    executive = Column(Text, nullable=True, comment="Executive summary (longer)")
    
    # Generation metadata
    model_used = Column(String(100), nullable=True, comment="Model that generated summaries")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="summaries")

    def __repr__(self):
        preview = self.one_sentence[:50] if self.one_sentence else "None"
        return f"<DocumentSummary(doc={self.document_id[:8]}..., preview='{preview}...')>"


# =============================================================================
# AGENT RUNS - Activity tracing and decision logging
# =============================================================================

class AgentRun(Base):
    """
    Log of agent task executions and outcomes.
    
    Source: memory_system_improvements.py `learnings` table, umi_client.py
    
    Tracks every agent decision for learning, debugging, and improvement.
    """
    __tablename__ = "agent_runs"

    id = Column(String(64), primary_key=True)
    
    # Task info
    agent_id = Column(String(100), nullable=False, index=True, comment="Agent that executed the task")
    task = Column(Text, nullable=False, comment="Task description")
    task_hash = Column(String(64), index=True, comment="Hash for deduplication/grouping")
    iteration = Column(Integer, default=1, comment="Retry count for same task")
    
    # Outcome
    success = Column(Boolean, nullable=False, index=True)
    score = Column(Integer, nullable=True, comment="Quality score 0-100")
    solution = Column(JSONB, nullable=True, comment="Structured output/result")
    feedback = Column(Text, nullable=True, comment="Feedback or error message")
    
    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    duration_ms = Column(Integer, nullable=True, comment="Execution time in milliseconds")
    
    # Usage tracking (for knowledge retrieval)
    relevance_count = Column(Integer, default=0, comment="Times retrieved as relevant")
    last_accessed = Column(DateTime, nullable=True)
    
    # Consolidation (for memory system)
    consolidated = Column(Boolean, default=False, comment="Whether merged into parent")
    parent_id = Column(String(64), ForeignKey("agent_runs.id"), nullable=True)
    
    # Context
    context = Column(JSONB, nullable=True, comment="Input context for the run")
    tools_used = Column(ARRAY(String), default=[], comment="Tools/APIs invoked")

    # Relationships
    parent = relationship("AgentRun", remote_side=[id], backref="children")

    __table_args__ = (
        Index("idx_run_agent_success", "agent_id", "success"),
        Index("idx_run_task_hash", "task_hash", "consolidated"),
        Index("idx_run_score", "score"),
        Index("idx_run_started", "started_at"),
    )

    def __repr__(self):
        status = "✓" if self.success else "✗"
        return f"<AgentRun({status} agent={self.agent_id}, task='{self.task[:30]}...', score={self.score})>"


# =============================================================================
# KNOWLEDGE NETWORK - Cross-agent intelligence sharing
# =============================================================================

class KnowledgeNode(Base):
    """
    Cross-agent knowledge nodes for shared intelligence.
    
    Source: knowledge_network.py KnowledgeNode dataclass
    
    Represents distilled, shareable knowledge that can propagate
    across agents in the system.
    """
    __tablename__ = "knowledge_nodes"

    node_id = Column(String(100), primary_key=True)
    agent_id = Column(String(100), nullable=False, index=True, comment="Source agent")
    
    # Classification
    knowledge_type = Column(
        Enum(NodeKnowledgeTypeEnum),
        nullable=False,
        comment="Type of knowledge"
    )
    scope = Column(
        Enum(ShareScopeEnum),
        default=ShareScopeEnum.PUBLIC,
        nullable=False,
        comment="Sharing scope"
    )
    
    # Content
    title = Column(String(500), nullable=False, index=True)
    content = Column(Text, nullable=False)
    tags = Column(ARRAY(String), default=[])
    problem_domain = Column(String(100), nullable=True, index=True)
    
    # Versioning
    version = Column(Integer, default=1)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Usage metrics
    access_count = Column(Integer, default=0)
    success_rate = Column(Float, default=0.0, comment="How often it helped (EMA)")
    relevance_score = Column(Float, default=1.0)
    
    # Sharing
    shared_with = Column(ARRAY(String), default=[], comment="Agent IDs this was shared with")
    
    # Context
    prerequisites = Column(ARRAY(String), default=[], comment="Required prior knowledge")
    related_nodes = Column(ARRAY(String), default=[], comment="Related node IDs")
    
    # Embedding for semantic search
    if PGVECTOR_AVAILABLE:
        embedding = Column(Vector(384), nullable=True)
    else:
        embedding = Column(Text, nullable=True)

    # Relationships
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
    flows_out = relationship(
        "KnowledgeFlow",
        foreign_keys="KnowledgeFlow.knowledge_node_id",
        back_populates="knowledge_node"
    )

    __table_args__ = (
        Index("idx_node_type_scope", "knowledge_type", "scope"),
        Index("idx_node_domain", "problem_domain"),
        Index("idx_node_tags", "tags", postgresql_using="gin"),
    )

    def __repr__(self):
        return f"<KnowledgeNode(id={self.node_id}, type={self.knowledge_type.value}, title='{self.title[:30]}...')>"


class KnowledgeEdge(Base):
    """
    Relationships between knowledge nodes.
    
    Source: knowledge_network.py KnowledgeEdge dataclass
    
    Supports relationships like: builds_on, contradicts, enhances, complements
    """
    __tablename__ = "knowledge_edges"

    edge_id = Column(String(100), primary_key=True)
    from_node_id = Column(String(100), ForeignKey("knowledge_nodes.node_id", ondelete="CASCADE"), nullable=False, index=True)
    to_node_id = Column(String(100), ForeignKey("knowledge_nodes.node_id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Relationship type
    relationship = Column(String(50), nullable=False, index=True, comment="builds_on, contradicts, enhances, etc.")
    strength = Column(Float, default=1.0, comment="Relationship strength 0-1")
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String(100), nullable=True, comment="Agent that created this edge")

    # Relationships
    from_node = relationship("KnowledgeNode", foreign_keys=[from_node_id], back_populates="outgoing_edges")
    to_node = relationship("KnowledgeNode", foreign_keys=[to_node_id], back_populates="incoming_edges")

    __table_args__ = (
        UniqueConstraint("from_node_id", "to_node_id", "relationship", name="uq_edge"),
        Index("idx_edge_relationship", "relationship"),
    )

    def __repr__(self):
        return f"<KnowledgeEdge({self.from_node_id[:8]}.. --{self.relationship}--> {self.to_node_id[:8]}..)>"


class KnowledgeFlow(Base):
    """
    Tracks knowledge transfer between agents.
    
    Source: knowledge_network.py KnowledgeFlow dataclass
    
    Records when knowledge propagates from one agent to another
    and measures the impact.
    """
    __tablename__ = "knowledge_flows"

    flow_id = Column(String(100), primary_key=True)
    
    # Participants
    from_agent_id = Column(String(100), nullable=False, index=True)
    to_agent_id = Column(String(100), nullable=False, index=True)
    knowledge_node_id = Column(String(100), ForeignKey("knowledge_nodes.node_id"), nullable=False, index=True)
    
    # Transfer details
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    outcome = Column(String(20), nullable=True, comment="success, failure, pending")
    impact_score = Column(Float, default=0.0, comment="Measured improvement from using this knowledge")
    
    # Context
    context = Column(JSONB, nullable=True, comment="Context in which knowledge was applied")

    # Relationships
    knowledge_node = relationship("KnowledgeNode", back_populates="flows_out")

    __table_args__ = (
        Index("idx_flow_agents", "from_agent_id", "to_agent_id"),
        Index("idx_flow_outcome", "outcome"),
    )

    def __repr__(self):
        return f"<KnowledgeFlow({self.from_agent_id} -> {self.to_agent_id}, outcome={self.outcome})>"


# =============================================================================
# CONCEPT GRAPH - Evidence linking
# =============================================================================

class ConceptNode(Base):
    """
    Global concept nodes for the knowledge graph.
    
    Source: concept_graph_agent.py (KeyBERT concepts linked via networkx)
    
    Represents unique concepts that appear across documents,
    enabling cross-document discovery.
    """
    __tablename__ = "concept_nodes"

    id = Column(String(64), primary_key=True)
    concept = Column(String(255), nullable=False, unique=True, index=True)
    
    # Aggregated stats
    document_count = Column(Integer, default=0, comment="Number of docs containing this concept")
    avg_relevance = Column(Float, default=0.0, comment="Average relevance across docs")
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    document_links = relationship("ConceptDocumentLink", back_populates="concept_node", cascade="all, delete-orphan")
    related_concepts = relationship(
        "ConceptRelation",
        foreign_keys="ConceptRelation.from_concept_id",
        back_populates="from_concept",
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<ConceptNode(concept='{self.concept}', docs={self.document_count})>"


class ConceptDocumentLink(Base):
    """
    Links concepts to documents with relevance scores.
    
    Source: concept_graph_agent.py document-concept edges
    """
    __tablename__ = "concept_document_links"

    id = Column(String(64), primary_key=True)
    concept_id = Column(String(64), ForeignKey("concept_nodes.id", ondelete="CASCADE"), nullable=False, index=True)
    document_id = Column(String(64), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    
    relevance = Column(Float, nullable=False, comment="Relevance score for this doc")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    concept_node = relationship("ConceptNode", back_populates="document_links")
    document = relationship("Document")

    __table_args__ = (
        UniqueConstraint("concept_id", "document_id", name="uq_concept_doc"),
    )

    def __repr__(self):
        return f"<ConceptDocumentLink(concept={self.concept_id[:8]}..., doc={self.document_id[:8]}...)>"


class ConceptRelation(Base):
    """
    Relationships between concepts (co-occurrence).
    
    Source: concept_graph_agent.py concept-concept edges
    """
    __tablename__ = "concept_relations"

    id = Column(String(64), primary_key=True)
    from_concept_id = Column(String(64), ForeignKey("concept_nodes.id", ondelete="CASCADE"), nullable=False, index=True)
    to_concept_id = Column(String(64), ForeignKey("concept_nodes.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Co-occurrence strength
    weight = Column(Float, default=1.0, comment="Co-occurrence weight")
    cooccurrence_count = Column(Integer, default=1, comment="Number of docs where both appear")
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    from_concept = relationship("ConceptNode", foreign_keys=[from_concept_id], back_populates="related_concepts")
    to_concept = relationship("ConceptNode", foreign_keys=[to_concept_id])

    __table_args__ = (
        UniqueConstraint("from_concept_id", "to_concept_id", name="uq_concept_relation"),
    )

    def __repr__(self):
        return f"<ConceptRelation({self.from_concept_id[:8]}.. <-> {self.to_concept_id[:8]}.. weight={self.weight:.2f})>"


# =============================================================================
# PRIORITY KNOWLEDGE - Promotion system
# =============================================================================

class PriorityKnowledge(Base):
    """
    High-priority knowledge items for quick access.
    
    Source: memory_system_improvements.py PriorityKnowledgeSharing
    
    Stores promoted knowledge with priority scoring based on
    confidence, recency, and usage frequency.
    """
    __tablename__ = "priority_knowledge"

    id = Column(String(64), primary_key=True)
    
    # Reference to source
    source_type = Column(String(50), nullable=False, comment="document, agent_run, knowledge_node")
    source_id = Column(String(100), nullable=False, index=True)
    
    # Priority scoring (from PriorityKnowledgeSharing)
    confidence = Column(Float, default=0.5, comment="Confidence score 0-1")
    recency_score = Column(Float, default=1.0, comment="Decays over time")
    usage_frequency = Column(Integer, default=0)
    priority_score = Column(Float, default=0.0, index=True, comment="Composite priority score")
    
    # Content snapshot
    title = Column(String(500), nullable=False)
    content_preview = Column(Text, nullable=True, comment="First 500 chars for quick access")
    
    # Timestamps
    promoted_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True, comment="Optional expiration")
    
    # Status
    is_active = Column(Boolean, default=True, index=True)

    __table_args__ = (
        Index("idx_priority_score_active", "priority_score", "is_active"),
        UniqueConstraint("source_type", "source_id", name="uq_priority_source"),
    )

    def calculate_priority(self) -> float:
        """
        Calculate composite priority score.
        Formula from memory_system_improvements.py PriorityKnowledgeSharing.
        """
        return (
            self.confidence * 0.4 +
            self.recency_score * 0.3 +
            min(self.usage_frequency / 100, 1.0) * 0.3
        )

    def __repr__(self):
        return f"<PriorityKnowledge(title='{self.title[:30]}...', priority={self.priority_score:.2f})>"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_all_tables(engine):
    """
    Create all tables in the database.
    
    Requires pgvector extension for embedding columns:
        CREATE EXTENSION IF NOT EXISTS vector;
    """
    Base.metadata.create_all(engine)


def get_vector_index_sql() -> str:
    """
    Returns SQL to create pgvector indexes for embedding columns.
    Run after table creation for optimal similarity search performance.
    """
    return """
    -- Enable pgvector extension
    CREATE EXTENSION IF NOT EXISTS vector;
    
    -- Create IVFFlat index on doc_chunks for approximate nearest neighbor
    -- Adjust lists parameter based on data size (sqrt(n) is a good starting point)
    CREATE INDEX IF NOT EXISTS idx_chunk_embedding 
    ON doc_chunks 
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
    
    -- Create index on knowledge_nodes embeddings
    CREATE INDEX IF NOT EXISTS idx_node_embedding 
    ON knowledge_nodes 
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
    """


# =============================================================================
# SCHEMA VERSION TRACKING
# =============================================================================

class SchemaVersion(Base):
    """Track schema migrations"""
    __tablename__ = "_schema_version"
    
    version = Column(Integer, primary_key=True)
    applied_at = Column(DateTime, default=datetime.utcnow)
    description = Column(String(255))


# Current schema version
SCHEMA_VERSION = 1
