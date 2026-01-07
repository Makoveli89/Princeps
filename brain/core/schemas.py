"""
Princeps Brain Layer - Pydantic Validation Schemas
====================================================

Pydantic models for validating all entity types before database insertion.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


# Enums
class OperationType(str, Enum):
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


class OperationStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class ResourceType(str, Enum):
    CODE_FILE = "code_file"
    DOCUMENT = "document"
    IMAGE = "image"
    CONFIG = "config"
    DATA = "data"
    BINARY = "binary"
    DIRECTORY = "directory"


class KnowledgeType(str, Enum):
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


class SecurityLevel(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class NodeKnowledgeType(str, Enum):
    SOLUTION = "solution"
    PATTERN = "pattern"
    INSIGHT = "insight"
    BEST_PRACTICE = "best_practice"
    FAILURE = "failure"
    OPTIMIZATION = "optimization"


class ShareScope(str, Enum):
    PRIVATE = "private"
    TEAM = "team"
    PUBLIC = "public"


class ArtifactType(str, Enum):
    EMBEDDING_INDEX = "embedding_index"
    SUMMARY = "summary"
    ENTITY_LIST = "entity_list"
    DEPENDENCY_GRAPH = "dependency_graph"
    ANALYSIS_REPORT = "analysis_report"
    CODE_ANALYSIS = "code_analysis"
    MODEL_OUTPUT = "model_output"


class DecisionType(str, Enum):
    TASK_SELECTION = "task_selection"
    TOOL_CHOICE = "tool_choice"
    ROUTING = "routing"
    PRIORITIZATION = "prioritization"
    ESCALATION = "escalation"
    RETRY = "retry"
    ABORT = "abort"


# Base Schemas
class BaseSchema(BaseModel):
    model_config = ConfigDict(
        from_attributes=True, populate_by_name=True, str_strip_whitespace=True
    )


class TimestampMixin(BaseModel):
    created_at: datetime | None = None
    updated_at: datetime | None = None


# Tenant Schemas
class TenantCreate(BaseSchema):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True


class TenantResponse(TenantCreate, TimestampMixin):
    id: UUID


# Repository Schemas
class RepositoryCreate(BaseSchema):
    tenant_id: UUID
    name: str = Field(..., min_length=1, max_length=255)
    url: str = Field(..., min_length=1, max_length=1000)
    clone_path: str | None = None
    default_branch: str = "main"
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RepositoryResponse(RepositoryCreate, TimestampMixin):
    id: UUID
    last_commit_sha: str | None = None
    file_count: int = 0
    is_active: bool = True


# Resource Schemas
class ResourceCreate(BaseSchema):
    tenant_id: UUID
    repository_id: UUID | None = None
    file_path: str
    file_name: str
    file_extension: str | None = None
    resource_type: ResourceType = ResourceType.CODE_FILE
    content_hash: str = Field(..., min_length=64, max_length=64)
    size_bytes: int | None = None
    language: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResourceResponse(ResourceCreate, TimestampMixin):
    id: UUID
    has_pii: bool = False
    has_secrets: bool = False
    is_active: bool = True
    is_parsed: bool = False


# Operation Schemas
class OperationCreate(BaseSchema):
    tenant_id: UUID
    op_type: OperationType
    inputs: dict[str, Any]
    correlation_id: str | None = None
    agent_id: str | None = None
    max_retries: int = 3
    metadata: dict[str, Any] = Field(default_factory=dict)


class OperationResponse(OperationCreate, TimestampMixin):
    id: UUID
    input_hash: str
    status: OperationStatus = OperationStatus.PENDING
    outputs: dict[str, Any] | None = None
    error_message: str | None = None
    retry_count: int = 0
    duration_ms: int | None = None


# Document Schemas
class DocumentCreate(BaseSchema):
    tenant_id: UUID
    title: str = Field(..., min_length=1, max_length=500)
    content: str
    doc_type: KnowledgeType = KnowledgeType.DOCUMENTATION
    source: str
    category: str | None = None
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentResponse(DocumentCreate, TimestampMixin):
    id: UUID
    content_hash: str
    word_count: int | None = None
    token_count: int | None = None
    is_chunked: bool = False
    is_embedded: bool = False
    is_analyzed: bool = False


# DocChunk Schemas
class DocChunkCreate(BaseSchema):
    tenant_id: UUID
    document_id: UUID
    content: str
    chunk_index: int = Field(..., ge=0)
    start_char: int | None = None
    end_char: int | None = None
    token_count: int | None = None
    embedding: list[float] | None = None
    embedding_model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocChunkResponse(DocChunkCreate, TimestampMixin):
    id: UUID
    embedded_at: datetime | None = None


# Artifact Schemas
class ArtifactCreate(BaseSchema):
    tenant_id: UUID
    operation_id: UUID
    artifact_type: ArtifactType
    name: str
    description: str | None = None
    content: dict[str, Any] | None = None
    file_path: str | None = None
    quality_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactResponse(ArtifactCreate, TimestampMixin):
    id: UUID
    usage_count: int = 0


# Decision Schemas
class DecisionCreate(BaseSchema):
    tenant_id: UUID
    agent_id: str
    decision_type: DecisionType
    context: dict[str, Any]
    decision: dict[str, Any]
    reasoning: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DecisionResponse(DecisionCreate, TimestampMixin):
    id: UUID
    outcome: dict[str, Any] | None = None
    was_correct: bool | None = None
    decided_at: datetime


# Knowledge Network Schemas
class KnowledgeNodeCreate(BaseSchema):
    tenant_id: UUID
    agent_id: str
    knowledge_type: NodeKnowledgeType
    scope: ShareScope = ShareScope.PUBLIC
    title: str
    content: str
    tags: list[str] = Field(default_factory=list)
    problem_domain: str | None = None
    embedding: list[float] | None = None


class KnowledgeNodeResponse(KnowledgeNodeCreate, TimestampMixin):
    node_id: UUID
    version: int = 1
    access_count: int = 0
    success_rate: float = 0.0


class KnowledgeEdgeCreate(BaseSchema):
    tenant_id: UUID
    from_node_id: UUID
    to_node_id: UUID
    relationship: str
    strength: float = 1.0


class KnowledgeEdgeResponse(KnowledgeEdgeCreate):
    edge_id: UUID
    created_at: datetime


# Bulk Schemas
class BulkDocumentCreate(BaseSchema):
    documents: list[DocumentCreate] = Field(..., min_length=1, max_length=100)


class BulkChunkCreate(BaseSchema):
    chunks: list[DocChunkCreate] = Field(..., min_length=1, max_length=500)


# Utility
def validate_and_prepare(schema_class: type[BaseSchema], data: dict[str, Any]) -> dict[str, Any]:
    instance = schema_class(**data)
    return instance.model_dump(exclude_unset=True)
