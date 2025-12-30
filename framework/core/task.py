"""
Task Schema - Unified task representation for the Dispatcher.

This module defines the Task model and related types for task routing:
- Task: Core task representation with type, payload, and context
- TaskType: Enumeration of supported task types
- TaskResult: Result from task execution
- TaskContext: Execution context with correlation and tenant info

The Task model provides a consistent interface between the Dispatcher
and all agents, enabling proper routing, logging, and traceability.

Strategic Intent:
Every request to the system (user query, document upload, automated trigger)
is represented as a Task. The Dispatcher examines the task type and routes
it to appropriate agents. Task IDs propagate through all agent runs for
full traceability.

Adapted from patterns in:
- brain/observability/logging_config.py (OperationContext, correlation IDs)
- brain/resilience/idempotency_service.py (input hashing)
- brain/core/models.py (OperationTypeEnum)
"""

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union


# =============================================================================
# Task Type Enumeration
# =============================================================================

class TaskType(Enum):
    """
    Types of tasks the Dispatcher can handle.

    Each task type maps to a specific workflow of agent invocations.
    """
    # Query and reasoning tasks
    ASK_QUESTION = "ask_question"           # Answer a user question
    COMPLEX_REASONING = "complex_reasoning"  # Multi-step reasoning
    SEARCH = "search"                        # Search knowledge base

    # Planning and execution tasks
    PLAN = "plan"                            # Create a plan for a goal
    EXECUTE_PLAN = "execute_plan"            # Execute an existing plan
    PLAN_AND_EXECUTE = "plan_and_execute"    # Full planning + execution

    # Document and knowledge tasks
    INGEST_DOCUMENT = "ingest_document"      # Ingest a new document
    ANALYZE_DOCUMENT = "analyze_document"    # Full document analysis
    SUMMARIZE = "summarize"                  # Generate summary
    EXTRACT_ENTITIES = "extract_entities"    # Extract named entities
    EXTRACT_TOPICS = "extract_topics"        # Extract topics
    EXTRACT_CONCEPTS = "extract_concepts"    # Build concept graph

    # Knowledge distillation (combined)
    DISTILL_KNOWLEDGE = "distill_knowledge"  # Run all distillation agents

    # Retrieval tasks
    RETRIEVE = "retrieve"                    # Retrieve relevant knowledge
    SEMANTIC_SEARCH = "semantic_search"      # Vector-based search

    # Administrative tasks
    HEALTH_CHECK = "health_check"            # System health check
    CLEAR_CACHE = "clear_cache"              # Clear caches

    # Custom/extensible
    CUSTOM = "custom"                        # Custom workflow


class TaskPriority(Enum):
    """Priority levels for task scheduling."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class TaskStatus(Enum):
    """Status of a task through its lifecycle."""
    PENDING = "pending"
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    WAITING = "waiting"  # Waiting for sub-tasks
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


# =============================================================================
# Task Context
# =============================================================================

@dataclass
class TaskContext:
    """
    Execution context for a task.

    Contains correlation IDs, tenant info, and other contextual data
    that propagates through all agent runs.
    """
    # Correlation and tracing
    correlation_id: str = field(default_factory=lambda: f"corr-{uuid.uuid4().hex[:12]}")
    parent_task_id: Optional[str] = None
    trace_id: Optional[str] = None

    # Tenant and user
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    # Environment
    environment: str = "development"

    # Preferences
    max_retries: int = 3
    timeout_seconds: float = 300.0
    priority: TaskPriority = TaskPriority.NORMAL

    # Feature flags
    enable_caching: bool = True
    enable_logging: bool = True
    enable_idempotency: bool = True

    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "correlation_id": self.correlation_id,
            "parent_task_id": self.parent_task_id,
            "trace_id": self.trace_id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "environment": self.environment,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "priority": self.priority.value,
            "enable_caching": self.enable_caching,
            "enable_logging": self.enable_logging,
            "enable_idempotency": self.enable_idempotency,
            "metadata": self.metadata,
        }

    def to_agent_context(self):
        """Convert to AgentContext for agent execution."""
        from agents import AgentContext

        return AgentContext(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            session_id=self.session_id,
            correlation_id=self.correlation_id,
            environment=self.environment,
            metadata=self.metadata,
        )


# =============================================================================
# Task Definition
# =============================================================================

@dataclass
class Task:
    """
    Core task representation for the Dispatcher.

    A Task encapsulates everything needed to route and execute a request:
    - What type of task it is
    - The payload/input data
    - Execution context
    - Routing hints
    """
    # Identity
    id: str = field(default_factory=lambda: f"task-{uuid.uuid4().hex[:16]}")
    type: TaskType = TaskType.CUSTOM

    # Payload
    payload: Dict[str, Any] = field(default_factory=dict)

    # Context
    context: TaskContext = field(default_factory=TaskContext)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Status
    status: TaskStatus = TaskStatus.PENDING

    # Routing hints
    preferred_agents: Optional[List[str]] = None
    excluded_agents: Optional[List[str]] = None
    workflow_override: Optional[str] = None

    # Idempotency
    input_hash: Optional[str] = None

    def __post_init__(self):
        """Compute input hash for idempotency."""
        if self.input_hash is None and self.context.enable_idempotency:
            self.input_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute deterministic hash of task inputs."""
        # Normalize payload for consistent hashing
        hash_content = {
            "type": self.type.value,
            "payload": self._normalize_for_hash(self.payload),
            "tenant_id": self.context.tenant_id,
        }
        json_str = json.dumps(hash_content, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _normalize_for_hash(self, value: Any) -> Any:
        """Normalize a value for consistent hashing."""
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip().lower()
        if isinstance(value, dict):
            return {k: self._normalize_for_hash(v) for k, v in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [self._normalize_for_hash(v) for v in value]
        return value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "type": self.type.value,
            "payload": self.payload,
            "context": self.context.to_dict(),
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status.value,
            "input_hash": self.input_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create Task from dictionary."""
        context_data = data.get("context", {})
        priority = context_data.get("priority", TaskPriority.NORMAL.value)
        if isinstance(priority, int):
            priority = TaskPriority(priority)
        elif isinstance(priority, str):
            priority = TaskPriority[priority.upper()]

        context = TaskContext(
            correlation_id=context_data.get("correlation_id", f"corr-{uuid.uuid4().hex[:12]}"),
            parent_task_id=context_data.get("parent_task_id"),
            tenant_id=context_data.get("tenant_id"),
            user_id=context_data.get("user_id"),
            session_id=context_data.get("session_id"),
            environment=context_data.get("environment", "development"),
            priority=priority,
            metadata=context_data.get("metadata", {}),
        )

        return cls(
            id=data.get("id", f"task-{uuid.uuid4().hex[:16]}"),
            type=TaskType(data.get("type", "custom")),
            payload=data.get("payload", {}),
            context=context,
            status=TaskStatus(data.get("status", "pending")),
            input_hash=data.get("input_hash"),
        )


# =============================================================================
# Task Result
# =============================================================================

@dataclass
class TaskResult:
    """
    Result from task execution.

    Contains the output, status, and execution metadata.
    """
    # Identity
    task_id: str
    task_type: TaskType

    # Status
    success: bool
    status: TaskStatus = TaskStatus.COMPLETED

    # Output
    output: Any = None
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None

    # Execution info
    duration_ms: float = 0.0
    agents_invoked: List[str] = field(default_factory=list)
    sub_task_ids: List[str] = field(default_factory=list)

    # Caching
    was_cached: bool = False
    cache_key: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "success": self.success,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "error_details": self.error_details,
            "duration_ms": self.duration_ms,
            "agents_invoked": self.agents_invoked,
            "sub_task_ids": self.sub_task_ids,
            "was_cached": self.was_cached,
            "metadata": self.metadata,
        }


# =============================================================================
# Sub-Task Definition
# =============================================================================

@dataclass
class SubTask:
    """
    A sub-task within a larger workflow.

    Used by the Dispatcher to break complex tasks into steps.
    """
    id: str = field(default_factory=lambda: f"subtask-{uuid.uuid4().hex[:12]}")
    parent_task_id: str = ""
    agent_type: str = ""  # Which agent to invoke
    input_data: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)  # IDs of prerequisite sub-tasks
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "parent_task_id": self.parent_task_id,
            "agent_type": self.agent_type,
            "input_data": self.input_data,
            "depends_on": self.depends_on,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
        }


# =============================================================================
# Workflow Definition
# =============================================================================

@dataclass
class WorkflowStep:
    """A single step in a workflow definition."""
    name: str
    agent_type: str
    input_mapping: Dict[str, str] = field(default_factory=dict)  # Maps workflow vars to agent input
    output_key: Optional[str] = None  # Key to store output in workflow context
    condition: Optional[str] = None  # Optional condition to check before running
    on_error: str = "fail"  # "fail", "skip", "retry", "fallback"
    fallback_agent: Optional[str] = None
    max_retries: int = 0


@dataclass
class Workflow:
    """
    Definition of a multi-step workflow.

    Workflows define the sequence of agent invocations for a task type.
    """
    name: str
    task_type: TaskType
    steps: List[WorkflowStep]
    description: str = ""
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "task_type": self.task_type.value,
            "steps": [
                {
                    "name": s.name,
                    "agent_type": s.agent_type,
                    "input_mapping": s.input_mapping,
                    "output_key": s.output_key,
                    "condition": s.condition,
                    "on_error": s.on_error,
                }
                for s in self.steps
            ],
            "description": self.description,
            "version": self.version,
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_task(
    task_type: Union[TaskType, str],
    payload: Dict[str, Any],
    tenant_id: Optional[str] = None,
    user_id: Optional[str] = None,
    priority: TaskPriority = TaskPriority.NORMAL,
    **kwargs,
) -> Task:
    """
    Factory function to create a Task.

    Args:
        task_type: Type of task (enum or string)
        payload: Task input data
        tenant_id: Tenant ID for isolation
        user_id: User ID for tracking
        priority: Task priority
        **kwargs: Additional context metadata

    Returns:
        Configured Task instance
    """
    if isinstance(task_type, str):
        task_type = TaskType(task_type)

    context = TaskContext(
        tenant_id=tenant_id,
        user_id=user_id,
        priority=priority,
        metadata=kwargs,
    )

    return Task(
        type=task_type,
        payload=payload,
        context=context,
    )


def create_question_task(
    question: str,
    tenant_id: Optional[str] = None,
    context_documents: Optional[List[str]] = None,
    **kwargs,
) -> Task:
    """Create a task for answering a question."""
    return create_task(
        task_type=TaskType.ASK_QUESTION,
        payload={
            "question": question,
            "context_documents": context_documents or [],
        },
        tenant_id=tenant_id,
        **kwargs,
    )


def create_document_task(
    content: str,
    source: str = "user_upload",
    title: Optional[str] = None,
    tenant_id: Optional[str] = None,
    analyze: bool = True,
    **kwargs,
) -> Task:
    """Create a task for document ingestion."""
    task_type = TaskType.ANALYZE_DOCUMENT if analyze else TaskType.INGEST_DOCUMENT

    return create_task(
        task_type=task_type,
        payload={
            "content": content,
            "source": source,
            "title": title,
        },
        tenant_id=tenant_id,
        **kwargs,
    )


def create_plan_task(
    goal: str,
    constraints: Optional[List[str]] = None,
    tenant_id: Optional[str] = None,
    execute: bool = False,
    **kwargs,
) -> Task:
    """Create a planning task."""
    task_type = TaskType.PLAN_AND_EXECUTE if execute else TaskType.PLAN

    return create_task(
        task_type=task_type,
        payload={
            "goal": goal,
            "constraints": constraints or [],
        },
        tenant_id=tenant_id,
        **kwargs,
    )
