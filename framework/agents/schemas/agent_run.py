"""
Agent Run Schema - Pydantic Models for Database Alignment

This module defines Pydantic models that match the Postgres agent_runs table schema
exactly, ensuring data validation before database operations.

Table Schema (agent_runs):
- id: UUID primary key
- task_id: UUID foreign key to tasks table
- agent_id: VARCHAR agent identifier
- agent_name: VARCHAR human-readable agent name
- agent_type: VARCHAR agent type classification
- tenant_id: UUID for multi-tenant isolation
- started_at: TIMESTAMP when run began
- completed_at: TIMESTAMP when run ended (nullable)
- status: VARCHAR (pending, in_progress, completed, failed, cancelled)
- success: BOOLEAN whether run succeeded
- input_data: JSONB task input and parameters
- output_data: JSONB task output and results
- context: JSONB additional context (tenant, model used, etc.)
- error: TEXT error message if failed
- error_type: VARCHAR error classification
- retry_count: INT number of retries attempted
- tokens_used: INT total tokens consumed
- model_used: VARCHAR which LLM model was used
- fallback_used: BOOLEAN whether fallback was triggered
- created_at: TIMESTAMP auto-generated
- updated_at: TIMESTAMP auto-updated
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator


class RunStatus(str, Enum):
    """Status of an agent run - matches DB enum"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class AgentRunCreate(BaseModel):
    """Schema for creating a new agent run record"""

    task_id: Optional[str] = Field(None, description="Parent task ID if applicable")
    agent_id: str = Field(..., description="Unique agent identifier")
    agent_name: str = Field(..., description="Human-readable agent name")
    agent_type: str = Field(..., description="Type of agent (summarization, code_gen, etc.)")
    tenant_id: Optional[str] = Field(None, description="Tenant ID for multi-tenant isolation")

    status: RunStatus = Field(default=RunStatus.PENDING)
    started_at: datetime = Field(default_factory=datetime.utcnow)

    input_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Task input including prompt, messages, parameters"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context: tenant info, routing preferences, etc."
    )

    class Config:
        use_enum_values = True


class AgentRunUpdate(BaseModel):
    """Schema for updating an existing agent run record"""

    status: Optional[RunStatus] = None
    completed_at: Optional[datetime] = None
    success: Optional[bool] = None

    output_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Task output including response text, structured output"
    )

    error: Optional[str] = Field(None, description="Error message if failed")
    error_type: Optional[str] = Field(None, description="Error classification")

    retry_count: Optional[int] = Field(None, ge=0)
    tokens_used: Optional[int] = Field(None, ge=0)
    model_used: Optional[str] = None
    fallback_used: Optional[bool] = None

    # Model usage tracking in context
    context: Optional[Dict[str, Any]] = None

    class Config:
        use_enum_values = True


class AgentRunRecord(BaseModel):
    """
    Complete agent run record - matches full DB row.

    Used for reading/displaying agent run data.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: Optional[str] = None
    agent_id: str
    agent_name: str
    agent_type: str
    tenant_id: Optional[str] = None

    started_at: datetime
    completed_at: Optional[datetime] = None

    status: RunStatus
    success: bool = False

    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)

    error: Optional[str] = None
    error_type: Optional[str] = None

    retry_count: int = 0
    tokens_used: int = 0
    model_used: Optional[str] = None
    fallback_used: bool = False

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True

    def to_db_dict(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for database insertion"""
        data = self.dict()
        # Convert datetimes to ISO format for JSON serialization
        for key in ['started_at', 'completed_at', 'created_at', 'updated_at']:
            if data[key] is not None:
                data[key] = data[key].isoformat() if isinstance(data[key], datetime) else data[key]
        return data

    @classmethod
    def from_db_row(cls, row: Dict[str, Any]) -> "AgentRunRecord":
        """Create instance from database row"""
        return cls(**row)


class ModelUsageRecord(BaseModel):
    """
    Tracks LLM model usage within a run.

    Can be stored in the context.model_usage field of AgentRunRecord
    or in a separate operations table.
    """

    provider: str = Field(..., description="LLM provider (anthropic, openai, google)")
    model: str = Field(..., description="Specific model used")

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)

    latency_ms: float = Field(default=0.0, ge=0)
    success: bool = True

    is_fallback: bool = Field(
        default=False,
        description="Whether this was a fallback attempt"
    )
    attempt_number: int = Field(default=1, ge=1)

    timestamp: datetime = Field(default_factory=datetime.utcnow)

    error: Optional[str] = None


class PIIScanResult(BaseModel):
    """
    Records PII scan results for audit trail.

    Stored in context.pii_scan of AgentRunRecord.
    """

    scan_performed: bool = True
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    has_pii: bool = False
    has_secrets: bool = False

    pii_types_found: List[str] = Field(
        default_factory=list,
        description="Types of PII detected (email, phone, ssn, etc.)"
    )
    secrets_types_found: List[str] = Field(
        default_factory=list,
        description="Types of secrets detected (api_key, password, etc.)"
    )

    content_redacted: bool = Field(
        default=False,
        description="Whether content was redacted before sending to LLM"
    )

    @validator('pii_types_found', 'secrets_types_found', pre=True, always=True)
    def ensure_list(cls, v):
        return v or []
