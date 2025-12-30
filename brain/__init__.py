"""
Princeps Brain Layer
====================

A sophisticated knowledge management system for AI agents.

Modules:
    - core: Database models, schemas, and utilities
    - ingestion: Document and repository ingestion pipeline
    - distillation: Knowledge extraction and summarization
    - observability: Logging, metrics, and run tracking
    - resilience: Fault tolerance, retries, and idempotency
    - security: PII scanning, tenant isolation, access control
    - interface: CLI and API endpoints
"""

__version__ = "0.1.0"
__author__ = "Princeps Team"

from .core.db import get_engine, get_session, init_db
from .core.models import (
    Artifact,
    Base,
    Decision,
    DocChunk,
    Document,
    KnowledgeEdge,
    KnowledgeNode,
    Operation,
    Repository,
    Resource,
    Tenant,
)

__all__ = [
    # Core models
    "Base",
    "Tenant",
    "Repository",
    "Resource",
    "Document",
    "DocChunk",
    "Operation",
    "Artifact",
    "Decision",
    "KnowledgeNode",
    "KnowledgeEdge",
    # Database utilities
    "get_session",
    "get_engine",
    "init_db",
]
