"""
Princeps Brain Layer - Metrics Reporter
========================================

Query database for operational metrics with console output and JSON export.

Usage:
    from brain.observability import MetricsReporter, get_metrics_summary

    metrics = get_metrics_summary(session, tenant_id)
    print(metrics.to_json())
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import case, func
from sqlalchemy.orm import Session

from ..core.models import (
    AgentRun,
    DocChunk,
    Document,
    DocumentConcept,
    DocumentEntity,
    DocumentSummary,
    DocumentTopic,
    Operation,
    OperationStatusEnum,
    Repository,
    Resource,
)
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class OperationMetrics:
    """Metrics for a specific operation type."""

    op_type: str
    total_count: int = 0
    success_count: int = 0
    failed_count: int = 0
    pending_count: int = 0
    skipped_count: int = 0
    avg_duration_ms: float = 0.0
    min_duration_ms: int = 0
    max_duration_ms: int = 0
    error_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {k: round(v, 2) if isinstance(v, float) else v for k, v in self.__dict__.items()}


@dataclass
class AgentMetrics:
    """Metrics for a specific agent."""

    agent_id: str
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    avg_score: float | None = None
    avg_duration_ms: float = 0.0
    success_rate: float = 0.0
    tasks_per_hour: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            k: round(v, 2) if isinstance(v, float) and v is not None else v
            for k, v in self.__dict__.items()
        }


@dataclass
class ContentMetrics:
    """Metrics for ingested content."""

    total_repositories: int = 0
    total_resources: int = 0
    total_documents: int = 0
    total_chunks: int = 0
    documents_with_summaries: int = 0
    documents_with_entities: int = 0
    documents_with_topics: int = 0
    documents_with_concepts: int = 0
    total_entities: int = 0
    total_topics: int = 0
    total_concepts: int = 0
    pii_flagged_count: int = 0
    secrets_flagged_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class SystemMetrics:
    """Overall system health metrics."""

    total_operations: int = 0
    operations_last_hour: int = 0
    operations_last_24h: int = 0
    total_agent_runs: int = 0
    runs_last_hour: int = 0
    runs_last_24h: int = 0
    overall_success_rate: float = 0.0
    overall_error_rate: float = 0.0
    avg_operation_duration_ms: float = 0.0
    avg_run_duration_ms: float = 0.0
    oldest_pending_operation: datetime | None = None
    last_operation_time: datetime | None = None
    last_run_time: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        for k, v in d.items():
            if isinstance(v, datetime):
                d[k] = v.isoformat() if v else None
            elif isinstance(v, float):
                d[k] = round(v, 2)
        return d


@dataclass
class MetricsSummary:
    """Complete metrics summary."""

    generated_at: datetime = field(default_factory=datetime.utcnow)
    tenant_id: str | None = None
    time_range_hours: int | None = None
    system: SystemMetrics = field(default_factory=SystemMetrics)
    content: ContentMetrics = field(default_factory=ContentMetrics)
    operations_by_type: list[OperationMetrics] = field(default_factory=list)
    agents: list[AgentMetrics] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "tenant_id": self.tenant_id,
            "time_range_hours": self.time_range_hours,
            "system": self.system.to_dict(),
            "content": self.content.to_dict(),
            "operations_by_type": [o.to_dict() for o in self.operations_by_type],
            "agents": [a.to_dict() for a in self.agents],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


class MetricsCollector:
    """Collect metrics from the database."""

    def __init__(self, session: Session, tenant_id: str | UUID | None = None):
        self.session = session
        self.tenant_id = UUID(tenant_id) if tenant_id and isinstance(tenant_id, str) else tenant_id

    def _tenant_filter(self, model):
        return model.tenant_id == self.tenant_id if self.tenant_id else True

    def collect_system_metrics(self, since: datetime | None = None) -> SystemMetrics:
        """Collect overall system health metrics."""
        metrics = SystemMetrics()
        now = datetime.utcnow()
        hour_ago, day_ago = now - timedelta(hours=1), now - timedelta(hours=24)

        base = self.session.query(Operation).filter(self._tenant_filter(Operation))
        metrics.total_operations = base.count()
        metrics.operations_last_hour = base.filter(Operation.created_at >= hour_ago).count()
        metrics.operations_last_24h = base.filter(Operation.created_at >= day_ago).count()

        success = base.filter(Operation.status == OperationStatusEnum.SUCCESS).count()
        failed = base.filter(Operation.status == OperationStatusEnum.FAILED).count()
        completed = success + failed
        if completed > 0:
            metrics.overall_success_rate = success / completed
            metrics.overall_error_rate = failed / completed

        avg = (
            self.session.query(func.avg(Operation.duration_ms))
            .filter(self._tenant_filter(Operation), Operation.duration_ms.isnot(None))
            .scalar()
        )
        metrics.avg_operation_duration_ms = float(avg or 0)

        runs = self.session.query(AgentRun).filter(self._tenant_filter(AgentRun))
        metrics.total_agent_runs = runs.count()
        metrics.runs_last_hour = runs.filter(AgentRun.started_at >= hour_ago).count()
        metrics.runs_last_24h = runs.filter(AgentRun.started_at >= day_ago).count()

        avg_run = (
            self.session.query(func.avg(AgentRun.duration_ms))
            .filter(self._tenant_filter(AgentRun), AgentRun.duration_ms.isnot(None))
            .scalar()
        )
        metrics.avg_run_duration_ms = float(avg_run or 0)

        metrics.oldest_pending_operation = (
            self.session.query(func.min(Operation.created_at))
            .filter(self._tenant_filter(Operation), Operation.status == OperationStatusEnum.PENDING)
            .scalar()
        )
        metrics.last_operation_time = (
            self.session.query(func.max(Operation.created_at))
            .filter(self._tenant_filter(Operation))
            .scalar()
        )
        metrics.last_run_time = (
            self.session.query(func.max(AgentRun.started_at))
            .filter(self._tenant_filter(AgentRun))
            .scalar()
        )

        return metrics

    def collect_content_metrics(self) -> ContentMetrics:
        """Collect metrics about ingested content."""
        m = ContentMetrics()
        m.total_repositories = (
            self.session.query(Repository)
            .filter(self._tenant_filter(Repository), Repository.is_active == True)
            .count()
        )
        m.total_resources = (
            self.session.query(Resource)
            .filter(self._tenant_filter(Resource), Resource.is_active == True)
            .count()
        )
        m.total_documents = (
            self.session.query(Document).filter(self._tenant_filter(Document)).count()
        )
        m.total_chunks = self.session.query(DocChunk).filter(self._tenant_filter(DocChunk)).count()
        m.documents_with_summaries = (
            self.session.query(func.count(func.distinct(DocumentSummary.document_id)))
            .filter(self._tenant_filter(DocumentSummary))
            .scalar()
            or 0
        )
        m.documents_with_entities = (
            self.session.query(func.count(func.distinct(DocumentEntity.document_id)))
            .filter(self._tenant_filter(DocumentEntity))
            .scalar()
            or 0
        )
        m.documents_with_topics = (
            self.session.query(func.count(func.distinct(DocumentTopic.document_id)))
            .filter(self._tenant_filter(DocumentTopic))
            .scalar()
            or 0
        )
        m.documents_with_concepts = (
            self.session.query(func.count(func.distinct(DocumentConcept.document_id)))
            .filter(self._tenant_filter(DocumentConcept))
            .scalar()
            or 0
        )
        m.total_entities = (
            self.session.query(DocumentEntity).filter(self._tenant_filter(DocumentEntity)).count()
        )
        m.total_topics = (
            self.session.query(DocumentTopic).filter(self._tenant_filter(DocumentTopic)).count()
        )
        m.total_concepts = (
            self.session.query(DocumentConcept).filter(self._tenant_filter(DocumentConcept)).count()
        )
        m.pii_flagged_count = (
            self.session.query(Resource)
            .filter(self._tenant_filter(Resource), Resource.has_pii == True)
            .count()
        )
        m.secrets_flagged_count = (
            self.session.query(Resource)
            .filter(self._tenant_filter(Resource), Resource.has_secrets == True)
            .count()
        )
        return m

    def collect_operation_metrics(self, since: datetime | None = None) -> list[OperationMetrics]:
        """Collect metrics grouped by operation type."""
        query = self.session.query(
            Operation.op_type,
            func.count(Operation.id).label("total"),
            func.sum(case((Operation.status == OperationStatusEnum.SUCCESS, 1), else_=0)).label(
                "success"
            ),
            func.sum(case((Operation.status == OperationStatusEnum.FAILED, 1), else_=0)).label(
                "failed"
            ),
            func.sum(case((Operation.status == OperationStatusEnum.PENDING, 1), else_=0)).label(
                "pending"
            ),
            func.sum(case((Operation.status == OperationStatusEnum.SKIPPED, 1), else_=0)).label(
                "skipped"
            ),
            func.avg(Operation.duration_ms).label("avg_duration"),
            func.min(Operation.duration_ms).label("min_duration"),
            func.max(Operation.duration_ms).label("max_duration"),
        ).filter(self._tenant_filter(Operation))
        if since:
            query = query.filter(Operation.created_at >= since)

        results = []
        for row in query.group_by(Operation.op_type).all():
            total, failed = row.total or 0, row.failed or 0
            results.append(
                OperationMetrics(
                    op_type=row.op_type.value if row.op_type else "unknown",
                    total_count=total,
                    success_count=row.success or 0,
                    failed_count=failed,
                    pending_count=row.pending or 0,
                    skipped_count=row.skipped or 0,
                    avg_duration_ms=float(row.avg_duration or 0),
                    min_duration_ms=row.min_duration or 0,
                    max_duration_ms=row.max_duration or 0,
                    error_rate=failed / total if total > 0 else 0,
                )
            )
        return results

    def collect_agent_metrics(self, since: datetime | None = None) -> list[AgentMetrics]:
        """Collect metrics grouped by agent."""
        if since:
            time_range_hours = max(1, (datetime.utcnow() - since).total_seconds() / 3600)
        else:
            first = (
                self.session.query(func.min(AgentRun.started_at))
                .filter(self._tenant_filter(AgentRun))
                .scalar()
            )
            time_range_hours = (
                max(1, (datetime.utcnow() - first).total_seconds() / 3600) if first else 1
            )

        query = self.session.query(
            AgentRun.agent_id,
            func.count(AgentRun.id).label("total"),
            func.sum(case((AgentRun.success == True, 1), else_=0)).label("success"),
            func.sum(case((AgentRun.success == False, 1), else_=0)).label("failed"),
            func.avg(AgentRun.score).label("avg_score"),
            func.avg(AgentRun.duration_ms).label("avg_duration"),
        ).filter(self._tenant_filter(AgentRun))
        if since:
            query = query.filter(AgentRun.started_at >= since)

        results = []
        for row in query.group_by(AgentRun.agent_id).all():
            total, success = row.total or 0, row.success or 0
            results.append(
                AgentMetrics(
                    agent_id=row.agent_id,
                    total_runs=total,
                    successful_runs=success,
                    failed_runs=row.failed or 0,
                    avg_score=float(row.avg_score) if row.avg_score else None,
                    avg_duration_ms=float(row.avg_duration or 0),
                    success_rate=success / total if total > 0 else 0,
                    tasks_per_hour=total / time_range_hours,
                )
            )
        return results


class MetricsReporter:
    """Generate and display metrics reports."""

    def __init__(
        self,
        session: Session,
        tenant_id: str | UUID | None = None,
        time_range_hours: int | None = None,
    ):
        self.session = session
        self.tenant_id = str(tenant_id) if tenant_id else None
        self.time_range_hours = time_range_hours
        self.collector = MetricsCollector(session, tenant_id)

    def collect_all(self) -> MetricsSummary:
        """Collect all metrics."""
        since = (
            datetime.utcnow() - timedelta(hours=self.time_range_hours)
            if self.time_range_hours
            else None
        )
        return MetricsSummary(
            tenant_id=self.tenant_id,
            time_range_hours=self.time_range_hours,
            system=self.collector.collect_system_metrics(since),
            content=self.collector.collect_content_metrics(),
            operations_by_type=self.collector.collect_operation_metrics(since),
            agents=self.collector.collect_agent_metrics(since),
        )

    def print_report(self) -> None:
        """Print formatted metrics report."""
        s = self.collect_all()
        print(f"\n{'='*60}\nPRINCEPS BRAIN LAYER - METRICS REPORT\n{'='*60}")
        print(f"Generated: {s.generated_at.isoformat()}")
        if self.tenant_id:
            print(f"Tenant: {self.tenant_id}")
        if self.time_range_hours:
            print(f"Time Range: Last {self.time_range_hours} hours")

        print(f"\n{'-'*40}\nSYSTEM OVERVIEW\n{'-'*40}")
        sys = s.system
        print(
            f"  Operations: {sys.total_operations:,} total, {sys.operations_last_hour:,} (1h), {sys.operations_last_24h:,} (24h)"
        )
        print(
            f"  Agent Runs: {sys.total_agent_runs:,} total, {sys.runs_last_hour:,} (1h), {sys.runs_last_24h:,} (24h)"
        )
        print(
            f"  Success/Error Rate: {sys.overall_success_rate*100:.1f}% / {sys.overall_error_rate*100:.1f}%"
        )
        print(
            f"  Avg Duration: {sys.avg_operation_duration_ms:.0f}ms (ops), {sys.avg_run_duration_ms:.0f}ms (runs)"
        )

        print(f"\n{'-'*40}\nCONTENT\n{'-'*40}")
        c = s.content
        print(
            f"  Repos: {c.total_repositories:,} | Resources: {c.total_resources:,} | Docs: {c.total_documents:,} | Chunks: {c.total_chunks:,}"
        )
        print(
            f"  Knowledge: {c.total_entities:,} entities, {c.total_topics:,} topics, {c.total_concepts:,} concepts"
        )
        print(
            f"  Security: {c.pii_flagged_count:,} PII flagged, {c.secrets_flagged_count:,} secrets flagged"
        )

        if s.operations_by_type:
            print(f"\n{'-'*40}\nOPERATIONS BY TYPE\n{'-'*40}")
            for op in s.operations_by_type:
                print(
                    f"  {op.op_type}: {op.total_count} total, {op.success_count} success, {op.error_rate*100:.1f}% errors"
                )

        if s.agents:
            print(f"\n{'-'*40}\nAGENTS\n{'-'*40}")
            for a in s.agents:
                print(
                    f"  {a.agent_id}: {a.total_runs} runs, {a.success_rate*100:.1f}% success, {a.tasks_per_hour:.1f}/hour"
                )
        print(f"\n{'='*60}\n")

    def export_json(self, indent: int = 2) -> str:
        return self.collect_all().to_json(indent=indent)

    def export_dict(self) -> dict[str, Any]:
        return self.collect_all().to_dict()


def get_metrics_summary(
    session: Session, tenant_id: str | UUID | None = None, time_range_hours: int | None = None
) -> MetricsSummary:
    """Get a quick metrics summary."""
    return MetricsReporter(session, tenant_id, time_range_hours).collect_all()


def print_metrics_report(
    session: Session, tenant_id: str | UUID | None = None, time_range_hours: int | None = None
) -> None:
    """Print a formatted metrics report."""
    MetricsReporter(session, tenant_id, time_range_hours).print_report()
