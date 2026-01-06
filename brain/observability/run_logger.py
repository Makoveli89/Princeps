"""
Princeps Brain Layer - Agent Run Logger
========================================

Records agent runs to the agent_runs table with lifecycle tracking,
correlation ID support, and result serialization.

Usage:
    from brain.observability import AgentRunLogger, log_agent_run

    with AgentRunLogger(session, tenant_id, "summarizer", "Summarize doc xyz") as run:
        result = do_work()
        run.set_success(solution=result, score=0.95)
"""

import functools
import hashlib
import json
import traceback as tb
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, TypeVar
from uuid import UUID

from sqlalchemy.orm import Session

from ..core.db import get_default_tenant_id
from ..core.models import AgentRun
from .logging_config import (
    OperationContext,
    generate_correlation_id,
    get_correlation_id,
    get_logger,
)

logger = get_logger(__name__)
F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class RunLoggerConfig:
    """Configuration for agent run logging."""

    default_max_retries: int = 3
    store_full_traceback: bool = True
    max_solution_size_kb: int = 100
    capture_tools_used: bool = True
    capture_model_version: bool = True
    auto_generate_task_hash: bool = True
    log_errors_to_console: bool = True
    raise_on_db_error: bool = False


DEFAULT_CONFIG = RunLoggerConfig()


@dataclass
class AgentRunResult:
    """Result of an agent run."""

    run_id: str
    success: bool
    agent_id: str
    task: str
    duration_ms: int
    score: float | None = None
    solution: dict[str, Any] | None = None
    feedback: str | None = None
    error_message: str | None = None
    correlation_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def compute_task_hash(task: str) -> str:
    """Compute deterministic hash of task description."""
    return hashlib.sha256(task.strip().lower().encode()).hexdigest()


def truncate_solution(solution: Any, max_size_kb: int = 100) -> dict[str, Any]:
    """Ensure solution JSON doesn't exceed max size."""
    if solution is None:
        return {}
    if not isinstance(solution, dict):
        solution = {"value": solution}
    json_str = json.dumps(solution, default=str)
    max_bytes = max_size_kb * 1024
    if len(json_str) <= max_bytes:
        return solution
    return {
        "_truncated": True,
        "_original_size_kb": len(json_str) // 1024,
        "preview": json_str[:1000] + "...",
    }


def ensure_uuid(value: str | UUID | None) -> UUID | None:
    """Convert string to UUID if needed."""
    if value is None:
        return None
    return value if isinstance(value, UUID) else UUID(str(value))


class AgentRunLogger:
    """
    Context manager for logging agent runs to the database.

    Usage:
        with AgentRunLogger(session, tenant_id, "my_agent", "Process document", context={"doc_id": "123"}) as run:
            result = process_document()
            run.set_success(solution=result, score=0.9)
    """

    def __init__(
        self,
        session: Session,
        tenant_id: str | UUID,
        agent_id: str,
        task: str,
        correlation_id: str | None = None,
        context: dict[str, Any] | None = None,
        tools_used: list[str] | None = None,
        model_version: str | None = None,
        config: RunLoggerConfig | None = None,
    ):
        self.session = session
        self.tenant_id = ensure_uuid(tenant_id)
        self.agent_id = agent_id
        self.task = task
        self.correlation_id = correlation_id or get_correlation_id() or generate_correlation_id()
        self.context = context or {}
        self.tools_used = tools_used or []
        self.model_version = model_version
        self.config = config or DEFAULT_CONFIG

        self.run_id: UUID | None = None
        self.run: AgentRun | None = None
        self.start_time: datetime | None = None
        self._success: bool | None = None
        self._score: float | None = None
        self._solution: dict[str, Any] | None = None
        self._feedback: str | None = None
        self._error_message: str | None = None
        self._operation_context: OperationContext | None = None

    def __enter__(self) -> "AgentRunLogger":
        self.start_time = datetime.utcnow()
        task_hash = compute_task_hash(self.task) if self.config.auto_generate_task_hash else None

        self.run = AgentRun(
            tenant_id=self.tenant_id,
            agent_id=self.agent_id,
            task=self.task,
            task_hash=task_hash,
            success=False,
            started_at=self.start_time,
            context=self.context,
            tools_used=self.tools_used,
            model_version=self.model_version,
            metadata={"correlation_id": self.correlation_id},
        )
        self.session.add(self.run)
        self.session.flush()
        self.run_id = self.run.id

        self._operation_context = OperationContext(
            correlation_id=self.correlation_id,
            operation_id=str(self.run_id),
            agent_id=self.agent_id,
        )
        self._operation_context.__enter__()

        logger.info(
            f"Agent run started: {self.agent_id}",
            extra={
                "extra_data": {
                    "event": "agent_run_start",
                    "run_id": str(self.run_id),
                    "agent_id": self.agent_id,
                }
            },
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.utcnow()
        duration_ms = int((end_time - self.start_time).total_seconds() * 1000)

        if exc_type is not None:
            self._success = False
            self._error_message = str(exc_val)
            if self.config.store_full_traceback:
                self.context["traceback"] = tb.format_exception(exc_type, exc_val, exc_tb)
            if self.config.log_errors_to_console:
                logger.error(
                    f"Agent run failed: {self.agent_id} - {exc_val}",
                    exc_info=(exc_type, exc_val, exc_tb),
                )
        else:
            if self._success is None:
                self._success = True
            logger.info(
                f"Agent run completed: {self.agent_id} (success={self._success}, duration={duration_ms}ms)"
            )

        if self.run:
            self.run.success = self._success
            self.run.completed_at = end_time
            self.run.duration_ms = duration_ms
            self.run.score = self._score
            self.run.solution = truncate_solution(self._solution, self.config.max_solution_size_kb)
            self.run.feedback = self._feedback
            self.run.context = self.context
            self.run.tools_used = self.tools_used
            if self._error_message:
                self.run.metadata = {
                    **(self.run.metadata or {}),
                    "error_message": self._error_message,
                }
            try:
                self.session.commit()
            except Exception as e:
                logger.error(f"Failed to commit agent run: {e}")
                self.session.rollback()
                if self.config.raise_on_db_error:
                    raise

        if self._operation_context:
            self._operation_context.__exit__(exc_type, exc_val, exc_tb)
        return False

    def set_success(
        self, solution: Any | None = None, score: float | None = None, feedback: str | None = None
    ):
        self._success, self._solution, self._score, self._feedback = True, solution, score, feedback

    def set_failure(
        self, error_message: str, score: float | None = None, feedback: str | None = None
    ):
        self._success, self._error_message, self._score, self._feedback = (
            False,
            error_message,
            score,
            feedback,
        )

    def set_result(
        self,
        solution: Any,
        success: bool = True,
        score: float | None = None,
        feedback: str | None = None,
    ):
        self._success, self._solution, self._score, self._feedback = (
            success,
            solution,
            score,
            feedback,
        )

    def add_tool_used(self, tool_name: str):
        if tool_name not in self.tools_used:
            self.tools_used.append(tool_name)

    def add_context(self, key: str, value: Any):
        self.context[key] = value

    def get_result(self) -> AgentRunResult:
        duration_ms = (
            int((datetime.utcnow() - self.start_time).total_seconds() * 1000)
            if self.start_time
            else 0
        )
        return AgentRunResult(
            run_id=str(self.run_id) if self.run_id else "",
            success=self._success or False,
            agent_id=self.agent_id,
            task=self.task,
            duration_ms=duration_ms,
            score=self._score,
            solution=self._solution,
            feedback=self._feedback,
            error_message=self._error_message,
            correlation_id=self.correlation_id,
        )


def log_agent_run(
    agent_id: str,
    task_param: str = "task",
    session_param: str = "session",
    tenant_id_param: str = "tenant_id",
    extract_score: Callable[[Any], float] | None = None,
) -> Callable[[F], F]:
    """Decorator to automatically log agent runs."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import inspect

            session = kwargs.get(session_param)
            tenant_id = kwargs.get(tenant_id_param)
            task = kwargs.get(task_param, func.__name__)

            if session is None:
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                if session_param in params:
                    idx = params.index(session_param)
                    if idx < len(args):
                        session = args[idx]

            if session is None:
                logger.warning(f"No session available for agent run logging: {agent_id}")
                return func(*args, **kwargs)

            if tenant_id is None:
                tenant_id = get_default_tenant_id(session)

            context = {
                k: v
                for k, v in kwargs.items()
                if k not in [session_param, tenant_id_param, task_param] and not k.startswith("_")
            }

            with AgentRunLogger(
                session=session,
                tenant_id=tenant_id,
                agent_id=agent_id,
                task=str(task),
                context=context,
            ) as run_logger:
                result = func(*args, **kwargs)
                score = extract_score(result) if extract_score else None
                run_logger.set_success(solution=result, score=score)
                return result

        return wrapper

    return decorator


def get_agent_runs(
    session: Session,
    tenant_id: str | UUID,
    agent_id: str | None = None,
    success: bool | None = None,
    limit: int = 100,
    since: datetime | None = None,
) -> list[AgentRun]:
    """Query agent runs with optional filters."""
    query = session.query(AgentRun).filter(AgentRun.tenant_id == ensure_uuid(tenant_id))
    if agent_id:
        query = query.filter(AgentRun.agent_id == agent_id)
    if success is not None:
        query = query.filter(AgentRun.success == success)
    if since:
        query = query.filter(AgentRun.started_at >= since)
    return query.order_by(AgentRun.started_at.desc()).limit(limit).all()


def get_similar_runs(
    session: Session, tenant_id: str | UUID, task: str, limit: int = 5
) -> list[AgentRun]:
    """Find similar past runs by task hash."""
    task_hash = compute_task_hash(task)
    return (
        session.query(AgentRun)
        .filter(AgentRun.tenant_id == ensure_uuid(tenant_id), AgentRun.task_hash == task_hash)
        .order_by(AgentRun.started_at.desc())
        .limit(limit)
        .all()
    )
