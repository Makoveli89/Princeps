"""
Princeps Brain Layer - Idempotency Service
===========================================

Guarantees repeatable operations via input-hash based de-duplication.

Usage:
    from brain.resilience import IdempotencyManager, IdempotencyConfig

    manager = IdempotencyManager(session, tenant_id)
    with manager.operation_scope(op_type, inputs) as scope:
        if scope.was_skipped:
            return scope.cached_result
        result = do_work()
        scope.set_result(result)
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from ..core.db import (
    get_or_create_operation,
    mark_operation_failed,
    mark_operation_started,
    mark_operation_success,
)
from ..core.models import Operation, OperationStatusEnum, OperationTypeEnum
from ..observability.logging_config import (
    OperationContext,
    generate_correlation_id,
    get_correlation_id,
    get_logger,
)

logger = get_logger(__name__)


@dataclass
class IdempotencyConfig:
    """Configuration for idempotency management."""

    normalize_paths: bool = True
    normalize_case: bool = False
    exclude_fields: list[str] = field(
        default_factory=lambda: ["timestamp", "correlation_id", "request_id"]
    )
    skip_on_success: bool = True
    skip_on_in_progress: bool = True
    skip_on_pending: bool = False
    stale_in_progress_minutes: int = 60
    auto_retry_stale: bool = True
    log_skips: bool = True
    log_hash_details: bool = False


DEFAULT_CONFIG = IdempotencyConfig()


@dataclass
class IdempotencyCheckResult:
    """Result of an idempotency check."""

    should_run: bool
    existing_operation: Operation | None = None
    cached_result: dict[str, Any] | None = None
    skip_reason: str | None = None
    computed_hash: str | None = None

    @property
    def was_skipped(self) -> bool:
        return not self.should_run


@dataclass
class OperationResult:
    """Result of an idempotent operation execution."""

    success: bool
    operation_id: str
    was_cached: bool = False
    outputs: dict[str, Any] | None = None
    error_message: str | None = None
    duration_ms: int = 0
    skip_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def normalize_input_value(value: Any, config: IdempotencyConfig) -> Any:
    """Normalize a single input value for consistent hashing."""
    if value is None:
        return None
    if isinstance(value, str):
        if config.normalize_paths and ("/" in value or "\\" in value):
            return value.replace("\\", "/").rstrip("/")
        return value.lower() if config.normalize_case else value
    if isinstance(value, (list, tuple)):
        return [normalize_input_value(v, config) for v in value]
    if isinstance(value, dict):
        return {k: normalize_input_value(v, config) for k, v in value.items()}
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def compute_input_hash(
    op_type: str | OperationTypeEnum,
    inputs: dict[str, Any],
    config: IdempotencyConfig | None = None,
) -> str:
    """Compute deterministic hash of operation inputs for idempotency."""
    config = config or DEFAULT_CONFIG
    op_type_str = op_type.value if isinstance(op_type, OperationTypeEnum) else op_type
    filtered = {k: v for k, v in inputs.items() if k not in config.exclude_fields}
    normalized = json.dumps(
        {"op_type": op_type_str, "inputs": normalize_input_value(filtered, config)},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(normalized.encode()).hexdigest()


class IdempotencyManager:
    """Manages idempotent operation execution."""

    def __init__(
        self, session: Session, tenant_id: str | UUID, config: IdempotencyConfig | None = None
    ):
        self.session = session
        self.tenant_id = str(tenant_id) if isinstance(tenant_id, UUID) else tenant_id
        self.config = config or DEFAULT_CONFIG

    def check_operation(
        self, op_type: OperationTypeEnum, inputs: dict[str, Any]
    ) -> IdempotencyCheckResult:
        """Check if an operation should run or return cached result."""
        input_hash = compute_input_hash(op_type, inputs, self.config)
        existing = (
            self.session.query(Operation)
            .filter(Operation.op_type == op_type, Operation.input_hash == input_hash)
            .first()
        )

        if existing is None:
            return IdempotencyCheckResult(should_run=True, computed_hash=input_hash)

        status = existing.status

        if status == OperationStatusEnum.SUCCESS and self.config.skip_on_success:
            if self.config.log_skips:
                logger.info(f"Skipping {op_type.value} - already succeeded: {existing.id}")
            return IdempotencyCheckResult(
                should_run=False,
                existing_operation=existing,
                cached_result=existing.outputs,
                skip_reason="Already completed successfully",
                computed_hash=input_hash,
            )

        if status == OperationStatusEnum.IN_PROGRESS:
            if self._is_operation_stale(existing) and self.config.auto_retry_stale:
                existing.status = OperationStatusEnum.FAILED
                existing.error_message = "Marked stale - exceeded timeout"
                self.session.flush()
                logger.warning(f"Marking stale operation as failed: {existing.id}")
                return IdempotencyCheckResult(
                    should_run=True, existing_operation=existing, computed_hash=input_hash
                )
            if self.config.skip_on_in_progress:
                return IdempotencyCheckResult(
                    should_run=False,
                    existing_operation=existing,
                    skip_reason="Already in progress",
                    computed_hash=input_hash,
                )

        if status == OperationStatusEnum.PENDING and self.config.skip_on_pending:
            return IdempotencyCheckResult(
                should_run=False,
                existing_operation=existing,
                skip_reason="Operation pending",
                computed_hash=input_hash,
            )

        return IdempotencyCheckResult(
            should_run=True, existing_operation=existing, computed_hash=input_hash
        )

    def should_run_operation(
        self, op_type: OperationTypeEnum, inputs: dict[str, Any]
    ) -> tuple[bool, Operation | None]:
        """Simple check returning (should_run, existing_operation)."""
        result = self.check_operation(op_type, inputs)
        return result.should_run, result.existing_operation

    def create_operation(
        self,
        op_type: OperationTypeEnum,
        inputs: dict[str, Any],
        correlation_id: str | None = None,
        agent_id: str | None = None,
        **kwargs,
    ) -> tuple[Operation, bool]:
        """Create a new operation record (or get existing)."""
        operation, created = get_or_create_operation(
            session=self.session,
            tenant_id=self.tenant_id,
            op_type=op_type,
            inputs=inputs,
            correlation_id=correlation_id or get_correlation_id(),
            agent_id=agent_id,
            **kwargs,
        )
        if created:
            logger.info(f"Created operation: {op_type.value} ({operation.id})")
        return operation, created

    def start_operation(self, operation_id: str | UUID) -> None:
        mark_operation_started(self.session, str(operation_id))

    def complete_operation(
        self,
        operation_id: str | UUID,
        outputs: dict[str, Any] | None = None,
        success: bool = True,
        error_message: str | None = None,
    ) -> None:
        if success:
            mark_operation_success(self.session, str(operation_id), outputs)
        else:
            mark_operation_failed(self.session, str(operation_id), error_message or "Unknown error")

    def operation_scope(
        self,
        op_type: OperationTypeEnum,
        inputs: dict[str, Any],
        correlation_id: str | None = None,
        agent_id: str | None = None,
    ) -> "IdempotentOperationScope":
        """Create a context manager for idempotent operation execution."""
        return IdempotentOperationScope(self, op_type, inputs, correlation_id, agent_id)

    def _is_operation_stale(self, operation: Operation) -> bool:
        if operation.started_at is None:
            return True
        return (datetime.utcnow() - operation.started_at).total_seconds() > (
            self.config.stale_in_progress_minutes * 60
        )

    def get_cached_result(
        self, op_type: OperationTypeEnum, inputs: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Get cached result for an operation if it exists and succeeded."""
        input_hash = compute_input_hash(op_type, inputs, self.config)
        op = (
            self.session.query(Operation)
            .filter(
                Operation.op_type == op_type,
                Operation.input_hash == input_hash,
                Operation.status == OperationStatusEnum.SUCCESS,
            )
            .first()
        )
        return op.outputs if op else None

    def invalidate_operation(self, op_type: OperationTypeEnum, inputs: dict[str, Any]) -> bool:
        """Invalidate a cached operation to force re-execution."""
        input_hash = compute_input_hash(op_type, inputs, self.config)
        result = (
            self.session.query(Operation)
            .filter(Operation.op_type == op_type, Operation.input_hash == input_hash)
            .update({"status": OperationStatusEnum.FAILED, "error_message": "Manually invalidated"})
        )
        if result > 0:
            logger.info(f"Invalidated operation: {op_type.value}")
        return result > 0


class IdempotentOperationScope:
    """Context manager for idempotent operation execution."""

    def __init__(
        self,
        manager: IdempotencyManager,
        op_type: OperationTypeEnum,
        inputs: dict[str, Any],
        correlation_id: str | None = None,
        agent_id: str | None = None,
    ):
        self.manager = manager
        self.op_type = op_type
        self.inputs = inputs
        self.correlation_id = correlation_id or get_correlation_id() or generate_correlation_id()
        self.agent_id = agent_id

        self.operation: Operation | None = None
        self.was_skipped: bool = False
        self.cached_result: dict[str, Any] | None = None
        self.skip_reason: str | None = None
        self._result: dict[str, Any] | None = None
        self._success: bool = True
        self._error_message: str | None = None
        self._operation_context: OperationContext | None = None
        self._start_time: datetime | None = None

    def __enter__(self) -> "IdempotentOperationScope":
        self._start_time = datetime.utcnow()
        check = self.manager.check_operation(self.op_type, self.inputs)

        if not check.should_run:
            self.was_skipped = True
            self.cached_result = check.cached_result
            self.skip_reason = check.skip_reason
            self.operation = check.existing_operation
            return self

        self.operation, _ = self.manager.create_operation(
            self.op_type, self.inputs, self.correlation_id, self.agent_id
        )
        self.manager.start_operation(self.operation.id)
        self._operation_context = OperationContext(
            correlation_id=self.correlation_id,
            operation_id=str(self.operation.id),
            agent_id=self.agent_id,
        )
        self._operation_context.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._operation_context:
            self._operation_context.__exit__(exc_type, exc_val, exc_tb)
        if self.was_skipped:
            return False

        if exc_type is not None:
            self.manager.complete_operation(
                self.operation.id, success=False, error_message=str(exc_val)
            )
        elif not self._success:
            self.manager.complete_operation(
                self.operation.id, success=False, error_message=self._error_message
            )
        else:
            self.manager.complete_operation(self.operation.id, outputs=self._result, success=True)
        return False

    def set_result(self, result: Any, success: bool = True) -> None:
        self._result = result if isinstance(result, dict) else {"result": result}
        self._success = success

    def set_failure(self, error_message: str) -> None:
        self._success = False
        self._error_message = error_message


# Convenience functions for backward compatibility
def check_idempotency(
    session: Session, tenant_id: str | UUID, op_type: OperationTypeEnum, inputs: dict[str, Any]
) -> tuple[Operation | None, bool]:
    """Check idempotency and return (existing_operation, should_run)."""
    manager = IdempotencyManager(session, tenant_id)
    result = manager.check_operation(op_type, inputs)
    return result.existing_operation, result.should_run


def mark_complete(
    session: Session, operation_id: str | UUID, outputs: dict[str, Any] | None = None
) -> None:
    """Mark an operation as complete."""
    mark_operation_success(session, str(operation_id), outputs)
