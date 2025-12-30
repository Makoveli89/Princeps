"""
Brain Resilience Module
=======================

Fault tolerance, retries, and idempotency.
"""

from .batch_utils import (
    BatchConfig,
    BatchProcessor,
    BatchResult,
    process_in_batches,
)
from .error_handler import (
    BrainError,
    DatabaseError,
    DistillationError,
    ErrorHandler,
    IngestionError,
    handle_errors,
)
from .idempotency_service import (
    IdempotencyCheckResult,
    IdempotencyConfig,
    IdempotencyManager,
    IdempotentOperationScope,
    OperationResult,
    check_idempotency,
    compute_input_hash,
    mark_complete,
)
from .retry_manager import (
    ExponentialBackoff,
    RetryConfig,
    RetryManager,
    retry_with_backoff,
)

__all__ = [
    # Idempotency
    "IdempotencyManager",
    "IdempotencyConfig",
    "IdempotentOperationScope",
    "IdempotencyCheckResult",
    "OperationResult",
    "check_idempotency",
    "mark_complete",
    "compute_input_hash",
    # Error Handling
    "ErrorHandler",
    "BrainError",
    "IngestionError",
    "DistillationError",
    "DatabaseError",
    "handle_errors",
    # Retry
    "RetryManager",
    "RetryConfig",
    "retry_with_backoff",
    "ExponentialBackoff",
    # Batch
    "BatchProcessor",
    "BatchConfig",
    "BatchResult",
    "process_in_batches",
]
