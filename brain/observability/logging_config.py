"""
Princeps Brain Layer - Logging Configuration
=============================================

Comprehensive logging framework with structured JSON logging,
correlation ID propagation, and context managers for operation-scoped logging.

Usage:
    from brain.observability import setup_logging, get_logger, OperationContext
    
    setup_logging(level="INFO", json_format=True)
    logger = get_logger(__name__)
    
    with OperationContext(correlation_id="req-123", operation_id="op-456"):
        logger.info("Processing started")
"""

import contextvars
import json
import logging
import sys
import traceback as tb
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

# Context variables for request tracing
_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar('correlation_id', default=None)
_operation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar('operation_id', default=None)
_agent_id: contextvars.ContextVar[str | None] = contextvars.ContextVar('agent_id', default=None)
_tenant_id: contextvars.ContextVar[str | None] = contextvars.ContextVar('tenant_id', default=None)
_extra_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar('extra_context', default={})

def get_correlation_id() -> str | None:
    return _correlation_id.get()

def set_correlation_id(correlation_id: str | None) -> contextvars.Token:
    return _correlation_id.set(correlation_id)

def get_operation_id() -> str | None:
    return _operation_id.get()

def set_operation_id(operation_id: str | None) -> contextvars.Token:
    return _operation_id.set(operation_id)

def get_agent_id() -> str | None:
    return _agent_id.get()

def set_agent_id(agent_id: str | None) -> contextvars.Token:
    return _agent_id.set(agent_id)

def get_tenant_id() -> str | None:
    return _tenant_id.get()

def set_tenant_id(tenant_id: str | None) -> contextvars.Token:
    return _tenant_id.set(tenant_id)

def generate_correlation_id() -> str:
    return f"corr-{uuid4().hex[:12]}"

def generate_operation_id() -> str:
    return f"op-{uuid4().hex[:12]}"


class OperationContext:
    """Context manager for operation-scoped logging with automatic ID propagation."""

    def __init__(self, correlation_id: str | None = None, operation_id: str | None = None,
                 agent_id: str | None = None, tenant_id: str | None = None,
                 auto_generate_correlation: bool = False, auto_generate_operation: bool = False, **extra_context):
        self.correlation_id = correlation_id
        self.operation_id = operation_id
        self.agent_id = agent_id
        self.tenant_id = tenant_id
        self.auto_generate_correlation = auto_generate_correlation
        self.auto_generate_operation = auto_generate_operation
        self.extra_context = extra_context
        self._tokens: list[contextvars.Token] = []

    def __enter__(self):
        if self.correlation_id:
            self._tokens.append(set_correlation_id(self.correlation_id))
        elif self.auto_generate_correlation and not get_correlation_id():
            self._tokens.append(set_correlation_id(generate_correlation_id()))
        if self.operation_id:
            self._tokens.append(set_operation_id(self.operation_id))
        elif self.auto_generate_operation:
            self._tokens.append(set_operation_id(generate_operation_id()))
        if self.agent_id:
            self._tokens.append(set_agent_id(self.agent_id))
        if self.tenant_id:
            self._tokens.append(set_tenant_id(self.tenant_id))
        if self.extra_context:
            current = _extra_context.get().copy()
            current.update(self.extra_context)
            self._tokens.append(_extra_context.set(current))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for token in reversed(self._tokens):
            token.var.reset(token)
        return False


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def __init__(self, include_timestamp: bool = True, include_location: bool = True,
                 include_context: bool = True, extra_fields: dict[str, Any] | None = None):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_location = include_location
        self.include_context = include_context
        self.extra_fields = extra_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {"level": record.levelname, "message": record.getMessage(), "logger": record.name}
        if self.include_timestamp:
            log_obj["timestamp"] = datetime.utcnow().isoformat() + "Z"
        if self.include_location:
            log_obj["location"] = {"file": record.filename, "line": record.lineno, "function": record.funcName}
        if self.include_context:
            if corr_id := get_correlation_id(): log_obj["correlation_id"] = corr_id
            if op_id := get_operation_id(): log_obj["operation_id"] = op_id
            if agent_id := get_agent_id(): log_obj["agent_id"] = agent_id
            if tenant_id := get_tenant_id(): log_obj["tenant_id"] = tenant_id
            if extra := _extra_context.get(): log_obj["context"] = extra
        if record.exc_info:
            log_obj["exception"] = {"type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                                    "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                                    "traceback": tb.format_exception(*record.exc_info) if record.exc_info[0] else None}
        if hasattr(record, 'extra_data'):
            log_obj["data"] = record.extra_data
        log_obj.update(self.extra_fields)
        return json.dumps(log_obj, default=str)


class ContextFormatter(logging.Formatter):
    """Human-readable formatter with context IDs."""
    DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(context)s%(name)s - %(message)s"

    def __init__(self, fmt: str | None = None, datefmt: str | None = None):
        super().__init__(fmt=fmt or self.DEFAULT_FORMAT, datefmt=datefmt or "%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        parts = []
        if corr_id := get_correlation_id(): parts.append(f"[{corr_id}]")
        if op_id := get_operation_id(): parts.append(f"[{op_id}]")
        if agent_id := get_agent_id(): parts.append(f"[{agent_id}]")
        record.context = " ".join(parts) + " " if parts else ""
        return super().format(record)


def setup_logging(level: str = "INFO", json_format: bool = False, log_to_console: bool = True,
                  log_to_file: bool = False, log_file_path: str | None = None,
                  max_file_size_mb: int = 10, backup_count: int = 5, include_location: bool = True,
                  extra_fields: dict[str, Any] | None = None) -> None:
    """Setup logging with structured formatters."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    root_logger.handlers.clear()

    formatter = StructuredFormatter(include_location=include_location, extra_fields=extra_fields or {}) if json_format else ContextFormatter()

    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if log_to_file:
        from logging.handlers import RotatingFileHandler
        log_path = Path(log_file_path or "./logs/princeps.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(log_path, maxBytes=max_file_size_mb * 1024 * 1024, backupCount=backup_count)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger."""
    return logging.getLogger(name)


class LogContext:
    """Helper for adding extra data to log records."""
    def __init__(self, **kwargs):
        self.extra = {"extra_data": kwargs}


class OperationLogger:
    """Context manager that logs operation lifecycle events."""

    def __init__(self, logger: logging.Logger, operation_name: str, correlation_id: str | None = None,
                 operation_id: str | None = None, agent_id: str | None = None, **context_data):
        self.logger = logger
        self.operation_name = operation_name
        self.correlation_id = correlation_id or get_correlation_id() or generate_correlation_id()
        self.operation_id = operation_id or generate_operation_id()
        self.agent_id = agent_id
        self.context_data = context_data
        self.start_time: datetime | None = None
        self._context: OperationContext | None = None

    def __enter__(self):
        self.start_time = datetime.utcnow()
        self._context = OperationContext(correlation_id=self.correlation_id, operation_id=self.operation_id,
                                          agent_id=self.agent_id, **self.context_data)
        self._context.__enter__()
        self.logger.info(f"Starting operation: {self.operation_name}",
                        extra={"extra_data": {"event": "operation_start", "operation": self.operation_name, **self.context_data}})
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = int((datetime.utcnow() - self.start_time).total_seconds() * 1000)
        if exc_type is None:
            self.logger.info(f"Completed operation: {self.operation_name} (duration: {duration_ms}ms)",
                            extra={"extra_data": {"event": "operation_success", "operation": self.operation_name, "duration_ms": duration_ms}})
        else:
            self.logger.error(f"Failed operation: {self.operation_name} (duration: {duration_ms}ms) - {exc_val}",
                             exc_info=(exc_type, exc_val, exc_tb),
                             extra={"extra_data": {"event": "operation_failed", "operation": self.operation_name, "duration_ms": duration_ms}})
        if self._context:
            self._context.__exit__(exc_type, exc_val, exc_tb)
        return False

    @property
    def duration_ms(self) -> int:
        return int((datetime.utcnow() - self.start_time).total_seconds() * 1000) if self.start_time else 0


def log_exception(logger: logging.Logger, message: str, exception: Exception | None = None, **kwargs) -> None:
    """Log an exception with context."""
    extra = {"extra_data": kwargs} if kwargs else {}
    if exception:
        logger.error(message, exc_info=(type(exception), exception, exception.__traceback__), extra=extra)
    else:
        logger.error(message, exc_info=True, extra=extra)
