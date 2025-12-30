"""
Brain Observability Module
==========================

Logging, metrics, and run tracking.
"""

from .logging_config import (
    ContextFormatter,
    LogContext,
    OperationContext,
    OperationLogger,
    StructuredFormatter,
    generate_correlation_id,
    generate_operation_id,
    get_agent_id,
    get_correlation_id,
    get_logger,
    get_operation_id,
    get_tenant_id,
    log_exception,
    set_agent_id,
    set_correlation_id,
    set_operation_id,
    set_tenant_id,
    setup_logging,
)
from .metrics_reporter import (
    AgentMetrics,
    ContentMetrics,
    MetricsCollector,
    MetricsReporter,
    MetricsSummary,
    OperationMetrics,
    SystemMetrics,
    get_metrics_summary,
    print_metrics_report,
)
from .run_logger import (
    AgentRunLogger,
    AgentRunResult,
    RunLoggerConfig,
    get_agent_runs,
    get_similar_runs,
    log_agent_run,
)

__all__ = [
    # Logging
    "setup_logging",
    "get_logger",
    "OperationContext",
    "OperationLogger",
    "LogContext",
    "get_correlation_id",
    "set_correlation_id",
    "get_operation_id",
    "set_operation_id",
    "get_agent_id",
    "set_agent_id",
    "get_tenant_id",
    "set_tenant_id",
    "generate_correlation_id",
    "generate_operation_id",
    "StructuredFormatter",
    "ContextFormatter",
    "log_exception",
    # Run Logger
    "AgentRunLogger",
    "log_agent_run",
    "AgentRunResult",
    "RunLoggerConfig",
    "get_agent_runs",
    "get_similar_runs",
    # Metrics
    "MetricsReporter",
    "MetricsCollector",
    "MetricsSummary",
    "SystemMetrics",
    "ContentMetrics",
    "OperationMetrics",
    "AgentMetrics",
    "get_metrics_summary",
    "print_metrics_report",
]
