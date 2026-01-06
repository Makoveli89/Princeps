"""
Brain Logger - Database Logging Integration for Agent System

This module provides the BrainLogger class that agents use to record their
invocations to the Brain's database (agent_runs table). It ensures:
- Every agent invocation is tracked with unique run IDs
- All significant actions are logged (inputs, outputs, errors)
- Model usage and failover events are recorded
- Tenant isolation is maintained via tenant_id
- PII scan results are stored for audit

Integration with BaseAgent:
- BaseAgent creates a BrainLogger instance or uses a shared one
- On task start: logger.start_run() creates AgentRun record
- During execution: logger.log_event() records sub-operations
- On completion: logger.complete_run() finalizes the record

The logger uses the SupabaseClient from brain_layer for database operations,
falling back to local logging if database is unavailable.
"""

import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Local imports
from framework.agents.schemas.agent_run import (
    AgentRunRecord,
    AgentRunUpdate,
    ModelUsageRecord,
    PIIScanResult,
    RunStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class LoggerConfig:
    """Configuration for BrainLogger"""

    # Database settings
    enable_db_logging: bool = True
    db_table_name: str = "agent_runs"

    # Fallback file logging
    enable_file_logging: bool = True
    log_file_path: str | None = None
    log_dir: str = "logs/agent_runs"

    # What to log
    log_inputs: bool = True
    log_outputs: bool = True
    log_prompts: bool = False  # May contain sensitive data
    log_model_usage: bool = True
    log_pii_scans: bool = True

    # Performance
    batch_size: int = 10
    flush_interval_seconds: float = 5.0

    # Retention
    max_log_age_days: int = 90


class BrainLogger:
    """
    Logger for recording agent runs to the Brain database.

    Provides methods to:
    - Start a new run record (start_run)
    - Log intermediate events (log_event)
    - Record model usage (log_model_usage)
    - Complete a run with results (complete_run)
    - Query historical runs (get_runs)

    Usage:
        logger = BrainLogger(supabase_client=client)

        run_id = logger.start_run(
            agent_id="summarizer_v1",
            agent_name="Summarization Agent",
            agent_type="summarization",
            tenant_id="tenant_123",
            input_data={"prompt": "Summarize this..."}
        )

        logger.log_event(run_id, "llm_call", {"model": "claude-3"})

        logger.complete_run(
            run_id,
            success=True,
            output_data={"summary": "..."},
            tokens_used=1500
        )
    """

    def __init__(
        self,
        supabase_client=None,
        config: LoggerConfig | None = None,
    ):
        """
        Initialize the BrainLogger.

        Args:
            supabase_client: SupabaseClient instance for database operations
            config: Logger configuration
        """
        self.config = config or LoggerConfig()
        self._supabase = supabase_client
        self._db_available = False

        # In-memory cache for active runs
        self._active_runs: dict[str, AgentRunRecord] = {}

        # Event buffer for batching
        self._event_buffer: list[dict[str, Any]] = []

        # Callbacks for external integrations
        self._callbacks: list[Callable[[dict[str, Any]], None]] = []

        # Initialize database connection
        self._init_database()

        # Ensure log directory exists for file fallback
        if self.config.enable_file_logging:
            log_dir = Path(self.config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

        logger.info("BrainLogger initialized")

    def _init_database(self):
        """Initialize database connection"""
        if not self.config.enable_db_logging:
            logger.info("Database logging disabled")
            return

        if self._supabase is None:
            # Try to create client from environment
            try:
                from brain_layer.supabase_pgvector.supabase_client import create_supabase_client

                self._supabase = create_supabase_client()
                self._db_available = True
                logger.info("Database connection established")
            except Exception as e:
                logger.warning(f"Could not connect to database: {e}")
                logger.info("Falling back to file-based logging")
                self._db_available = False
        else:
            self._db_available = True

    def add_callback(self, callback: Callable[[dict[str, Any]], None]):
        """Add a callback for log events"""
        self._callbacks.append(callback)

    def _notify_callbacks(self, event: dict[str, Any]):
        """Notify all registered callbacks"""
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.warning(f"Callback failed: {e}")

    def start_run(
        self,
        agent_id: str,
        agent_name: str,
        agent_type: str,
        tenant_id: str | None = None,
        task_id: str | None = None,
        input_data: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        """
        Start a new agent run and create a record.

        Args:
            agent_id: Unique identifier for the agent
            agent_name: Human-readable agent name
            agent_type: Type classification (summarization, code_gen, etc.)
            tenant_id: Tenant ID for multi-tenant isolation
            task_id: Parent task ID if applicable
            input_data: Task input (prompt, parameters, etc.)
            context: Additional context (routing preferences, etc.)

        Returns:
            run_id: Unique identifier for this run
        """
        run_id = str(uuid.uuid4())

        # Create run record
        run_record = AgentRunRecord(
            id=run_id,
            task_id=task_id,
            agent_id=agent_id,
            agent_name=agent_name,
            agent_type=agent_type,
            tenant_id=tenant_id,
            started_at=datetime.utcnow(),
            status=RunStatus.IN_PROGRESS,
            input_data=input_data or {},
            context=context or {},
        )

        # Store in active runs
        self._active_runs[run_id] = run_record

        # Persist to database
        if self._db_available:
            try:
                self._supabase.insert(self.config.db_table_name, run_record.to_db_dict())
            except Exception as e:
                logger.error(f"Failed to insert run record: {e}")
                self._log_to_file(run_record.to_db_dict())

        # Log to file as backup
        if self.config.enable_file_logging:
            self._log_to_file(
                {
                    "event": "run_started",
                    "run_id": run_id,
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "agent_type": agent_type,
                    "tenant_id": tenant_id,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

        # Notify callbacks
        self._notify_callbacks(
            {
                "event_type": "run_started",
                "run_id": run_id,
                "agent_id": agent_id,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        logger.info(f"Started run {run_id} for agent {agent_name}")
        return run_id

    def log_event(
        self,
        run_id: str,
        event_type: str,
        data: dict[str, Any],
    ):
        """
        Log an event during a run.

        Args:
            run_id: The run ID
            event_type: Type of event (llm_call, retry, fallback, error, etc.)
            data: Event data
        """
        event = {
            "run_id": run_id,
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Add to buffer
        self._event_buffer.append(event)

        # Update in-memory run context
        if run_id in self._active_runs:
            run = self._active_runs[run_id]
            if "events" not in run.context:
                run.context["events"] = []
            run.context["events"].append(event)

        # Notify callbacks
        self._notify_callbacks(event)

        # Log to file
        if self.config.enable_file_logging:
            self._log_to_file(event)

        logger.debug(f"Run {run_id}: {event_type}")

    def log_model_usage(
        self,
        run_id: str,
        provider: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
        success: bool = True,
        is_fallback: bool = False,
        attempt_number: int = 1,
        error: str | None = None,
    ):
        """
        Log LLM model usage for a run.

        Args:
            run_id: The run ID
            provider: LLM provider (anthropic, openai, google)
            model: Specific model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            latency_ms: Response latency in milliseconds
            success: Whether the call succeeded
            is_fallback: Whether this was a fallback attempt
            attempt_number: Which attempt this was
            error: Error message if failed
        """
        if not self.config.log_model_usage:
            return

        usage = ModelUsageRecord(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency_ms=latency_ms,
            success=success,
            is_fallback=is_fallback,
            attempt_number=attempt_number,
            error=error,
        )

        # Update run context
        if run_id in self._active_runs:
            run = self._active_runs[run_id]
            if "model_usage" not in run.context:
                run.context["model_usage"] = []
            run.context["model_usage"].append(usage.dict())

        # Log event
        self.log_event(run_id, "model_usage", usage.dict())

    def log_pii_scan(
        self,
        run_id: str,
        has_pii: bool = False,
        has_secrets: bool = False,
        pii_types: list[str] | None = None,
        secret_types: list[str] | None = None,
        content_redacted: bool = False,
    ):
        """
        Log PII scan results for a run.

        Args:
            run_id: The run ID
            has_pii: Whether PII was detected
            has_secrets: Whether secrets were detected
            pii_types: Types of PII found
            secret_types: Types of secrets found
            content_redacted: Whether content was redacted
        """
        if not self.config.log_pii_scans:
            return

        scan_result = PIIScanResult(
            has_pii=has_pii,
            has_secrets=has_secrets,
            pii_types_found=pii_types or [],
            secrets_types_found=secret_types or [],
            content_redacted=content_redacted,
        )

        # Update run context
        if run_id in self._active_runs:
            run = self._active_runs[run_id]
            run.context["pii_scan"] = scan_result.dict()

        # Log event
        self.log_event(run_id, "pii_scan", scan_result.dict())

    def complete_run(
        self,
        run_id: str,
        success: bool,
        output_data: dict[str, Any] | None = None,
        error: str | None = None,
        error_type: str | None = None,
        tokens_used: int = 0,
        model_used: str | None = None,
        fallback_used: bool = False,
        retry_count: int = 0,
    ):
        """
        Complete a run and finalize the record.

        Args:
            run_id: The run ID
            success: Whether the run succeeded
            output_data: Task output
            error: Error message if failed
            error_type: Error classification
            tokens_used: Total tokens consumed
            model_used: Primary model that produced the result
            fallback_used: Whether fallback was triggered
            retry_count: Number of retries attempted
        """
        completed_at = datetime.utcnow()

        # Update in-memory record
        if run_id in self._active_runs:
            run = self._active_runs[run_id]
            run.completed_at = completed_at
            run.status = RunStatus.COMPLETED if success else RunStatus.FAILED
            run.success = success
            run.output_data = output_data or {}
            run.error = error
            run.error_type = error_type
            run.tokens_used = tokens_used
            run.model_used = model_used
            run.fallback_used = fallback_used
            run.retry_count = retry_count
            run.updated_at = completed_at

            # Persist to database
            if self._db_available:
                try:
                    update_data = AgentRunUpdate(
                        status=run.status,
                        completed_at=completed_at,
                        success=success,
                        output_data=output_data,
                        error=error,
                        error_type=error_type,
                        tokens_used=tokens_used,
                        model_used=model_used,
                        fallback_used=fallback_used,
                        retry_count=retry_count,
                        context=run.context,
                    )
                    self._supabase.update(
                        self.config.db_table_name,
                        update_data.dict(exclude_none=True),
                        {"id": run_id},
                    )
                except Exception as e:
                    logger.error(f"Failed to update run record: {e}")
                    self._log_to_file(run.to_db_dict())

            # Log to file
            if self.config.enable_file_logging:
                self._log_to_file(
                    {
                        "event": "run_completed",
                        "run_id": run_id,
                        "success": success,
                        "tokens_used": tokens_used,
                        "model_used": model_used,
                        "fallback_used": fallback_used,
                        "error": error,
                        "timestamp": completed_at.isoformat(),
                    }
                )

            # Clean up active runs
            del self._active_runs[run_id]

        # Notify callbacks
        self._notify_callbacks(
            {
                "event_type": "run_completed",
                "run_id": run_id,
                "success": success,
                "timestamp": completed_at.isoformat(),
            }
        )

        logger.info(f"Completed run {run_id}: success={success}")

    def fail_run(
        self,
        run_id: str,
        error: str,
        error_type: str = "UnknownError",
    ):
        """
        Mark a run as failed.

        Args:
            run_id: The run ID
            error: Error message
            error_type: Error classification
        """
        self.complete_run(
            run_id=run_id,
            success=False,
            error=error,
            error_type=error_type,
        )

    def cancel_run(self, run_id: str):
        """
        Cancel an active run.

        Args:
            run_id: The run ID
        """
        if run_id in self._active_runs:
            run = self._active_runs[run_id]
            run.status = RunStatus.CANCELLED
            run.completed_at = datetime.utcnow()

            if self._db_available:
                try:
                    self._supabase.update(
                        self.config.db_table_name,
                        {"status": "cancelled", "completed_at": datetime.utcnow().isoformat()},
                        {"id": run_id},
                    )
                except Exception as e:
                    logger.error(f"Failed to cancel run: {e}")

            del self._active_runs[run_id]
            logger.info(f"Cancelled run {run_id}")

    def get_run(self, run_id: str) -> AgentRunRecord | None:
        """
        Get a run record by ID.

        Args:
            run_id: The run ID

        Returns:
            AgentRunRecord if found
        """
        # Check active runs first
        if run_id in self._active_runs:
            return self._active_runs[run_id]

        # Query database
        if self._db_available:
            try:
                row = self._supabase.select_by_id(self.config.db_table_name, run_id)
                if row:
                    return AgentRunRecord.from_db_row(row)
            except Exception as e:
                logger.error(f"Failed to get run: {e}")

        return None

    def get_runs(
        self,
        agent_id: str | None = None,
        tenant_id: str | None = None,
        status: RunStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AgentRunRecord]:
        """
        Query run records with filters.

        Args:
            agent_id: Filter by agent ID
            tenant_id: Filter by tenant ID
            status: Filter by status
            limit: Max records to return
            offset: Records to skip

        Returns:
            List of AgentRunRecord
        """
        if not self._db_available:
            return []

        try:
            filters = {}
            if agent_id:
                filters["agent_id"] = agent_id
            if tenant_id:
                filters["tenant_id"] = tenant_id
            if status:
                filters["status"] = status.value if isinstance(status, RunStatus) else status

            rows = self._supabase.select(
                self.config.db_table_name,
                filters=filters if filters else None,
                order_by="-started_at",
                limit=limit,
                offset=offset,
            )

            return [AgentRunRecord.from_db_row(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to query runs: {e}")
            return []

    def _log_to_file(self, data: dict[str, Any]):
        """Log data to file as fallback/backup"""
        if not self.config.enable_file_logging:
            return

        try:
            log_dir = Path(self.config.log_dir)
            date_str = datetime.utcnow().strftime("%Y-%m-%d")
            log_file = log_dir / f"agent_runs_{date_str}.jsonl"

            with open(log_file, "a") as f:
                f.write(json.dumps(data, default=str) + "\n")

        except Exception as e:
            logger.error(f"Failed to write to log file: {e}")

    def get_active_runs(self) -> dict[str, AgentRunRecord]:
        """Get all currently active runs"""
        return self._active_runs.copy()

    def get_stats(
        self,
        agent_id: str | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Get statistics for runs.

        Args:
            agent_id: Filter by agent ID
            tenant_id: Filter by tenant ID

        Returns:
            Statistics dictionary
        """
        if not self._db_available:
            return {"error": "Database not available"}

        try:
            # This would ideally be an RPC function for efficiency
            runs = self.get_runs(
                agent_id=agent_id,
                tenant_id=tenant_id,
                limit=1000,
            )

            if not runs:
                return {
                    "total_runs": 0,
                    "successful_runs": 0,
                    "failed_runs": 0,
                    "success_rate": 0.0,
                    "total_tokens": 0,
                    "avg_tokens_per_run": 0.0,
                    "fallback_rate": 0.0,
                }

            total = len(runs)
            successful = sum(1 for r in runs if r.success)
            failed = total - successful
            total_tokens = sum(r.tokens_used for r in runs)
            fallbacks = sum(1 for r in runs if r.fallback_used)

            return {
                "total_runs": total,
                "successful_runs": successful,
                "failed_runs": failed,
                "success_rate": successful / total if total > 0 else 0.0,
                "total_tokens": total_tokens,
                "avg_tokens_per_run": total_tokens / total if total > 0 else 0.0,
                "fallback_rate": fallbacks / total if total > 0 else 0.0,
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}


# Singleton instance for shared use
_default_logger: BrainLogger | None = None


def get_brain_logger() -> BrainLogger:
    """Get or create the default BrainLogger instance"""
    global _default_logger
    if _default_logger is None:
        _default_logger = BrainLogger()
    return _default_logger


def set_brain_logger(logger_instance: BrainLogger):
    """Set the default BrainLogger instance"""
    global _default_logger
    _default_logger = logger_instance
