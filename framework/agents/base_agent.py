"""
Base Agent Class for Princeps AI System

This module defines the BaseAgent class that standardizes core behaviors for all agent types:
- Task handling with dynamic input processing
- Multi-LLM support with automatic routing and fallback
- Built-in retry logic for failed or low-confidence responses
- Comprehensive logging hooks for auditing and learning
- Error handling with graceful degradation
- Brain Layer integration for persistent logging
- Tenant isolation via tenant_id
- PII scanning before LLM calls

Strategic Intent:
The BaseAgent is the foundation for all specialized agents. It standardizes core
behaviors like initializing a task, dispatching prompts to language models, handling
responses, error retry logic, and logging results. The BaseAgent enables dynamic
multi-LLM support: routing prompts to different LLMs based on task needs or as
fallbacks. If one model fails or returns low-confidence, it automatically retries
with an alternative model, ensuring no single point of failure.

Integration with Brain Layer:
- All runs are logged to the agent_runs table via BrainLogger
- Uses AgentRunRecord Pydantic model for schema validation
- Attaches tenant_id to each run context for isolation
- Logs model usage, fallbacks, and PII scan results

Adapted from patterns in:
- anthropic_claude_agent.py, gemini_ai_agent.py, code_llama_agent.py (API integration)
- embedding_prototype.py (cascade/fallback patterns)
- Codex base_agent.py (abstract base class structure)
- retry_manager.py and error_handler.py (resilience utilities)
"""

import os
import json
import logging
import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union, TypeVar, Generic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of an agent task"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class LLMProvider(Enum):
    """Supported LLM providers"""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    META = "meta"  # Code Llama / local models
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


@dataclass
class AgentConfig:
    """Configuration for agent behavior"""

    # LLM settings
    primary_llm: LLMProvider = LLMProvider.ANTHROPIC
    fallback_llms: List[LLMProvider] = field(default_factory=lambda: [LLMProvider.OPENAI, LLMProvider.GOOGLE])

    # Retry settings (adapted from retry_manager.py patterns)
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    exponential_backoff: bool = True
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 30.0

    # Quality thresholds
    min_confidence_score: float = 0.7
    retry_on_low_confidence: bool = True

    # Timeout settings
    request_timeout_seconds: float = 60.0
    total_timeout_seconds: float = 300.0

    # Logging settings
    log_prompts: bool = True
    log_responses: bool = True
    log_to_file: bool = False
    log_file_path: Optional[str] = None

    # Brain Layer integration
    enable_brain_logging: bool = True
    log_to_database: bool = True

    # Security settings
    enable_pii_scanning: bool = True
    redact_pii_before_llm: bool = False  # If True, redact PII before sending to LLM
    block_on_secrets: bool = True  # If True, block calls containing secrets

    # Performance settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    batch_size: int = 5

    # Model preferences per provider
    model_preferences: Dict[str, str] = field(default_factory=lambda: {
        "anthropic": "claude-3-5-sonnet-20241022",
        "openai": "gpt-4-turbo-preview",
        "google": "gemini-pro",
        "meta": "codellama/CodeLlama-7b-Python-hf",
    })

    # Default generation parameters
    default_temperature: float = 0.7
    default_max_tokens: int = 4096
    default_top_p: float = 0.9


@dataclass
class AgentContext:
    """
    Context for agent execution including tenant isolation.

    This is attached to every run to ensure proper data isolation
    and traceability in multi-tenant environments.
    """
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    environment: str = "development"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "environment": self.environment,
            "metadata": self.metadata,
        }


@dataclass
class AgentTask:
    """Represents a task to be processed by an agent"""

    task_id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:12]}")
    task_type: str = "general"
    description: str = ""

    # Input data
    prompt: str = ""
    messages: List[Dict[str, str]] = field(default_factory=list)
    context: Optional[str] = None
    system_prompt: Optional[str] = None

    # Task configuration
    parameters: Dict[str, Any] = field(default_factory=dict)
    intents: set = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Routing preferences
    preferred_llm: Optional[LLMProvider] = None
    required_capabilities: List[str] = field(default_factory=list)

    # Tenant context
    agent_context: Optional[AgentContext] = None

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Status
    status: TaskStatus = TaskStatus.PENDING
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "description": self.description,
            "prompt": self.prompt,
            "messages": self.messages,
            "context": self.context,
            "system_prompt": self.system_prompt,
            "parameters": self.parameters,
            "intents": list(self.intents) if self.intents else [],
            "metadata": self.metadata,
            "preferred_llm": self.preferred_llm.value if self.preferred_llm else None,
            "required_capabilities": self.required_capabilities,
            "agent_context": self.agent_context.to_dict() if self.agent_context else None,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status.value,
            "retry_count": self.retry_count,
        }


@dataclass
class AgentResponse:
    """Standardized response from an agent"""

    task_id: str
    success: bool
    status: TaskStatus

    # Response content
    response_text: str = ""
    structured_output: Optional[Dict[str, Any]] = None

    # LLM metadata
    llm_provider: Optional[LLMProvider] = None
    model_used: str = ""

    # Quality metrics
    confidence_score: float = 0.0

    # Usage statistics
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Timing
    processing_time_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    # Error handling
    error: Optional[str] = None
    error_type: Optional[str] = None
    retry_count: int = 0
    fallback_used: bool = False

    # Audit trail
    execution_log: List[Dict[str, Any]] = field(default_factory=list)
    run_id: Optional[str] = None  # Brain Layer run ID for traceability

    # PII scan results
    pii_detected: bool = False
    pii_types: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary"""
        return {
            "task_id": self.task_id,
            "success": self.success,
            "status": self.status.value,
            "response_text": self.response_text,
            "structured_output": self.structured_output,
            "llm_provider": self.llm_provider.value if self.llm_provider else None,
            "model_used": self.model_used,
            "confidence_score": self.confidence_score,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "processing_time_seconds": self.processing_time_seconds,
            "timestamp": self.timestamp.isoformat(),
            "error": self.error,
            "error_type": self.error_type,
            "retry_count": self.retry_count,
            "fallback_used": self.fallback_used,
            "execution_log": self.execution_log,
            "run_id": self.run_id,
            "pii_detected": self.pii_detected,
            "pii_types": self.pii_types,
        }


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the Princeps system.

    Provides standardized behaviors for:
    - Task initialization and handling
    - Multi-LLM routing with fallback support
    - Retry logic for failed or low-confidence responses
    - Comprehensive logging and auditing via Brain Layer
    - Error handling with graceful degradation
    - Tenant isolation via tenant_id
    - PII scanning before external LLM calls

    Subclasses must implement:
    - _initialize_capabilities(): Define agent-specific capabilities
    - _get_system_prompt(): Return the system prompt for this agent
    - _process_response(): Process and validate LLM response
    - _fallback_handler(): Provide fallback when all LLMs fail

    Integration with Brain Layer:
    - Uses BrainLogger to record all runs to agent_runs table
    - Validates data with AgentRunRecord Pydantic model
    - Logs model usage, fallbacks, PII scans to context
    - Attaches tenant_id for multi-tenant isolation
    """

    def __init__(
        self,
        agent_name: str,
        agent_type: str,
        config: Optional[AgentConfig] = None,
        llm_client=None,  # MultiLLMClient instance
        brain_logger=None,  # BrainLogger instance
        security_scanner=None,  # SecurityScanner instance
        default_context: Optional[AgentContext] = None,
    ):
        """
        Initialize the base agent.

        Args:
            agent_name: Unique name for this agent instance
            agent_type: Type of agent (e.g., "summarization", "code_generation")
            config: Agent configuration settings
            llm_client: Multi-LLM client for API calls
            brain_logger: BrainLogger for database logging
            security_scanner: SecurityScanner for PII detection
            default_context: Default agent context with tenant info
        """
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.config = config or AgentConfig()
        self.llm_client = llm_client
        self.default_context = default_context

        # Generate unique agent ID
        self.agent_id = f"{agent_type}_{uuid.uuid4().hex[:8]}"

        # Brain Layer integration
        self._brain_logger = brain_logger
        self._init_brain_logger()

        # Security scanner for PII detection
        self._security_scanner = security_scanner
        self._init_security_scanner()

        # Agent state
        self.initialized = False
        self.capabilities: List[str] = []
        self.active_tasks: Dict[str, AgentTask] = {}
        self.task_history: List[AgentResponse] = []

        # Performance metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_retries = 0
        self.average_response_time = 0.0
        self.total_tokens_used = 0

        # Logging hooks (in addition to Brain Layer logging)
        self._log_hooks: List[Callable[[Dict[str, Any]], None]] = []

        # Response cache
        self._cache: Dict[str, AgentResponse] = {}

        # Initialize agent
        self._initialize()

        logger.info(f"Agent '{self.agent_name}' ({self.agent_type}) initialized with ID {self.agent_id}")

    def _init_brain_logger(self):
        """Initialize Brain Layer logger"""
        if not self.config.enable_brain_logging:
            return

        if self._brain_logger is None:
            try:
                from framework.agents.brain_logger import get_brain_logger
                self._brain_logger = get_brain_logger()
                logger.info("Brain logger initialized")
            except Exception as e:
                logger.warning(f"Could not initialize Brain logger: {e}")
                self._brain_logger = None

    def _init_security_scanner(self):
        """Initialize security scanner for PII detection"""
        if not self.config.enable_pii_scanning:
            return

        if self._security_scanner is None:
            try:
                from brain.security.security_scanner import SecurityScanner
                self._security_scanner = SecurityScanner()
                logger.info("Security scanner initialized")
            except Exception as e:
                logger.warning(f"Could not initialize security scanner: {e}")
                self._security_scanner = None

    def _initialize(self):
        """Initialize the agent with capabilities and dependencies"""
        try:
            self.capabilities = self._initialize_capabilities()
            self.initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize agent '{self.agent_name}': {e}")
            self.initialized = False

    @abstractmethod
    def _initialize_capabilities(self) -> List[str]:
        """
        Initialize and return agent-specific capabilities.

        Returns:
            List of capability strings this agent supports
        """
        pass

    @abstractmethod
    def _get_system_prompt(self, task: AgentTask) -> str:
        """
        Generate the system prompt for this agent.

        Args:
            task: The task being processed

        Returns:
            System prompt string
        """
        pass

    @abstractmethod
    def _process_response(
        self,
        raw_response: str,
        task: AgentTask
    ) -> Dict[str, Any]:
        """
        Process and validate the raw LLM response.

        Args:
            raw_response: Raw text response from LLM
            task: The original task

        Returns:
            Processed response dictionary with 'text' and optional 'structured_output'
        """
        pass

    @abstractmethod
    def _fallback_handler(self, task: AgentTask, error: str) -> AgentResponse:
        """
        Handle cases when all LLM calls fail.

        Args:
            task: The failed task
            error: Error description

        Returns:
            Fallback AgentResponse
        """
        pass

    def set_context(self, context: AgentContext):
        """
        Set the default context for this agent.

        Args:
            context: AgentContext with tenant_id and other metadata
        """
        self.default_context = context
        logger.info(f"Agent context set: tenant_id={context.tenant_id}")

    def add_log_hook(self, hook: Callable[[Dict[str, Any]], None]):
        """
        Add a logging hook for auditing and learning.

        Args:
            hook: Callable that receives log entries
        """
        self._log_hooks.append(hook)

    def _log_event(self, event_type: str, data: Dict[str, Any], run_id: Optional[str] = None):
        """
        Log an event and trigger hooks.

        Args:
            event_type: Type of event (e.g., "task_started", "llm_call", "error")
            data: Event data
            run_id: Optional Brain Layer run ID
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "agent_id": self.agent_id,
            "event_type": event_type,
            "data": data,
        }

        # Log to standard logger
        if event_type == "error":
            logger.error(f"[{self.agent_name}] {event_type}: {data}")
        else:
            logger.info(f"[{self.agent_name}] {event_type}: {data.get('message', '')}")

        # Log to Brain Layer
        if self._brain_logger and run_id:
            try:
                self._brain_logger.log_event(run_id, event_type, data)
            except Exception as e:
                logger.warning(f"Failed to log to Brain: {e}")

        # Trigger hooks
        for hook in self._log_hooks:
            try:
                hook(log_entry)
            except Exception as e:
                logger.warning(f"Log hook failed: {e}")

        # Write to file if enabled
        if self.config.log_to_file and self.config.log_file_path:
            try:
                with open(self.config.log_file_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
            except Exception as e:
                logger.warning(f"Failed to write log to file: {e}")

    def _scan_for_pii(self, content: str) -> Dict[str, Any]:
        """
        Scan content for PII and secrets.

        Args:
            content: Text content to scan

        Returns:
            Dictionary with scan results
        """
        if not self._security_scanner or not self.config.enable_pii_scanning:
            return {"scanned": False}

        try:
            result = self._security_scanner.scan(content)
            return {
                "scanned": True,
                "has_pii": result.has_pii,
                "has_secrets": result.has_secrets,
                "pii_types": result.pii_found,
                "secret_types": result.secrets_found,
            }
        except Exception as e:
            logger.warning(f"PII scan failed: {e}")
            return {"scanned": False, "error": str(e)}

    def _redact_content(self, content: str) -> str:
        """
        Redact PII and secrets from content.

        Args:
            content: Text content to redact

        Returns:
            Redacted content
        """
        if not self._security_scanner:
            return content

        try:
            return self._security_scanner.redact(content)
        except Exception as e:
            logger.warning(f"Redaction failed: {e}")
            return content

    def _generate_cache_key(self, task: AgentTask) -> str:
        """Generate a cache key for a task"""
        key_data = f"{task.prompt}:{task.system_prompt}:{task.task_type}"
        return f"cache_{hash(key_data) % 10**10}"

    def _check_cache(self, task: AgentTask) -> Optional[AgentResponse]:
        """Check if a cached response exists for this task"""
        if not self.config.enable_caching:
            return None

        cache_key = self._generate_cache_key(task)
        cached = self._cache.get(cache_key)

        if cached:
            cache_age = (datetime.now() - cached.timestamp).total_seconds()
            if cache_age < self.config.cache_ttl_seconds:
                self._log_event("cache_hit", {"task_id": task.task_id})
                return cached
            else:
                # Cache expired
                del self._cache[cache_key]

        return None

    def _cache_response(self, task: AgentTask, response: AgentResponse):
        """Cache a response for future use"""
        if self.config.enable_caching and response.success:
            cache_key = self._generate_cache_key(task)
            self._cache[cache_key] = response

    async def execute_task(self, task: AgentTask) -> AgentResponse:
        """
        Execute a task with full retry and fallback support.

        Args:
            task: The task to execute

        Returns:
            AgentResponse with results or error information
        """
        start_time = time.time()
        execution_log: List[Dict[str, Any]] = []
        run_id: Optional[str] = None

        # Merge default context with task context
        if task.agent_context is None and self.default_context:
            task.agent_context = self.default_context

        # Get tenant_id for logging
        tenant_id = task.agent_context.tenant_id if task.agent_context else None

        # Check initialization
        if not self.initialized:
            return AgentResponse(
                task_id=task.task_id,
                success=False,
                status=TaskStatus.FAILED,
                error="Agent not properly initialized",
                error_type="InitializationError",
            )

        # Check cache
        cached_response = self._check_cache(task)
        if cached_response:
            cached_response.task_id = task.task_id
            return cached_response

        # Start Brain Layer logging
        if self._brain_logger and self.config.enable_brain_logging:
            try:
                run_id = self._brain_logger.start_run(
                    agent_id=self.agent_id,
                    agent_name=self.agent_name,
                    agent_type=self.agent_type,
                    tenant_id=tenant_id,
                    task_id=task.task_id,
                    input_data={
                        "prompt": task.prompt[:500] if self.config.log_prompts else "[REDACTED]",
                        "task_type": task.task_type,
                        "description": task.description,
                        "parameters": task.parameters,
                    },
                    context=task.agent_context.to_dict() if task.agent_context else {},
                )
            except Exception as e:
                logger.warning(f"Failed to start Brain logging: {e}")

        # PII Scanning
        pii_scan_result = {"scanned": False}
        content_to_send = task.prompt

        if self.config.enable_pii_scanning and task.prompt:
            pii_scan_result = self._scan_for_pii(task.prompt)

            if pii_scan_result.get("scanned"):
                # Log PII scan to Brain
                if self._brain_logger and run_id:
                    self._brain_logger.log_pii_scan(
                        run_id=run_id,
                        has_pii=pii_scan_result.get("has_pii", False),
                        has_secrets=pii_scan_result.get("has_secrets", False),
                        pii_types=pii_scan_result.get("pii_types", []),
                        secret_types=pii_scan_result.get("secret_types", []),
                        content_redacted=self.config.redact_pii_before_llm,
                    )

                # Block on secrets if configured
                if pii_scan_result.get("has_secrets") and self.config.block_on_secrets:
                    self._log_event("security_block", {
                        "task_id": task.task_id,
                        "reason": "Secrets detected in content",
                        "secret_types": pii_scan_result.get("secret_types", []),
                    }, run_id)

                    if self._brain_logger and run_id:
                        self._brain_logger.fail_run(
                            run_id=run_id,
                            error="Blocked: secrets detected in content",
                            error_type="SecurityError",
                        )

                    return AgentResponse(
                        task_id=task.task_id,
                        success=False,
                        status=TaskStatus.FAILED,
                        error="Blocked: secrets detected in content",
                        error_type="SecurityError",
                        run_id=run_id,
                    )

                # Redact if configured
                if self.config.redact_pii_before_llm and pii_scan_result.get("has_pii"):
                    content_to_send = self._redact_content(task.prompt)

        # Update task status
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        self.active_tasks[task.task_id] = task

        self._log_event("task_started", {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "tenant_id": tenant_id,
            "message": f"Starting task: {task.description[:100]}..."
        }, run_id)

        # Determine LLM order (primary + fallbacks)
        llm_order = self._determine_llm_order(task)

        last_error = None
        response = None
        fallback_used = False

        for llm_idx, llm_provider in enumerate(llm_order):
            if llm_idx > 0:
                fallback_used = True
                self._log_event("fallback_attempt", {
                    "task_id": task.task_id,
                    "llm_provider": llm_provider.value,
                    "attempt": llm_idx + 1,
                }, run_id)

            # Attempt with retries
            for attempt in range(self.config.max_retries):
                try:
                    task.retry_count = attempt

                    if attempt > 0:
                        task.status = TaskStatus.RETRYING
                        self.total_retries += 1

                        # Calculate delay with exponential backoff
                        delay = self.config.retry_delay_seconds
                        if self.config.exponential_backoff:
                            delay *= (self.config.backoff_multiplier ** attempt)
                            delay = min(delay, self.config.max_backoff_seconds)

                        self._log_event("retry_attempt", {
                            "task_id": task.task_id,
                            "attempt": attempt + 1,
                            "delay_seconds": delay,
                        }, run_id)

                        await asyncio.sleep(delay)

                    # Create modified task with potentially redacted content
                    call_task = task
                    if content_to_send != task.prompt:
                        call_task = AgentTask(**{**task.to_dict(), "prompt": content_to_send})
                        call_task.agent_context = task.agent_context

                    # Make LLM call
                    call_start = time.time()
                    raw_response, usage = await self._call_llm(call_task, llm_provider)
                    call_latency = (time.time() - call_start) * 1000

                    # Log model usage to Brain
                    if self._brain_logger and run_id:
                        self._brain_logger.log_model_usage(
                            run_id=run_id,
                            provider=llm_provider.value,
                            model=self.config.model_preferences.get(llm_provider.value, ""),
                            input_tokens=usage.get("input_tokens", 0),
                            output_tokens=usage.get("output_tokens", 0),
                            latency_ms=call_latency,
                            success=True,
                            is_fallback=fallback_used,
                            attempt_number=attempt + 1,
                        )

                    execution_log.append({
                        "llm_provider": llm_provider.value,
                        "attempt": attempt + 1,
                        "success": True,
                        "latency_ms": call_latency,
                        "timestamp": datetime.now().isoformat(),
                    })

                    # Process response
                    processed = self._process_response(raw_response, task)

                    # Calculate confidence
                    confidence = self._calculate_confidence(processed, task)

                    # Check if confidence meets threshold
                    if (confidence < self.config.min_confidence_score and
                        self.config.retry_on_low_confidence and
                        attempt < self.config.max_retries - 1):

                        self._log_event("low_confidence_retry", {
                            "task_id": task.task_id,
                            "confidence": confidence,
                            "threshold": self.config.min_confidence_score,
                        }, run_id)
                        continue

                    # Success!
                    processing_time = time.time() - start_time

                    response = AgentResponse(
                        task_id=task.task_id,
                        success=True,
                        status=TaskStatus.COMPLETED,
                        response_text=processed.get("text", raw_response),
                        structured_output=processed.get("structured_output"),
                        llm_provider=llm_provider,
                        model_used=self.config.model_preferences.get(llm_provider.value, ""),
                        confidence_score=confidence,
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                        total_tokens=usage.get("total_tokens", 0),
                        processing_time_seconds=processing_time,
                        retry_count=attempt,
                        fallback_used=fallback_used,
                        execution_log=execution_log,
                        run_id=run_id,
                        pii_detected=pii_scan_result.get("has_pii", False),
                        pii_types=pii_scan_result.get("pii_types", []),
                    )

                    # Update metrics
                    self._update_metrics(response)

                    # Cache response
                    self._cache_response(task, response)

                    # Update task status
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now()

                    self._log_event("task_completed", {
                        "task_id": task.task_id,
                        "success": True,
                        "confidence": confidence,
                        "processing_time": processing_time,
                        "message": "Task completed successfully",
                    }, run_id)

                    # Complete Brain logging
                    if self._brain_logger and run_id:
                        self._brain_logger.complete_run(
                            run_id=run_id,
                            success=True,
                            output_data={
                                "response_preview": response.response_text[:500] if self.config.log_responses else "[REDACTED]",
                                "confidence_score": confidence,
                            },
                            tokens_used=response.total_tokens,
                            model_used=response.model_used,
                            fallback_used=fallback_used,
                            retry_count=attempt,
                        )

                    # Cleanup
                    del self.active_tasks[task.task_id]
                    self.task_history.append(response)

                    return response

                except Exception as e:
                    last_error = str(e)
                    execution_log.append({
                        "llm_provider": llm_provider.value,
                        "attempt": attempt + 1,
                        "success": False,
                        "error": last_error,
                        "timestamp": datetime.now().isoformat(),
                    })

                    # Log failed model usage
                    if self._brain_logger and run_id:
                        self._brain_logger.log_model_usage(
                            run_id=run_id,
                            provider=llm_provider.value,
                            model=self.config.model_preferences.get(llm_provider.value, ""),
                            success=False,
                            is_fallback=fallback_used,
                            attempt_number=attempt + 1,
                            error=last_error,
                        )

                    self._log_event("llm_call_failed", {
                        "task_id": task.task_id,
                        "llm_provider": llm_provider.value,
                        "attempt": attempt + 1,
                        "error": last_error,
                    }, run_id)

        # All attempts failed - use fallback handler
        self._log_event("all_attempts_failed", {
            "task_id": task.task_id,
            "last_error": last_error,
        }, run_id)

        response = self._fallback_handler(task, last_error or "All LLM attempts failed")
        response.execution_log = execution_log
        response.fallback_used = True
        response.processing_time_seconds = time.time() - start_time
        response.run_id = run_id

        # Update task status
        task.status = TaskStatus.FAILED
        task.completed_at = datetime.now()

        # Complete Brain logging with failure
        if self._brain_logger and run_id:
            self._brain_logger.complete_run(
                run_id=run_id,
                success=False,
                error=last_error or "All LLM attempts failed",
                error_type="AllProvidersFailedError",
                fallback_used=True,
            )

        # Cleanup
        if task.task_id in self.active_tasks:
            del self.active_tasks[task.task_id]
        self.task_history.append(response)

        # Update metrics
        self._update_metrics(response)

        return response

    def execute_task_sync(self, task: AgentTask) -> AgentResponse:
        """
        Synchronous wrapper for execute_task.

        Args:
            task: The task to execute

        Returns:
            AgentResponse with results
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.execute_task(task))

    async def _call_llm(
        self,
        task: AgentTask,
        llm_provider: LLMProvider
    ) -> tuple[str, Dict[str, int]]:
        """
        Make an LLM API call.

        Args:
            task: The task containing prompt/messages
            llm_provider: Which LLM to use

        Returns:
            Tuple of (response_text, usage_dict)
        """
        if self.llm_client is None:
            raise RuntimeError("No LLM client configured. Please provide a MultiLLMClient instance.")

        # Get system prompt
        system_prompt = task.system_prompt or self._get_system_prompt(task)

        # Get model and parameters
        model = self.config.model_preferences.get(llm_provider.value)
        params = {
            "temperature": task.parameters.get("temperature", self.config.default_temperature),
            "max_tokens": task.parameters.get("max_tokens", self.config.default_max_tokens),
            "top_p": task.parameters.get("top_p", self.config.default_top_p),
        }

        # Log if enabled
        if self.config.log_prompts:
            self._log_event("llm_request", {
                "task_id": task.task_id,
                "llm_provider": llm_provider.value,
                "model": model,
                "prompt_preview": task.prompt[:200] + "..." if len(task.prompt) > 200 else task.prompt,
            })

        # Make the call through the multi-LLM client
        response = await self.llm_client.generate(
            prompt=task.prompt,
            messages=task.messages,
            system_prompt=system_prompt,
            provider=llm_provider,
            model=model,
            **params
        )

        # Log response if enabled
        if self.config.log_responses:
            self._log_event("llm_response", {
                "task_id": task.task_id,
                "llm_provider": llm_provider.value,
                "response_preview": response["text"][:200] + "..." if len(response["text"]) > 200 else response["text"],
            })

        return response["text"], response.get("usage", {})

    def _determine_llm_order(self, task: AgentTask) -> List[LLMProvider]:
        """
        Determine the order of LLMs to try based on task preferences and config.

        Args:
            task: The task being processed

        Returns:
            Ordered list of LLM providers to try
        """
        order = []

        # Task-specific preference takes priority
        if task.preferred_llm:
            order.append(task.preferred_llm)

        # Then primary LLM
        if self.config.primary_llm not in order:
            order.append(self.config.primary_llm)

        # Then fallbacks
        for fallback in self.config.fallback_llms:
            if fallback not in order:
                order.append(fallback)

        return order

    def _calculate_confidence(
        self,
        processed_response: Dict[str, Any],
        task: AgentTask
    ) -> float:
        """
        Calculate confidence score for a response.

        Override this in subclasses for domain-specific confidence calculation.

        Args:
            processed_response: The processed response
            task: The original task

        Returns:
            Confidence score between 0.0 and 1.0
        """
        text = processed_response.get("text", "")

        if not text or not text.strip():
            return 0.0

        score = 0.5  # Base score

        # Length-based heuristic (not too short)
        if len(text.split()) > 10:
            score += 0.2

        # Check for error indicators
        error_indicators = ["i cannot", "i'm sorry", "i don't", "error", "failed"]
        if any(indicator in text.lower() for indicator in error_indicators):
            score -= 0.2

        # Check for structure (if expecting structured output)
        if processed_response.get("structured_output"):
            score += 0.2

        return max(0.0, min(1.0, score))

    def _update_metrics(self, response: AgentResponse):
        """Update agent performance metrics"""
        self.total_requests += 1

        if response.success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        self.total_tokens_used += response.total_tokens

        # Update average response time
        n = self.total_requests
        self.average_response_time = (
            (self.average_response_time * (n - 1) + response.processing_time_seconds) / n
        )

    def create_task(
        self,
        prompt: str,
        task_type: str = "general",
        description: str = "",
        tenant_id: Optional[str] = None,
        **kwargs
    ) -> AgentTask:
        """
        Create a new task for this agent.

        Args:
            prompt: The main prompt/query
            task_type: Type of task
            description: Human-readable description
            tenant_id: Tenant ID for isolation (overrides default context)
            **kwargs: Additional task parameters

        Returns:
            Configured AgentTask
        """
        # Create context with tenant_id
        context = kwargs.get("agent_context")
        if context is None and (tenant_id or self.default_context):
            context = AgentContext(
                tenant_id=tenant_id or (self.default_context.tenant_id if self.default_context else None),
                user_id=self.default_context.user_id if self.default_context else None,
                session_id=self.default_context.session_id if self.default_context else None,
            )

        return AgentTask(
            task_type=task_type,
            description=description or prompt[:100],
            prompt=prompt,
            messages=kwargs.get("messages", []),
            context=kwargs.get("context"),
            system_prompt=kwargs.get("system_prompt"),
            parameters=kwargs.get("parameters", {}),
            intents=kwargs.get("intents", set()),
            metadata=kwargs.get("metadata", {}),
            preferred_llm=kwargs.get("preferred_llm"),
            required_capabilities=kwargs.get("required_capabilities", []),
            agent_context=context,
        )

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get agent capabilities and status.

        Returns:
            Dictionary of agent capabilities and metadata
        """
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "initialized": self.initialized,
            "capabilities": self.capabilities,
            "config": {
                "primary_llm": self.config.primary_llm.value,
                "fallback_llms": [llm.value for llm in self.config.fallback_llms],
                "max_retries": self.config.max_retries,
                "min_confidence_score": self.config.min_confidence_score,
                "pii_scanning_enabled": self.config.enable_pii_scanning,
                "brain_logging_enabled": self.config.enable_brain_logging,
            },
            "metrics": {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": (
                    self.successful_requests / max(1, self.total_requests)
                ),
                "total_retries": self.total_retries,
                "average_response_time": self.average_response_time,
                "total_tokens_used": self.total_tokens_used,
            },
            "active_tasks": len(self.active_tasks),
            "cache_size": len(self._cache),
            "default_tenant_id": self.default_context.tenant_id if self.default_context else None,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.get_capabilities()["metrics"]

    def clear_cache(self):
        """Clear the response cache"""
        self._cache.clear()
        self._log_event("cache_cleared", {"message": "Response cache cleared"})

    def reset_metrics(self):
        """Reset performance metrics"""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_retries = 0
        self.average_response_time = 0.0
        self.total_tokens_used = 0
        self._log_event("metrics_reset", {"message": "Performance metrics reset"})


# Convenience function for creating agents
def create_agent(
    agent_class: type,
    agent_name: str,
    agent_type: str,
    config: Optional[AgentConfig] = None,
    llm_client=None,
    brain_logger=None,
    security_scanner=None,
    tenant_id: Optional[str] = None,
) -> BaseAgent:
    """
    Factory function for creating agent instances.

    Args:
        agent_class: The agent class to instantiate
        agent_name: Name for the agent
        agent_type: Type of agent
        config: Optional configuration
        llm_client: Optional LLM client
        brain_logger: Optional Brain logger
        security_scanner: Optional security scanner
        tenant_id: Optional tenant ID for default context

    Returns:
        Configured agent instance
    """
    default_context = None
    if tenant_id:
        default_context = AgentContext(tenant_id=tenant_id)

    return agent_class(
        agent_name=agent_name,
        agent_type=agent_type,
        config=config,
        llm_client=llm_client,
        brain_logger=brain_logger,
        security_scanner=security_scanner,
        default_context=default_context,
    )
