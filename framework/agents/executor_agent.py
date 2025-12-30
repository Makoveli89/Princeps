"""
ExecutorAgent - Tool-Using Execution Agent

This module implements the ExecutorAgent, responsible for carrying out plans
formulated by the PlannerAgent. It acts as the action agent that knows how
to execute each step by invoking tools, APIs, or system commands.

Strategic Intent:
The ExecutorAgent provides modular separation between planning and execution.
The Planner decides what needs to be done, and the Executor knows how to do it
step-by-step. This design enables:
- Easy addition of new capabilities via tool registration
- Reliable execution with error handling and recovery
- Detailed operation logging for audit and learning
- Safe execution through tool sandboxing

Integration with Brain Layer:
- Each step execution is logged as an operation
- Overall run status is tracked in agent_runs
- Artifacts and outputs can be stored for reference
- Tenant isolation is maintained throughout

Adapted from patterns in:
- base_agent.py: Core agent infrastructure
- tool_using_agent.py: Tool invocation patterns
- shell_command_agent.py: Command execution patterns
- security_scanner.py: Input validation
"""

import asyncio
import logging
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from framework.agents.base_agent import (
    BaseAgent,
    AgentConfig,
    AgentContext,
    AgentTask,
    AgentResponse,
    TaskStatus,
    LLMProvider,
)

from framework.tools.tool_registry import (
    ToolRegistry,
    ToolResult,
    ToolExecutionStatus,
    ToolSecurityLevel,
    TenantPolicy,
    get_tool_registry,
)

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """Strategy for executing plan steps"""
    SEQUENTIAL = "sequential"  # Execute steps one after another
    PARALLEL = "parallel"  # Execute independent steps in parallel
    ADAPTIVE = "adaptive"  # Dynamically choose based on dependencies


class StepStatus(Enum):
    """Status of an individual step execution"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"
    RETRYING = "retrying"


class ErrorRecoveryStrategy(Enum):
    """Strategy for handling step failures"""
    FAIL_FAST = "fail_fast"  # Stop execution on first failure
    CONTINUE = "continue"  # Continue with next steps
    RETRY = "retry"  # Retry failed step
    FALLBACK = "fallback"  # Try alternative action
    REPLAN = "replan"  # Signal PlannerAgent to revise plan


@dataclass
class StepExecution:
    """Record of a single step execution"""
    step_id: str
    step_number: int
    action: str
    description: str

    # Execution details
    tool_name: str
    parameters: Dict[str, Any]

    # Status and timing
    status: StepStatus = StepStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_ms: float = 0.0

    # Results
    output: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None

    # Retries
    attempt_count: int = 0
    max_attempts: int = 3

    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    blocks: List[str] = field(default_factory=list)

    # Context
    input_variables: Dict[str, Any] = field(default_factory=dict)
    output_variables: Dict[str, Any] = field(default_factory=dict)

    # Security
    security_warnings: List[str] = field(default_factory=list)
    was_sandboxed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_number": self.step_number,
            "action": self.action,
            "description": self.description,
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "execution_time_ms": self.execution_time_ms,
            "output": self.output if not isinstance(self.output, bytes) else "<binary>",
            "error": self.error,
            "error_type": self.error_type,
            "attempt_count": self.attempt_count,
            "depends_on": self.depends_on,
            "security_warnings": self.security_warnings,
        }


@dataclass
class ExecutionResult:
    """Result of executing a complete plan"""
    plan_id: str
    success: bool
    status: str

    # Step results
    steps_completed: int
    steps_failed: int
    steps_skipped: int
    total_steps: int

    step_executions: List[StepExecution]

    # Aggregate metrics
    total_execution_time_ms: float = 0.0
    tools_used: List[str] = field(default_factory=list)

    # Outputs
    final_output: Any = None
    output_variables: Dict[str, Any] = field(default_factory=dict)

    # Errors
    errors: List[Dict[str, Any]] = field(default_factory=list)
    requires_replan: bool = False
    replan_reason: Optional[str] = None

    # Timestamps
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "success": self.success,
            "status": self.status,
            "steps_completed": self.steps_completed,
            "steps_failed": self.steps_failed,
            "steps_skipped": self.steps_skipped,
            "total_steps": self.total_steps,
            "step_executions": [s.to_dict() for s in self.step_executions],
            "total_execution_time_ms": self.total_execution_time_ms,
            "tools_used": self.tools_used,
            "final_output": self.final_output,
            "output_variables": self.output_variables,
            "errors": self.errors,
            "requires_replan": self.requires_replan,
            "replan_reason": self.replan_reason,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class ExecutorConfig:
    """Configuration specific to the ExecutorAgent"""
    # Execution settings
    execution_strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL
    max_parallel_steps: int = 5

    # Error handling
    error_recovery: ErrorRecoveryStrategy = ErrorRecoveryStrategy.RETRY
    max_retries_per_step: int = 3
    retry_delay_seconds: float = 1.0

    # Timeouts
    step_timeout_seconds: float = 300.0
    total_timeout_seconds: float = 3600.0

    # Security
    require_tool_approval: bool = False
    sandbox_shell_commands: bool = True
    max_security_level: ToolSecurityLevel = ToolSecurityLevel.ELEVATED

    # Logging
    log_step_outputs: bool = True
    log_step_parameters: bool = True
    store_artifacts: bool = True

    # Variable handling
    enable_variable_substitution: bool = True
    variable_prefix: str = "${"
    variable_suffix: str = "}"


class ExecutorAgent(BaseAgent):
    """
    Tool-Using Execution Agent.

    The ExecutorAgent carries out plans formulated by the PlannerAgent by:
    1. Parsing structured plans into executable steps
    2. Looking up appropriate tools for each action
    3. Executing tools with proper parameters
    4. Handling errors and implementing recovery strategies
    5. Logging all operations for audit and learning

    Features:
    - Multi-tool execution with registry lookup
    - Variable passing between steps
    - Error recovery with retries and fallbacks
    - Security sandboxing for dangerous operations
    - Detailed operation logging
    - Tenant-scoped execution

    Usage:
        executor = ExecutorAgent(
            agent_name="task_executor",
            tool_registry=registry,
            executor_config=ExecutorConfig(error_recovery=ErrorRecoveryStrategy.RETRY),
        )

        # Execute a plan from PlannerAgent
        result = await executor.execute_plan(plan)

        # Or execute via task
        task = executor.create_execution_task(plan)
        response = await executor.execute_task(task)
    """

    def __init__(
        self,
        agent_name: str = "executor_agent",
        agent_type: str = "executor",
        config: Optional[AgentConfig] = None,
        executor_config: Optional[ExecutorConfig] = None,
        tool_registry: Optional[ToolRegistry] = None,
        llm_client=None,
        brain_logger=None,
        security_scanner=None,
        default_context: Optional[AgentContext] = None,
    ):
        """
        Initialize the ExecutorAgent.

        Args:
            agent_name: Unique name for this executor instance
            agent_type: Type classification (default: "executor")
            config: Base agent configuration
            executor_config: Executor-specific configuration
            tool_registry: ToolRegistry for tool lookups
            llm_client: MultiLLMClient for sub-queries
            brain_logger: BrainLogger for database logging
            security_scanner: SecurityScanner for input/output scanning
            default_context: Default context with tenant info
        """
        super().__init__(
            agent_name=agent_name,
            agent_type=agent_type,
            config=config,
            llm_client=llm_client,
            brain_logger=brain_logger,
            security_scanner=security_scanner,
            default_context=default_context,
        )

        self.executor_config = executor_config or ExecutorConfig()
        self.tool_registry = tool_registry or get_tool_registry()

        # Execution state
        self._active_executions: Dict[str, ExecutionResult] = {}
        self._execution_history: List[ExecutionResult] = []

        # Variable context for step chaining
        self._variable_context: Dict[str, Any] = {}

        # Callbacks
        self._step_callbacks: List[Callable[[StepExecution], None]] = []
        self._completion_callbacks: List[Callable[[ExecutionResult], None]] = []

        # Statistics
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_steps_executed": 0,
            "total_steps_failed": 0,
            "avg_execution_time_ms": 0.0,
            "tools_usage": {},
        }

        logger.info(f"ExecutorAgent '{agent_name}' initialized with {len(self.tool_registry._tools)} tools available")

    def _initialize_capabilities(self) -> List[str]:
        """Define executor-specific capabilities"""
        return [
            "plan-execution",
            "tool-invocation",
            "shell-commands",
            "api-requests",
            "file-operations",
            "error-recovery",
            "variable-chaining",
            "parallel-execution",
        ]

    def _get_system_prompt(self, task: AgentTask) -> str:
        """Generate system prompt for execution tasks"""
        return """You are an execution agent responsible for carrying out task plans.
Your role is to:
1. Parse the plan into executable steps
2. Identify the appropriate tool for each step
3. Execute steps in the correct order
4. Handle errors and implement recovery strategies
5. Report results and any issues

Always prioritize safety and correctness over speed.
Log any unexpected behaviors or security concerns.
"""

    def _process_response(
        self,
        raw_response: str,
        task: AgentTask,
    ) -> Dict[str, Any]:
        """Process response from execution task"""
        return {
            "text": raw_response,
            "structured_output": None,
        }

    def _fallback_handler(self, task: AgentTask, error: str) -> AgentResponse:
        """Handle cases when execution completely fails"""
        return AgentResponse(
            task_id=task.task_id,
            success=False,
            status=TaskStatus.FAILED,
            response_text=f"Execution failed: {error}",
            error=error,
            error_type="ExecutionFailure",
        )

    def add_step_callback(self, callback: Callable[[StepExecution], None]):
        """Add callback for step completion events"""
        self._step_callbacks.append(callback)

    def add_completion_callback(self, callback: Callable[[ExecutionResult], None]):
        """Add callback for execution completion events"""
        self._completion_callbacks.append(callback)

    async def execute_plan(
        self,
        plan: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Execute a plan generated by PlannerAgent.

        Args:
            plan: Execution plan dictionary with steps
            context: Additional execution context
            tenant_id: Tenant ID for isolation

        Returns:
            ExecutionResult with step-by-step results
        """
        plan_id = plan.get("plan_id", f"plan_{uuid.uuid4().hex[:12]}")
        steps = plan.get("steps", [])

        if not steps:
            return ExecutionResult(
                plan_id=plan_id,
                success=False,
                status="no_steps",
                steps_completed=0,
                steps_failed=0,
                steps_skipped=0,
                total_steps=0,
                step_executions=[],
                errors=[{"error": "No steps in plan"}],
            )

        # Initialize execution context
        execution_context = {
            "plan_id": plan_id,
            "tenant_id": tenant_id or (self.default_context.tenant_id if self.default_context else None),
            "variables": {},
            **(context or {}),
        }

        # Track execution
        run_id: Optional[str] = None
        start_time = time.time()

        # Start Brain logging
        if self._brain_logger and self.config.enable_brain_logging:
            try:
                run_id = self._brain_logger.start_run(
                    agent_id=self.agent_id,
                    agent_name=self.agent_name,
                    agent_type=self.agent_type,
                    tenant_id=execution_context["tenant_id"],
                    task_id=plan_id,
                    input_data={
                        "plan_id": plan_id,
                        "total_steps": len(steps),
                        "goal": plan.get("goal", ""),
                    },
                    context=execution_context,
                )
            except Exception as e:
                logger.warning(f"Failed to start Brain logging: {e}")

        self._log_event("plan_execution_started", {
            "plan_id": plan_id,
            "total_steps": len(steps),
            "execution_strategy": self.executor_config.execution_strategy.value,
        }, run_id)

        # Convert plan steps to step executions
        step_executions = self._prepare_step_executions(steps, plan_id)

        # Initialize result
        result = ExecutionResult(
            plan_id=plan_id,
            success=False,
            status="in_progress",
            steps_completed=0,
            steps_failed=0,
            steps_skipped=0,
            total_steps=len(step_executions),
            step_executions=step_executions,
            started_at=datetime.now(),
        )

        self._active_executions[plan_id] = result

        try:
            # Execute based on strategy
            if self.executor_config.execution_strategy == ExecutionStrategy.SEQUENTIAL:
                await self._execute_sequential(result, execution_context, run_id)
            elif self.executor_config.execution_strategy == ExecutionStrategy.PARALLEL:
                await self._execute_parallel(result, execution_context, run_id)
            else:
                await self._execute_adaptive(result, execution_context, run_id)

        except asyncio.TimeoutError:
            result.status = "timeout"
            result.errors.append({"error": "Execution timed out", "error_type": "TimeoutError"})

        except Exception as e:
            logger.error(f"Plan execution failed: {e}")
            result.status = "error"
            result.errors.append({"error": str(e), "error_type": type(e).__name__})

        finally:
            # Finalize result
            result.completed_at = datetime.now()
            result.total_execution_time_ms = (time.time() - start_time) * 1000
            result.output_variables = execution_context.get("variables", {})
            result.tools_used = list(set(
                s.tool_name for s in result.step_executions
                if s.status == StepStatus.COMPLETED
            ))

            # Determine overall success
            result.success = result.steps_failed == 0 and result.steps_completed > 0
            if result.status == "in_progress":
                result.status = "completed" if result.success else "partial_failure"

            # Get final output from last successful step
            for step in reversed(result.step_executions):
                if step.status == StepStatus.COMPLETED and step.output is not None:
                    result.final_output = step.output
                    break

            # Update statistics
            self._update_execution_stats(result)

            # Log completion
            self._log_event("plan_execution_completed", {
                "plan_id": plan_id,
                "success": result.success,
                "steps_completed": result.steps_completed,
                "steps_failed": result.steps_failed,
                "execution_time_ms": result.total_execution_time_ms,
            }, run_id)

            # Complete Brain logging
            if self._brain_logger and run_id:
                self._brain_logger.complete_run(
                    run_id=run_id,
                    success=result.success,
                    output_data={
                        "steps_completed": result.steps_completed,
                        "steps_failed": result.steps_failed,
                        "final_output_preview": str(result.final_output)[:500] if result.final_output else None,
                    },
                    tokens_used=0,
                )

            # Remove from active
            self._active_executions.pop(plan_id, None)
            self._execution_history.append(result)

            # Notify callbacks
            for callback in self._completion_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.warning(f"Completion callback failed: {e}")

        return result

    def _prepare_step_executions(
        self,
        steps: List[Dict[str, Any]],
        plan_id: str,
    ) -> List[StepExecution]:
        """Convert plan steps to step execution objects"""
        step_executions = []

        for i, step in enumerate(steps):
            # Determine tool and parameters
            tool_name, parameters = self._parse_step_action(step)

            execution = StepExecution(
                step_id=f"{plan_id}_step_{i + 1}",
                step_number=i + 1,
                action=step.get("action", tool_name),
                description=step.get("description", ""),
                tool_name=tool_name,
                parameters=parameters,
                depends_on=step.get("dependencies", []),
                max_attempts=self.executor_config.max_retries_per_step,
            )

            step_executions.append(execution)

        # Build dependency graph (blocks)
        step_map = {s.step_number: s for s in step_executions}
        for step in step_executions:
            for dep in step.depends_on:
                if dep in step_map:
                    step_map[dep].blocks.append(step.step_id)

        return step_executions

    def _parse_step_action(self, step: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Parse a step to extract tool name and parameters"""
        # Check for explicit tool specification
        if "tool" in step:
            return step["tool"], step.get("parameters", {})

        # Check for action type
        action = step.get("action", "").lower()

        # Map common actions to tools
        action_to_tool = {
            "shell": "shell",
            "command": "shell",
            "execute": "shell",
            "run": "shell",
            "api": "web_request",
            "http": "web_request",
            "request": "web_request",
            "fetch": "web_request",
            "read": "file_read",
            "read_file": "file_read",
            "write": "file_write",
            "write_file": "file_write",
            "save": "file_write",
            "wait": "wait",
            "delay": "wait",
            "sleep": "wait",
            "log": "log",
            "print": "log",
            "store": "store_variable",
            "set": "store_variable",
            "get": "get_variable",
            "transform": "json_transform",
            "json": "json_transform",
        }

        tool_name = action_to_tool.get(action, "shell")  # Default to shell

        # Build parameters
        parameters = {}

        if tool_name == "shell":
            parameters["command"] = step.get("command", step.get("parameters", {}).get("command", ""))
            if not parameters["command"]:
                # Try to construct from description
                parameters["command"] = step.get("description", "echo 'No command specified'")

        elif tool_name == "web_request":
            parameters = {
                "url": step.get("url", step.get("endpoint", "")),
                "method": step.get("method", "GET"),
                "headers": step.get("headers", {}),
                "body": step.get("body", step.get("data")),
            }

        elif tool_name in ["file_read", "file_write"]:
            parameters = {
                "path": step.get("path", step.get("file", "")),
                "content": step.get("content", ""),
            }

        elif tool_name == "wait":
            parameters["seconds"] = step.get("seconds", step.get("duration", 1))

        elif tool_name == "log":
            parameters["message"] = step.get("message", step.get("description", ""))
            parameters["level"] = step.get("level", "info")

        else:
            # Pass through all parameters
            parameters = step.get("parameters", {})

        return tool_name, parameters

    async def _execute_sequential(
        self,
        result: ExecutionResult,
        context: Dict[str, Any],
        run_id: Optional[str],
    ):
        """Execute steps sequentially"""
        for step in result.step_executions:
            # Check for timeout
            elapsed = (datetime.now() - result.started_at).total_seconds()
            if elapsed > self.executor_config.total_timeout_seconds:
                step.status = StepStatus.SKIPPED
                step.error = "Skipped due to timeout"
                result.steps_skipped += 1
                continue

            # Check dependencies
            if step.depends_on:
                deps_met = all(
                    any(
                        s.step_number == dep and s.status == StepStatus.COMPLETED
                        for s in result.step_executions
                    )
                    for dep in step.depends_on
                )
                if not deps_met:
                    step.status = StepStatus.SKIPPED
                    step.error = "Dependencies not met"
                    result.steps_skipped += 1
                    continue

            # Execute the step
            await self._execute_step(step, context, run_id)

            # Update result counts
            if step.status == StepStatus.COMPLETED:
                result.steps_completed += 1
            elif step.status == StepStatus.FAILED:
                result.steps_failed += 1
                result.errors.append({
                    "step_id": step.step_id,
                    "step_number": step.step_number,
                    "error": step.error,
                    "error_type": step.error_type,
                })

                # Check recovery strategy
                if self.executor_config.error_recovery == ErrorRecoveryStrategy.FAIL_FAST:
                    # Mark remaining steps as skipped
                    for remaining in result.step_executions:
                        if remaining.status == StepStatus.PENDING:
                            remaining.status = StepStatus.SKIPPED
                            remaining.error = "Skipped due to previous failure"
                            result.steps_skipped += 1
                    break

                elif self.executor_config.error_recovery == ErrorRecoveryStrategy.REPLAN:
                    result.requires_replan = True
                    result.replan_reason = f"Step {step.step_number} failed: {step.error}"
                    break

    async def _execute_parallel(
        self,
        result: ExecutionResult,
        context: Dict[str, Any],
        run_id: Optional[str],
    ):
        """Execute independent steps in parallel"""
        # Group steps by dependency level
        levels = self._build_dependency_levels(result.step_executions)

        for level in levels:
            # Execute steps at this level in parallel
            semaphore = asyncio.Semaphore(self.executor_config.max_parallel_steps)

            async def execute_with_semaphore(step: StepExecution):
                async with semaphore:
                    await self._execute_step(step, context, run_id)

            tasks = [execute_with_semaphore(step) for step in level]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Update counts
            for step in level:
                if step.status == StepStatus.COMPLETED:
                    result.steps_completed += 1
                elif step.status == StepStatus.FAILED:
                    result.steps_failed += 1
                    result.errors.append({
                        "step_id": step.step_id,
                        "error": step.error,
                    })

            # Check for failures
            if result.steps_failed > 0 and self.executor_config.error_recovery == ErrorRecoveryStrategy.FAIL_FAST:
                break

    async def _execute_adaptive(
        self,
        result: ExecutionResult,
        context: Dict[str, Any],
        run_id: Optional[str],
    ):
        """Execute with adaptive strategy based on dependencies"""
        # Start with sequential but parallelize when possible
        await self._execute_sequential(result, context, run_id)

    def _build_dependency_levels(
        self,
        steps: List[StepExecution],
    ) -> List[List[StepExecution]]:
        """Build levels of steps that can be executed in parallel"""
        levels = []
        remaining = list(steps)
        completed_steps = set()

        while remaining:
            # Find steps with all dependencies met
            current_level = []
            for step in remaining:
                deps_met = all(dep in completed_steps for dep in step.depends_on)
                if deps_met or not step.depends_on:
                    current_level.append(step)

            if not current_level:
                # Circular dependency or missing deps - just add all remaining
                current_level = remaining[:]

            levels.append(current_level)

            for step in current_level:
                remaining.remove(step)
                completed_steps.add(step.step_number)

        return levels

    async def _execute_step(
        self,
        step: StepExecution,
        context: Dict[str, Any],
        run_id: Optional[str],
    ):
        """Execute a single step"""
        step.status = StepStatus.IN_PROGRESS
        step.started_at = datetime.now()
        start_time = time.time()

        # Log step start
        self._log_event("step_started", {
            "step_id": step.step_id,
            "step_number": step.step_number,
            "tool": step.tool_name,
            "action": step.action,
        }, run_id)

        # Substitute variables in parameters
        if self.executor_config.enable_variable_substitution:
            step.parameters = self._substitute_variables(step.parameters, context)
            step.input_variables = {
                k: v for k, v in context.get("variables", {}).items()
                if str(v) in str(step.parameters)
            }

        # Retry loop
        while step.attempt_count < step.max_attempts:
            step.attempt_count += 1

            try:
                # Execute via tool registry
                tool_result = await self.tool_registry.execute(
                    step.tool_name,
                    step.parameters,
                    context={
                        **context,
                        "step_id": step.step_id,
                        "attempt": step.attempt_count,
                    },
                    timeout=self.executor_config.step_timeout_seconds,
                )

                # Process result
                step.output = tool_result.output
                step.security_warnings = tool_result.security_warnings
                step.was_sandboxed = tool_result.was_sandboxed

                if tool_result.status == ToolExecutionStatus.SUCCESS:
                    step.status = StepStatus.COMPLETED

                    # Store output as variable if needed
                    if step.output is not None:
                        var_name = f"step_{step.step_number}_output"
                        context.setdefault("variables", {})[var_name] = step.output

                    break

                elif tool_result.status == ToolExecutionStatus.BLOCKED:
                    step.status = StepStatus.BLOCKED
                    step.error = tool_result.error
                    step.error_type = "BlockedError"
                    break  # Don't retry blocked steps

                elif tool_result.status == ToolExecutionStatus.TIMEOUT:
                    step.error = tool_result.error
                    step.error_type = "TimeoutError"

                else:  # FAILED
                    step.error = tool_result.error
                    step.error_type = tool_result.error_type

                # Check if should retry
                if step.attempt_count < step.max_attempts:
                    if self.executor_config.error_recovery in [
                        ErrorRecoveryStrategy.RETRY,
                        ErrorRecoveryStrategy.FALLBACK
                    ]:
                        step.status = StepStatus.RETRYING
                        await asyncio.sleep(
                            self.executor_config.retry_delay_seconds * step.attempt_count
                        )
                        continue

                step.status = StepStatus.FAILED

            except Exception as e:
                step.error = str(e)
                step.error_type = type(e).__name__

                if step.attempt_count < step.max_attempts:
                    step.status = StepStatus.RETRYING
                    await asyncio.sleep(self.executor_config.retry_delay_seconds)
                    continue

                step.status = StepStatus.FAILED

        # Finalize step
        step.completed_at = datetime.now()
        step.execution_time_ms = (time.time() - start_time) * 1000

        # Log step completion
        self._log_event("step_completed", {
            "step_id": step.step_id,
            "step_number": step.step_number,
            "status": step.status.value,
            "execution_time_ms": step.execution_time_ms,
            "attempts": step.attempt_count,
            "error": step.error,
        }, run_id)

        # Notify callbacks
        for callback in self._step_callbacks:
            try:
                callback(step)
            except Exception as e:
                logger.warning(f"Step callback failed: {e}")

    def _substitute_variables(
        self,
        parameters: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Substitute variables in parameters"""
        variables = context.get("variables", {})
        prefix = self.executor_config.variable_prefix
        suffix = self.executor_config.variable_suffix

        def substitute(value: Any) -> Any:
            if isinstance(value, str):
                result = value
                for var_name, var_value in variables.items():
                    placeholder = f"{prefix}{var_name}{suffix}"
                    if placeholder in result:
                        if result == placeholder:
                            return var_value
                        result = result.replace(placeholder, str(var_value))
                return result
            elif isinstance(value, dict):
                return {k: substitute(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [substitute(v) for v in value]
            return value

        return {k: substitute(v) for k, v in parameters.items()}

    def _update_execution_stats(self, result: ExecutionResult):
        """Update execution statistics"""
        self.execution_stats["total_executions"] += 1

        if result.success:
            self.execution_stats["successful_executions"] += 1
        else:
            self.execution_stats["failed_executions"] += 1

        self.execution_stats["total_steps_executed"] += result.steps_completed
        self.execution_stats["total_steps_failed"] += result.steps_failed

        # Update average execution time
        n = self.execution_stats["total_executions"]
        old_avg = self.execution_stats["avg_execution_time_ms"]
        self.execution_stats["avg_execution_time_ms"] = (
            (old_avg * (n - 1) + result.total_execution_time_ms) / n
        )

        # Track tool usage
        for tool in result.tools_used:
            self.execution_stats["tools_usage"][tool] = (
                self.execution_stats["tools_usage"].get(tool, 0) + 1
            )

    async def execute_task(self, task: AgentTask) -> AgentResponse:
        """
        Execute a task containing a plan.

        This overrides the base execute_task to handle execution-specific tasks.
        """
        # Check if task contains a plan
        plan = task.parameters.get("plan")

        if plan:
            # Execute the plan
            result = await self.execute_plan(
                plan=plan,
                context=task.parameters.get("context", {}),
                tenant_id=task.agent_context.tenant_id if task.agent_context else None,
            )

            return AgentResponse(
                task_id=task.task_id,
                success=result.success,
                status=TaskStatus.COMPLETED if result.success else TaskStatus.FAILED,
                response_text=f"Executed {result.steps_completed}/{result.total_steps} steps",
                structured_output={
                    "execution_result": result.to_dict(),
                },
                confidence_score=result.steps_completed / max(result.total_steps, 1),
                processing_time_seconds=result.total_execution_time_ms / 1000,
            )

        # Fall back to base execution for non-plan tasks
        return await super().execute_task(task)

    def create_execution_task(
        self,
        plan: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ) -> AgentTask:
        """
        Create an execution task from a plan.

        Args:
            plan: Execution plan from PlannerAgent
            context: Additional execution context
            tenant_id: Tenant ID for isolation

        Returns:
            Configured AgentTask for execution
        """
        return self.create_task(
            prompt=f"Execute plan: {plan.get('goal', 'Unknown goal')}",
            task_type="execution",
            description=f"Execute {len(plan.get('steps', []))} steps",
            tenant_id=tenant_id,
            parameters={
                "plan": plan,
                "context": context or {},
            },
        )

    def get_available_tools(
        self,
        tenant_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get list of available tools for execution"""
        tools = self.tool_registry.list_tools(
            security_level=self.executor_config.max_security_level,
            tenant_id=tenant_id,
        )
        return [t.to_dict() for t in tools]

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return {
            **self.execution_stats,
            "active_executions": len(self._active_executions),
            "available_tools": len(self.tool_registry._tools),
        }

    def get_capabilities(self) -> Dict[str, Any]:
        """Get agent capabilities including executor-specific info"""
        base_capabilities = super().get_capabilities()
        base_capabilities.update({
            "executor_config": {
                "execution_strategy": self.executor_config.execution_strategy.value,
                "error_recovery": self.executor_config.error_recovery.value,
                "max_retries": self.executor_config.max_retries_per_step,
                "max_security_level": self.executor_config.max_security_level.value,
            },
            "execution_stats": self.get_execution_stats(),
            "available_tools": [
                t.name for t in self.tool_registry.list_tools()
            ],
        })
        return base_capabilities


# Convenience factory function
def create_executor_agent(
    tool_registry: Optional[ToolRegistry] = None,
    brain_logger=None,
    execution_strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL,
    error_recovery: ErrorRecoveryStrategy = ErrorRecoveryStrategy.RETRY,
    max_security_level: ToolSecurityLevel = ToolSecurityLevel.ELEVATED,
    tenant_id: Optional[str] = None,
) -> ExecutorAgent:
    """
    Create a configured ExecutorAgent.

    Args:
        tool_registry: ToolRegistry instance
        brain_logger: Optional BrainLogger instance
        execution_strategy: How to execute steps
        error_recovery: How to handle failures
        max_security_level: Maximum allowed tool security level
        tenant_id: Default tenant ID

    Returns:
        Configured ExecutorAgent instance
    """
    config = AgentConfig(
        enable_brain_logging=brain_logger is not None,
    )

    executor_config = ExecutorConfig(
        execution_strategy=execution_strategy,
        error_recovery=error_recovery,
        max_security_level=max_security_level,
    )

    default_context = None
    if tenant_id:
        default_context = AgentContext(tenant_id=tenant_id)

    return ExecutorAgent(
        agent_name="task_executor",
        config=config,
        executor_config=executor_config,
        tool_registry=tool_registry,
        brain_logger=brain_logger,
        default_context=default_context,
    )
