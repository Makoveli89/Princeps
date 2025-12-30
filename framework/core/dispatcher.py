"""
Dispatcher - Central Task Routing and Orchestration Engine

This module implements the Dispatcher that routes tasks to appropriate agents:
- Task type to agent workflow mapping
- Multi-step workflow execution
- Sub-task coordination and dependency management
- Idempotency checking for duplicate detection
- Comprehensive logging and tracing
- Error handling with fallback strategies

The Dispatcher is the executive coordinator that knows about all agents
and when to invoke each. It examines task types and routes them through
appropriate agent workflows.

Strategic Intent:
When a request comes in (user query, document upload, automated trigger),
the Dispatcher examines the task type to decide how to handle it. It may
break the task into sub-tasks and call multiple agents in sequence. For
example, a complex Q&A might: (1) call RetrieverAgent for context, (2)
call PlannerAgent to compose answer, (3) call SummarizerAgent for brevity.

Adapted from patterns in:
- brain/observability/run_logger.py (correlation ID propagation)
- brain/resilience/idempotency_service.py (duplicate detection)
- brain/observability/logging_config.py (OperationContext)
"""

import asyncio
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from framework.core.task import (
    Task,
    TaskType,
    TaskStatus,
    TaskResult,
    TaskContext,
    TaskPriority,
    SubTask,
    Workflow,
    WorkflowStep,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Dispatcher Configuration
# =============================================================================

@dataclass
class DispatcherConfig:
    """Configuration for the Dispatcher."""

    # Workflow settings
    max_workflow_steps: int = 20
    step_timeout_seconds: float = 60.0
    total_timeout_seconds: float = 300.0

    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    exponential_backoff: bool = True

    # Concurrency
    max_concurrent_tasks: int = 10
    max_concurrent_subtasks: int = 5

    # Caching and idempotency
    enable_result_cache: bool = True
    cache_ttl_seconds: int = 3600
    enable_idempotency: bool = True

    # Logging
    enable_logging: bool = True
    log_payloads: bool = False  # Log full payloads (may contain sensitive data)
    log_results: bool = True

    # Agent defaults
    default_agents: Dict[str, str] = field(default_factory=lambda: {
        "planner": "PlannerAgent",
        "executor": "ExecutorAgent",
        "retriever": "RetrieverAgent",
        "summarizer": "SummarizerAgent",
        "entity": "EntityExtractionAgent",
        "topic": "TopicAgent",
        "concept": "ConceptAgent",
    })


# =============================================================================
# Agent Registry
# =============================================================================

class AgentRegistry:
    """
    Registry of available agents for the Dispatcher.

    Provides lazy loading and caching of agent instances.
    """

    def __init__(self):
        self._agents: Dict[str, Any] = {}
        self._agent_classes: Dict[str, Type] = {}
        self._lock = asyncio.Lock()

    def register(self, name: str, agent_class: Type) -> None:
        """Register an agent class."""
        self._agent_classes[name] = agent_class

    async def get(self, name: str, context: Optional[TaskContext] = None) -> Any:
        """Get or create an agent instance."""
        async with self._lock:
            cache_key = f"{name}:{context.tenant_id if context else 'default'}"

            if cache_key in self._agents:
                return self._agents[cache_key]

            # Try to get agent class
            if name in self._agent_classes:
                agent_class = self._agent_classes[name]
            else:
                # Try to import dynamically
                agent_class = await self._import_agent(name)
                if agent_class:
                    self._agent_classes[name] = agent_class

            if agent_class is None:
                raise ValueError(f"Unknown agent: {name}")

            # Create instance
            agent_context = context.to_agent_context() if context else None
            agent = agent_class(context=agent_context)

            self._agents[cache_key] = agent
            return agent

    async def _import_agent(self, name: str) -> Optional[Type]:
        """Dynamically import an agent by name."""
        agent_mapping = {
            "planner": ("agents", "PlannerAgent"),
            "PlannerAgent": ("agents", "PlannerAgent"),
            "executor": ("agents", "ExecutorAgent"),
            "ExecutorAgent": ("agents", "ExecutorAgent"),
            "retriever": ("agents", "RetrieverAgent"),
            "RetrieverAgent": ("agents", "RetrieverAgent"),
            "summarizer": ("agents", "SummarizerAgent"),
            "SummarizerAgent": ("agents", "SummarizerAgent"),
            "entity": ("agents", "EntityExtractionAgent"),
            "EntityExtractionAgent": ("agents", "EntityExtractionAgent"),
            "topic": ("agents", "TopicAgent"),
            "TopicAgent": ("agents", "TopicAgent"),
            "concept": ("agents", "ConceptAgent"),
            "ConceptAgent": ("agents", "ConceptAgent"),
        }

        if name not in agent_mapping:
            return None

        module_name, class_name = agent_mapping[name]

        try:
            import importlib
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            logger.warning(f"Failed to import agent {name}: {e}")
            return None

    def list_agents(self) -> List[str]:
        """List registered agent names."""
        return list(self._agent_classes.keys())


# Global agent registry
_agent_registry = AgentRegistry()


def get_agent_registry() -> AgentRegistry:
    """Get the global agent registry."""
    return _agent_registry


# =============================================================================
# Result Cache
# =============================================================================

class ResultCache:
    """Cache for task results to enable idempotency."""

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 3600,
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[TaskResult, float]] = {}
        self._lock = asyncio.Lock()

    async def get(self, input_hash: str) -> Optional[TaskResult]:
        """Get cached result if available."""
        async with self._lock:
            if input_hash not in self._cache:
                return None

            result, timestamp = self._cache[input_hash]

            if time.time() - timestamp > self.ttl_seconds:
                del self._cache[input_hash]
                return None

            return result

    async def set(self, input_hash: str, result: TaskResult) -> None:
        """Cache a task result."""
        async with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= self.max_size:
                oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]

            self._cache[input_hash] = (result, time.time())

    async def clear(self) -> None:
        """Clear the cache."""
        async with self._lock:
            self._cache.clear()


# =============================================================================
# Workflow Definitions
# =============================================================================

# Pre-defined workflows for common task types
DEFAULT_WORKFLOWS: Dict[TaskType, Workflow] = {
    TaskType.ASK_QUESTION: Workflow(
        name="answer_question",
        task_type=TaskType.ASK_QUESTION,
        description="Answer a user question using retrieval and planning",
        steps=[
            WorkflowStep(
                name="retrieve_context",
                agent_type="retriever",
                input_mapping={"query": "question"},
                output_key="retrieved_context",
            ),
            WorkflowStep(
                name="generate_answer",
                agent_type="planner",
                input_mapping={
                    "goal": "question",
                    "context": "retrieved_context",
                },
                output_key="answer",
            ),
        ],
    ),
    TaskType.PLAN: Workflow(
        name="create_plan",
        task_type=TaskType.PLAN,
        description="Create a plan for a goal",
        steps=[
            WorkflowStep(
                name="retrieve_context",
                agent_type="retriever",
                input_mapping={"query": "goal"},
                output_key="context",
                on_error="skip",
            ),
            WorkflowStep(
                name="create_plan",
                agent_type="planner",
                input_mapping={
                    "goal": "goal",
                    "constraints": "constraints",
                    "context": "context",
                },
                output_key="plan",
            ),
        ],
    ),
    TaskType.PLAN_AND_EXECUTE: Workflow(
        name="plan_and_execute",
        task_type=TaskType.PLAN_AND_EXECUTE,
        description="Create and execute a plan",
        steps=[
            WorkflowStep(
                name="retrieve_context",
                agent_type="retriever",
                input_mapping={"query": "goal"},
                output_key="context",
                on_error="skip",
            ),
            WorkflowStep(
                name="create_plan",
                agent_type="planner",
                input_mapping={
                    "goal": "goal",
                    "constraints": "constraints",
                    "context": "context",
                },
                output_key="plan",
            ),
            WorkflowStep(
                name="execute_plan",
                agent_type="executor",
                input_mapping={"plan": "plan"},
                output_key="execution_result",
            ),
        ],
    ),
    TaskType.ANALYZE_DOCUMENT: Workflow(
        name="analyze_document",
        task_type=TaskType.ANALYZE_DOCUMENT,
        description="Full document analysis with all distillation agents",
        steps=[
            WorkflowStep(
                name="summarize",
                agent_type="summarizer",
                input_mapping={"content": "content", "source_id": "document_id"},
                output_key="summary",
            ),
            WorkflowStep(
                name="extract_entities",
                agent_type="entity",
                input_mapping={"content": "content", "source_id": "document_id"},
                output_key="entities",
            ),
            WorkflowStep(
                name="extract_topics",
                agent_type="topic",
                input_mapping={"content": "content", "source_id": "document_id"},
                output_key="topics",
            ),
            WorkflowStep(
                name="extract_concepts",
                agent_type="concept",
                input_mapping={"content": "content", "source_id": "document_id"},
                output_key="concepts",
            ),
        ],
    ),
    TaskType.DISTILL_KNOWLEDGE: Workflow(
        name="distill_knowledge",
        task_type=TaskType.DISTILL_KNOWLEDGE,
        description="Run all knowledge distillation agents",
        steps=[
            WorkflowStep(
                name="summarize",
                agent_type="summarizer",
                input_mapping={"content": "content", "source_id": "source_id"},
                output_key="summary",
                on_error="skip",
            ),
            WorkflowStep(
                name="extract_entities",
                agent_type="entity",
                input_mapping={"content": "content", "source_id": "source_id"},
                output_key="entities",
                on_error="skip",
            ),
            WorkflowStep(
                name="extract_topics",
                agent_type="topic",
                input_mapping={"content": "content", "source_id": "source_id"},
                output_key="topics",
                on_error="skip",
            ),
            WorkflowStep(
                name="extract_concepts",
                agent_type="concept",
                input_mapping={"content": "content", "source_id": "source_id"},
                output_key="concepts",
                on_error="skip",
            ),
        ],
    ),
    TaskType.SUMMARIZE: Workflow(
        name="summarize",
        task_type=TaskType.SUMMARIZE,
        description="Generate summary",
        steps=[
            WorkflowStep(
                name="summarize",
                agent_type="summarizer",
                input_mapping={"content": "content", "source_id": "source_id"},
                output_key="summary",
            ),
        ],
    ),
    TaskType.RETRIEVE: Workflow(
        name="retrieve",
        task_type=TaskType.RETRIEVE,
        description="Retrieve relevant knowledge",
        steps=[
            WorkflowStep(
                name="retrieve",
                agent_type="retriever",
                input_mapping={"query": "query", "filters": "filters", "top_k": "top_k"},
                output_key="results",
            ),
        ],
    ),
}


# =============================================================================
# Dispatcher Implementation
# =============================================================================

class Dispatcher:
    """
    Central task routing and orchestration engine.

    The Dispatcher receives tasks and routes them to appropriate agents
    through defined workflows. It handles:
    - Task type to workflow mapping
    - Multi-step workflow execution
    - Sub-task coordination
    - Result caching and idempotency
    - Error handling and fallbacks

    Usage:
        dispatcher = Dispatcher(config=config)
        result = await dispatcher.dispatch(task)
    """

    def __init__(
        self,
        config: Optional[DispatcherConfig] = None,
        agent_registry: Optional[AgentRegistry] = None,
    ):
        self.config = config or DispatcherConfig()
        self.agent_registry = agent_registry or get_agent_registry()

        # Workflow registry
        self._workflows: Dict[TaskType, Workflow] = DEFAULT_WORKFLOWS.copy()

        # Result cache
        self._result_cache = ResultCache(
            ttl_seconds=self.config.cache_ttl_seconds,
        ) if self.config.enable_result_cache else None

        # Task tracking
        self._active_tasks: Dict[str, Task] = {}
        self._task_semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)

        # Statistics
        self._stats = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "cached_results": 0,
            "total_agents_invoked": 0,
            "average_duration_ms": 0.0,
            "tasks_by_type": {},
        }

    # =========================================================================
    # Workflow Registration
    # =========================================================================

    def register_workflow(self, workflow: Workflow) -> None:
        """Register a custom workflow."""
        self._workflows[workflow.task_type] = workflow
        logger.info(f"Registered workflow: {workflow.name} for {workflow.task_type.value}")

    def get_workflow(self, task_type: TaskType) -> Optional[Workflow]:
        """Get workflow for a task type."""
        return self._workflows.get(task_type)

    def list_workflows(self) -> List[str]:
        """List registered workflow names."""
        return [w.name for w in self._workflows.values()]

    # =========================================================================
    # Core Dispatch Interface
    # =========================================================================

    async def dispatch(self, task: Task) -> TaskResult:
        """
        Dispatch a task for execution.

        This is the main entry point for task execution. It:
        1. Checks for cached results (idempotency)
        2. Gets the appropriate workflow
        3. Executes workflow steps
        4. Caches and returns result

        Args:
            task: Task to dispatch

        Returns:
            TaskResult with execution outcome
        """
        start_time = time.time()

        # Check idempotency cache
        if self.config.enable_idempotency and task.input_hash:
            cached = await self._check_cache(task.input_hash)
            if cached:
                self._stats["cached_results"] += 1
                cached.was_cached = True
                return cached

        # Acquire semaphore for concurrency control
        async with self._task_semaphore:
            try:
                # Update task status
                task.status = TaskStatus.IN_PROGRESS
                task.started_at = datetime.utcnow()
                self._active_tasks[task.id] = task

                # Log task start
                if self.config.enable_logging:
                    self._log_task_start(task)

                # Get workflow
                workflow = self._get_workflow_for_task(task)

                if workflow:
                    # Execute workflow
                    result = await self._execute_workflow(task, workflow)
                else:
                    # Handle simple/direct tasks
                    result = await self._execute_direct(task)

                # Update timing
                result.duration_ms = (time.time() - start_time) * 1000

                # Cache result
                if self.config.enable_result_cache and task.input_hash:
                    await self._result_cache.set(task.input_hash, result)

                # Update stats
                self._update_stats(task, result)

                # Log completion
                if self.config.enable_logging:
                    self._log_task_complete(task, result)

                return result

            except Exception as e:
                logger.error(f"Task dispatch failed: {e}", exc_info=True)

                result = TaskResult(
                    task_id=task.id,
                    task_type=task.type,
                    success=False,
                    status=TaskStatus.FAILED,
                    error=str(e),
                    duration_ms=(time.time() - start_time) * 1000,
                )

                self._stats["failed_tasks"] += 1

                if self.config.enable_logging:
                    self._log_task_error(task, e)

                return result

            finally:
                # Cleanup
                task.completed_at = datetime.utcnow()
                task.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
                self._active_tasks.pop(task.id, None)

    async def _check_cache(self, input_hash: str) -> Optional[TaskResult]:
        """Check for cached result."""
        if self._result_cache is None:
            return None
        return await self._result_cache.get(input_hash)

    def _get_workflow_for_task(self, task: Task) -> Optional[Workflow]:
        """Get the workflow for a task, considering overrides."""
        # Check for workflow override
        if task.workflow_override:
            for workflow in self._workflows.values():
                if workflow.name == task.workflow_override:
                    return workflow

        return self._workflows.get(task.type)

    # =========================================================================
    # Workflow Execution
    # =========================================================================

    async def _execute_workflow(
        self,
        task: Task,
        workflow: Workflow,
    ) -> TaskResult:
        """Execute a multi-step workflow."""
        # Initialize workflow context with task payload
        context = dict(task.payload)
        context["task_id"] = task.id

        agents_invoked = []
        sub_task_ids = []

        for step in workflow.steps:
            # Check step limit
            if len(agents_invoked) >= self.config.max_workflow_steps:
                logger.warning(f"Max workflow steps reached for task {task.id}")
                break

            # Check condition if present
            if step.condition and not self._evaluate_condition(step.condition, context):
                logger.debug(f"Skipping step {step.name}: condition not met")
                continue

            try:
                # Execute step
                step_result = await self._execute_step(step, context, task.context)

                # Store result
                if step.output_key:
                    context[step.output_key] = step_result

                agents_invoked.append(step.agent_type)
                sub_task_ids.append(f"{task.id}:{step.name}")

            except Exception as e:
                logger.error(f"Step {step.name} failed: {e}")

                if step.on_error == "fail":
                    raise
                elif step.on_error == "skip":
                    continue
                elif step.on_error == "retry":
                    # Retry with backoff
                    for attempt in range(step.max_retries):
                        try:
                            await asyncio.sleep(self.config.retry_delay_seconds * (2 ** attempt))
                            step_result = await self._execute_step(step, context, task.context)
                            if step.output_key:
                                context[step.output_key] = step_result
                            break
                        except Exception:
                            if attempt == step.max_retries - 1:
                                if step.fallback_agent:
                                    # Try fallback
                                    fallback_step = WorkflowStep(
                                        name=f"{step.name}_fallback",
                                        agent_type=step.fallback_agent,
                                        input_mapping=step.input_mapping,
                                        output_key=step.output_key,
                                    )
                                    step_result = await self._execute_step(
                                        fallback_step, context, task.context
                                    )
                                    if step.output_key:
                                        context[step.output_key] = step_result

        # Build result
        return TaskResult(
            task_id=task.id,
            task_type=task.type,
            success=True,
            status=TaskStatus.COMPLETED,
            output=context,
            agents_invoked=agents_invoked,
            sub_task_ids=sub_task_ids,
        )

    async def _execute_step(
        self,
        step: WorkflowStep,
        context: Dict[str, Any],
        task_context: TaskContext,
    ) -> Any:
        """Execute a single workflow step."""
        # Build input from mapping
        step_input = {}
        for agent_key, context_key in step.input_mapping.items():
            if context_key in context:
                step_input[agent_key] = context[context_key]

        # Get agent
        agent = await self.agent_registry.get(step.agent_type, task_context)

        self._stats["total_agents_invoked"] += 1

        # Execute based on agent type
        if step.agent_type in ["retriever", "RetrieverAgent"]:
            result = await agent.retrieve(
                query=step_input.get("query", ""),
                filters=step_input.get("filters"),
                top_k=step_input.get("top_k", 10),
            )
            return result.to_dict() if hasattr(result, "to_dict") else result

        elif step.agent_type in ["planner", "PlannerAgent"]:
            from agents import AgentTask

            agent_task = AgentTask(
                task_id=f"plan-{uuid.uuid4().hex[:8]}",
                input_data={
                    "goal": step_input.get("goal", ""),
                    "constraints": step_input.get("constraints", []),
                    "context": step_input.get("context", {}),
                },
            )
            response = await agent.run(agent_task)
            return response.output

        elif step.agent_type in ["executor", "ExecutorAgent"]:
            from agents import AgentTask

            agent_task = AgentTask(
                task_id=f"exec-{uuid.uuid4().hex[:8]}",
                input_data={
                    "plan": step_input.get("plan", {}),
                },
            )
            response = await agent.run(agent_task)
            return response.output

        elif step.agent_type in ["summarizer", "SummarizerAgent"]:
            result = await agent.summarize(
                content=step_input.get("content", ""),
                source_id=step_input.get("source_id"),
            )
            return result.to_dict() if hasattr(result, "to_dict") else result

        elif step.agent_type in ["entity", "EntityExtractionAgent"]:
            result = await agent.extract_entities(
                content=step_input.get("content", ""),
                source_id=step_input.get("source_id"),
            )
            return result.to_dict() if hasattr(result, "to_dict") else result

        elif step.agent_type in ["topic", "TopicAgent"]:
            result = await agent.analyze_topics(
                content=step_input.get("content", ""),
                source_id=step_input.get("source_id"),
            )
            return result.to_dict() if hasattr(result, "to_dict") else result

        elif step.agent_type in ["concept", "ConceptAgent"]:
            result = await agent.extract_concepts(
                content=step_input.get("content", ""),
                source_id=step_input.get("source_id"),
            )
            return result.to_dict() if hasattr(result, "to_dict") else result

        else:
            # Generic execution via run()
            from agents import AgentTask

            agent_task = AgentTask(
                task_id=f"step-{uuid.uuid4().hex[:8]}",
                input_data=step_input,
            )
            response = await agent.run(agent_task)
            return response.output

    async def _execute_direct(self, task: Task) -> TaskResult:
        """Execute a task directly without workflow (for simple cases)."""
        # Simple tasks that map directly to one agent
        direct_mapping = {
            TaskType.SUMMARIZE: "summarizer",
            TaskType.EXTRACT_ENTITIES: "entity",
            TaskType.EXTRACT_TOPICS: "topic",
            TaskType.EXTRACT_CONCEPTS: "concept",
            TaskType.RETRIEVE: "retriever",
            TaskType.SEMANTIC_SEARCH: "retriever",
        }

        agent_type = direct_mapping.get(task.type)

        if agent_type is None:
            return TaskResult(
                task_id=task.id,
                task_type=task.type,
                success=False,
                status=TaskStatus.FAILED,
                error=f"No handler for task type: {task.type.value}",
            )

        # Execute
        step = WorkflowStep(
            name="direct",
            agent_type=agent_type,
            input_mapping={k: k for k in task.payload.keys()},
            output_key="result",
        )

        result = await self._execute_step(step, task.payload, task.context)

        return TaskResult(
            task_id=task.id,
            task_type=task.type,
            success=True,
            status=TaskStatus.COMPLETED,
            output=result,
            agents_invoked=[agent_type],
        )

    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a condition expression against context."""
        # Simple condition evaluation (could be expanded)
        try:
            # Support basic conditions like "key exists" or "key == value"
            if "==" in condition:
                key, value = condition.split("==")
                return str(context.get(key.strip(), "")) == value.strip().strip('"\'')
            elif "!=" in condition:
                key, value = condition.split("!=")
                return str(context.get(key.strip(), "")) != value.strip().strip('"\'')
            elif condition.startswith("has "):
                key = condition[4:].strip()
                return key in context and context[key]
            else:
                return bool(context.get(condition.strip()))
        except Exception:
            return True  # Default to true on parse error

    # =========================================================================
    # Batch Dispatch
    # =========================================================================

    async def dispatch_batch(
        self,
        tasks: List[Task],
        max_concurrent: Optional[int] = None,
    ) -> List[TaskResult]:
        """
        Dispatch multiple tasks with controlled concurrency.

        Args:
            tasks: List of tasks to dispatch
            max_concurrent: Max concurrent executions (default from config)

        Returns:
            List of TaskResults in same order as input
        """
        max_concurrent = max_concurrent or self.config.max_concurrent_tasks
        semaphore = asyncio.Semaphore(max_concurrent)

        async def dispatch_one(task: Task) -> TaskResult:
            async with semaphore:
                return await self.dispatch(task)

        return await asyncio.gather(*[dispatch_one(t) for t in tasks])

    # =========================================================================
    # Logging
    # =========================================================================

    def _log_task_start(self, task: Task) -> None:
        """Log task start."""
        logger.info(
            f"Dispatching task: id={task.id}, type={task.type.value}, "
            f"correlation_id={task.context.correlation_id}"
        )

        if self.config.log_payloads:
            logger.debug(f"Task payload: {task.payload}")

    def _log_task_complete(self, task: Task, result: TaskResult) -> None:
        """Log task completion."""
        logger.info(
            f"Task completed: id={task.id}, success={result.success}, "
            f"duration={result.duration_ms:.2f}ms, agents={result.agents_invoked}"
        )

        if self.config.log_results and result.output:
            output_preview = str(result.output)[:200]
            logger.debug(f"Task output: {output_preview}...")

    def _log_task_error(self, task: Task, error: Exception) -> None:
        """Log task error."""
        logger.error(
            f"Task failed: id={task.id}, error={error}",
            exc_info=True,
        )

    # =========================================================================
    # Statistics
    # =========================================================================

    def _update_stats(self, task: Task, result: TaskResult) -> None:
        """Update internal statistics."""
        self._stats["total_tasks"] += 1

        if result.success:
            self._stats["successful_tasks"] += 1
        else:
            self._stats["failed_tasks"] += 1

        # Update by type
        type_key = task.type.value
        if type_key not in self._stats["tasks_by_type"]:
            self._stats["tasks_by_type"][type_key] = 0
        self._stats["tasks_by_type"][type_key] += 1

        # Update average duration
        n = self._stats["total_tasks"]
        old_avg = self._stats["average_duration_ms"]
        self._stats["average_duration_ms"] = old_avg + (result.duration_ms - old_avg) / n

    def get_stats(self) -> Dict[str, Any]:
        """Get dispatcher statistics."""
        return {
            **self._stats,
            "active_tasks": len(self._active_tasks),
            "registered_workflows": list(self._workflows.keys()),
        }

    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """Get list of active tasks."""
        return [t.to_dict() for t in self._active_tasks.values()]

    async def clear_cache(self) -> None:
        """Clear the result cache."""
        if self._result_cache:
            await self._result_cache.clear()


# =============================================================================
# Factory and Convenience Functions
# =============================================================================

# Global dispatcher instance
_dispatcher: Optional[Dispatcher] = None


def get_dispatcher(config: Optional[DispatcherConfig] = None) -> Dispatcher:
    """Get or create the global dispatcher."""
    global _dispatcher

    if _dispatcher is None:
        _dispatcher = Dispatcher(config)

    return _dispatcher


def set_dispatcher(dispatcher: Dispatcher) -> None:
    """Set the global dispatcher."""
    global _dispatcher
    _dispatcher = dispatcher


async def dispatch_task(task: Task) -> TaskResult:
    """Convenience function to dispatch a task using global dispatcher."""
    dispatcher = get_dispatcher()
    return await dispatcher.dispatch(task)


async def dispatch_question(
    question: str,
    tenant_id: Optional[str] = None,
    **kwargs,
) -> TaskResult:
    """Convenience function to dispatch a question task."""
    from framework.core.task import create_question_task

    task = create_question_task(question, tenant_id=tenant_id, **kwargs)
    return await dispatch_task(task)


async def dispatch_document(
    content: str,
    source: str = "user_upload",
    tenant_id: Optional[str] = None,
    analyze: bool = True,
    **kwargs,
) -> TaskResult:
    """Convenience function to dispatch a document task."""
    from framework.core.task import create_document_task

    task = create_document_task(
        content=content,
        source=source,
        tenant_id=tenant_id,
        analyze=analyze,
        **kwargs,
    )
    return await dispatch_task(task)


async def dispatch_plan(
    goal: str,
    constraints: Optional[List[str]] = None,
    tenant_id: Optional[str] = None,
    execute: bool = False,
    **kwargs,
) -> TaskResult:
    """Convenience function to dispatch a planning task."""
    from framework.core.task import create_plan_task

    task = create_plan_task(
        goal=goal,
        constraints=constraints,
        tenant_id=tenant_id,
        execute=execute,
        **kwargs,
    )
    return await dispatch_task(task)
