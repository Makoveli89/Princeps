"""
Core Module - Task Routing and Orchestration

This module provides the central orchestration infrastructure for Princeps:
- Task: Unified task representation with type, payload, and context
- Dispatcher: Central task routing engine with workflow execution
- Workflow: Multi-step workflow definitions

Usage:
    from core import (
        Task,
        TaskType,
        TaskResult,
        Dispatcher,
        dispatch_task,
        dispatch_question,
        create_task,
    )

    # Create and dispatch a task
    task = create_task(
        task_type=TaskType.ASK_QUESTION,
        payload={"question": "How do I implement auth?"},
        tenant_id="tenant_123",
    )
    result = await dispatch_task(task)

    # Or use convenience functions
    result = await dispatch_question("How do I implement auth?")
"""

from framework.core.task import (
    # Enums
    TaskType,
    TaskPriority,
    TaskStatus,
    # Data classes
    Task,
    TaskContext,
    TaskResult,
    SubTask,
    Workflow,
    WorkflowStep,
    # Factory functions
    create_task,
    create_question_task,
    create_document_task,
    create_plan_task,
)

from framework.core.dispatcher import (
    # Main class
    Dispatcher,
    DispatcherConfig,
    # Registry
    AgentRegistry,
    get_agent_registry,
    # Cache
    ResultCache,
    # Global access
    get_dispatcher,
    set_dispatcher,
    # Convenience functions
    dispatch_task,
    dispatch_question,
    dispatch_document,
    dispatch_plan,
    # Workflow definitions
    DEFAULT_WORKFLOWS,
)

__all__ = [
    # Task types and status
    "TaskType",
    "TaskPriority",
    "TaskStatus",
    # Task classes
    "Task",
    "TaskContext",
    "TaskResult",
    "SubTask",
    "Workflow",
    "WorkflowStep",
    # Task factory functions
    "create_task",
    "create_question_task",
    "create_document_task",
    "create_plan_task",
    # Dispatcher
    "Dispatcher",
    "DispatcherConfig",
    # Registry
    "AgentRegistry",
    "get_agent_registry",
    # Cache
    "ResultCache",
    # Global dispatcher
    "get_dispatcher",
    "set_dispatcher",
    # Dispatch functions
    "dispatch_task",
    "dispatch_question",
    "dispatch_document",
    "dispatch_plan",
    # Workflows
    "DEFAULT_WORKFLOWS",
]
