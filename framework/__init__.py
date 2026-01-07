"""
Princeps Multi-Agent Framework
==============================

A sophisticated multi-agent orchestration system with:
- Specialized agents (planner, executor, retriever, summarizer, etc.)
- Multi-LLM support with automatic fallback
- Tool registry and execution
- Task dispatching and workflow management
- Evaluation and A/B testing

Usage:
    from framework.agents import BaseAgent, PlannerAgent
    from framework.llms import MultiLLMClient
    from framework.core import Dispatcher, Task
"""

from framework.agents import BaseAgent
from framework.core import Dispatcher, Task, TaskType
from framework.llms import MultiLLMClient

__all__ = [
    "BaseAgent",
    "Task",
    "TaskType",
    "Dispatcher",
    "MultiLLMClient",
]
