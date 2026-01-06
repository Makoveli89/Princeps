"""
Agents Module - Agent Base Class and Multi-LLM Support

This module provides the foundational agent infrastructure for the Princeps system:
- BaseAgent: Abstract base class with standardized behaviors for all agent types
- Multi-LLM support: Dynamic routing to different LLM providers
- Built-in retry logic, logging, and error handling
- Brain Layer integration for persistent logging
- Tenant isolation and PII scanning
"""

from framework.agents.base_agent import (
    AgentConfig,
    AgentContext,
    AgentResponse,
    AgentTask,
    BaseAgent,
    LLMProvider,
    TaskStatus,
    create_agent,
)
from framework.agents.brain_logger import BrainLogger, LoggerConfig, get_brain_logger
from framework.agents.concept_agent import (
    ConceptAgent,
    ConceptConfig,
    ConceptExtractionModel,
    ConceptGraph,
    ConceptGraphResult,
    ConceptNode,
    ConceptRelation,
    NodeType,
    RelationType,
    create_concept_agent,
    get_concept_graph,
)
from framework.agents.entity_agent import (
    EntityConfig,
    EntityExtractionAgent,
    EntityExtractionResult,
    EntityLabel,
    ExtractedEntity,
    NERModel,
    create_entity_agent,
)
from framework.agents.executor_agent import (
    ErrorRecoveryStrategy,
    ExecutionResult,
    ExecutionStrategy,
    ExecutorAgent,
    ExecutorConfig,
    StepExecution,
    StepStatus,
    create_executor_agent,
)
from framework.agents.planner_agent import (
    ExecutionPlan,
    PlannerAgent,
    PlannerConfig,
    PlanningMode,
    PlanStep,
    PlanType,
    create_planner_agent,
)
from framework.agents.retriever_agent import (
    RerankerType,
    RetrievalFilter,
    RetrievalMode,
    RetrievalResponse,
    RetrievalResult,
    RetrievalSource,
    RetrieverAgent,
    RetrieverConfig,
    create_retriever_agent,
)
from framework.agents.schemas.agent_run import (
    AgentRunCreate,
    AgentRunRecord,
    AgentRunUpdate,
    RunStatus,
)
from framework.agents.summarizer_agent import (
    SummarizationModel,
    SummarizerAgent,
    SummarizerConfig,
    SummaryLevel,
    SummaryResult,
    create_summarizer_agent,
)
from framework.agents.topic_agent import (
    ClassificationMode,
    Topic,
    TopicAgent,
    TopicConfig,
    TopicExtractionResult,
    TopicModelType,
    create_topic_agent,
)

__all__ = [
    # Base agent classes
    "BaseAgent",
    "AgentTask",
    "AgentResponse",
    "AgentConfig",
    "AgentContext",
    "TaskStatus",
    "LLMProvider",
    "create_agent",
    # Planner agent
    "PlannerAgent",
    "PlannerConfig",
    "PlanningMode",
    "PlanType",
    "PlanStep",
    "ExecutionPlan",
    "create_planner_agent",
    # Executor agent
    "ExecutorAgent",
    "ExecutorConfig",
    "ExecutionStrategy",
    "ErrorRecoveryStrategy",
    "StepStatus",
    "StepExecution",
    "ExecutionResult",
    "create_executor_agent",
    # Retriever agent
    "RetrieverAgent",
    "RetrieverConfig",
    "RetrievalMode",
    "RetrievalSource",
    "RerankerType",
    "RetrievalFilter",
    "RetrievalResult",
    "RetrievalResponse",
    "create_retriever_agent",
    # Summarizer agent (Knowledge Distillation)
    "SummarizerAgent",
    "SummarizerConfig",
    "SummarizationModel",
    "SummaryLevel",
    "SummaryResult",
    "create_summarizer_agent",
    # Entity extraction agent (Knowledge Distillation)
    "EntityExtractionAgent",
    "EntityConfig",
    "EntityLabel",
    "NERModel",
    "ExtractedEntity",
    "EntityExtractionResult",
    "create_entity_agent",
    # Topic agent (Knowledge Distillation)
    "TopicAgent",
    "TopicConfig",
    "TopicModelType",
    "ClassificationMode",
    "Topic",
    "TopicExtractionResult",
    "create_topic_agent",
    # Concept agent (Knowledge Distillation)
    "ConceptAgent",
    "ConceptConfig",
    "ConceptExtractionModel",
    "RelationType",
    "NodeType",
    "ConceptNode",
    "ConceptRelation",
    "ConceptGraphResult",
    "ConceptGraph",
    "get_concept_graph",
    "create_concept_agent",
    # Brain logger
    "BrainLogger",
    "LoggerConfig",
    "get_brain_logger",
    # Schemas
    "AgentRunCreate",
    "AgentRunUpdate",
    "AgentRunRecord",
    "RunStatus",
]
