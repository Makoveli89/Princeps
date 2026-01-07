"""
LLMs Module - Multi-LLM Client and Provider Abstractions

This module provides a unified interface for interacting with multiple LLM providers:
- MultiLLMClient: Unified interface for Claude, GPT, Gemini, and local models
- Provider-specific adapters with consistent API
- Automatic fallback and content-based routing logic
- Council of experts pattern for critical queries
- Load balancing across providers
- Rate limiting, batching, and error handling
"""

from framework.llms.llm_council import (
    CouncilDecision,
    CouncilMember,
    JudgeVerdict,
    LLMCouncil,
    PlanParser,
    PlanProposal,
    PlanQuality,
    VotingStrategy,
    create_council,
)
from framework.llms.multi_llm_client import (
    ContentClassifier,
    ContentType,
    CouncilResult,
    LLMConfig,
    LLMProvider,
    LLMResponse,
    LoadBalancer,
    MultiLLMClient,
    ProviderStatus,
    create_llm_client,
)

__all__ = [
    # Main client
    "MultiLLMClient",
    "create_llm_client",
    # Response and config
    "LLMResponse",
    "LLMConfig",
    # Enums
    "LLMProvider",
    "ProviderStatus",
    "ContentType",
    # Council of experts (basic)
    "CouncilResult",
    # LLM Council (advanced)
    "LLMCouncil",
    "VotingStrategy",
    "PlanQuality",
    "CouncilMember",
    "PlanProposal",
    "CouncilDecision",
    "JudgeVerdict",
    "PlanParser",
    "create_council",
    # Utilities
    "ContentClassifier",
    "LoadBalancer",
]
