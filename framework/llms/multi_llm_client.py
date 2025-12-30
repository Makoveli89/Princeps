"""
Multi-LLM Client for Princeps AI System

This module implements an abstraction layer for multiple LLM APIs, providing:
- Uniform interface (generate() method) for all providers
- Automatic routing to the correct LLM client based on content type
- Fallback sequences when primary models fail
- Council-of-experts pattern for critical queries
- Batching support for high-throughput scenarios
- Rate limiting and quota management
- Health monitoring and provider status tracking
- Load balancing across providers

Strategic Intent:
The MultiLLMClient abstracts over multiple LLM APIs to provide unified access.
It handles automatic fallback when providers fail, content-based routing
(e.g., code queries to code-specialized models), and implements the
council-of-experts pattern where multiple models can be queried and their
answers compared for critical decisions.

Adapted from patterns in:
- anthropic_claude_agent.py: Claude API integration
- gemini_ai_agent.py: Google Gemini API integration
- code_llama_agent.py: Local model inference
- embedding_prototype.py: Cascade/fallback retrieval patterns
"""

import os
import json
import logging
import asyncio
import time
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import threading
from collections import deque
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers"""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    META = "meta"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class ProviderStatus(Enum):
    """Status of an LLM provider"""
    AVAILABLE = "available"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    UNAVAILABLE = "unavailable"
    INITIALIZING = "initializing"


class ContentType(Enum):
    """Content type for routing decisions"""
    CODE = "code"
    GENERAL = "general"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    MATH = "math"
    TRANSLATION = "translation"


@dataclass
class LLMConfig:
    """Configuration for LLM client"""

    # Provider-specific API keys (can also use environment variables)
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None

    # Default models per provider
    default_models: Dict[str, str] = field(default_factory=lambda: {
        "anthropic": "claude-3-5-sonnet-20241022",
        "openai": "gpt-4-turbo-preview",
        "google": "gemini-pro",
        "meta": "codellama/CodeLlama-7b-Python-hf",
        "local": "default",
    })

    # Content-based model preferences (route specific content types to specialized models)
    content_routing: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "code": {
            "anthropic": "claude-3-5-sonnet-20241022",
            "openai": "gpt-4-turbo-preview",
            "meta": "codellama/CodeLlama-34b-Python-hf",
        },
        "creative": {
            "anthropic": "claude-3-5-sonnet-20241022",
            "openai": "gpt-4-turbo-preview",
        },
        "analysis": {
            "anthropic": "claude-3-5-sonnet-20241022",
            "openai": "gpt-4-turbo-preview",
        },
        "math": {
            "openai": "gpt-4-turbo-preview",
            "anthropic": "claude-3-5-sonnet-20241022",
        },
    })

    # Rate limiting
    requests_per_minute: Dict[str, int] = field(default_factory=lambda: {
        "anthropic": 50,
        "openai": 60,
        "google": 60,
        "meta": 1000,  # Local model, no API limit
        "local": 1000,
    })

    # Timeout settings
    request_timeout: float = 60.0
    connection_timeout: float = 10.0

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0

    # Fallback order (default, can be overridden per content type)
    fallback_order: List[str] = field(default_factory=lambda: [
        "anthropic", "openai", "google", "local"
    ])

    # Content-specific fallback orders
    content_fallback_order: Dict[str, List[str]] = field(default_factory=lambda: {
        "code": ["anthropic", "openai", "meta", "google"],
        "creative": ["anthropic", "openai", "google"],
        "analysis": ["anthropic", "openai", "google"],
        "math": ["openai", "anthropic", "google"],
    })

    # Batching
    enable_batching: bool = True
    batch_size: int = 5
    batch_delay_ms: int = 100

    # Health check interval
    health_check_interval_seconds: int = 60

    # Council of experts settings
    enable_council: bool = True
    council_min_providers: int = 2
    council_agreement_threshold: float = 0.7  # Similarity threshold for agreement

    # Load balancing
    enable_load_balancing: bool = True
    load_balance_strategy: str = "round_robin"  # round_robin, least_loaded, weighted


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider"""

    success: bool
    text: str
    provider: LLMProvider
    model: str

    # Usage statistics
    usage: Dict[str, int] = field(default_factory=dict)

    # Metadata
    finish_reason: Optional[str] = None
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    # Error info
    error: Optional[str] = None
    error_type: Optional[str] = None

    # Raw response for debugging
    raw_response: Optional[Dict[str, Any]] = None

    # Council metadata (if used)
    is_council_response: bool = False
    council_votes: Optional[Dict[str, str]] = None
    council_agreement: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "text": self.text,
            "provider": self.provider.value,
            "model": self.model,
            "usage": self.usage,
            "finish_reason": self.finish_reason,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
            "error": self.error,
            "error_type": self.error_type,
            "is_council_response": self.is_council_response,
            "council_votes": self.council_votes,
            "council_agreement": self.council_agreement,
        }


@dataclass
class CouncilResult:
    """Result from council of experts query"""
    responses: Dict[str, LLMResponse]
    consensus_text: str
    agreement_score: float
    majority_provider: str
    all_agree: bool


class RateLimiter:
    """Token bucket rate limiter for API calls"""

    def __init__(self, requests_per_minute: int):
        self.rpm = requests_per_minute
        self.tokens = requests_per_minute
        self.last_refill = time.time()
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        """Try to acquire a token. Returns True if successful."""
        with self._lock:
            self._refill()
            if self.tokens > 0:
                self.tokens -= 1
                return True
            return False

    def _refill(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * (self.rpm / 60.0)
        self.tokens = min(self.rpm, self.tokens + tokens_to_add)
        self.last_refill = now

    async def wait_for_token(self, timeout: float = 60.0) -> bool:
        """Wait until a token is available"""
        start = time.time()
        while time.time() - start < timeout:
            if self.acquire():
                return True
            await asyncio.sleep(0.1)
        return False

    def get_available_tokens(self) -> float:
        """Get current available tokens"""
        with self._lock:
            self._refill()
            return self.tokens


class LoadBalancer:
    """Load balancer for distributing requests across providers"""

    def __init__(self, providers: List[str], strategy: str = "round_robin"):
        self.providers = providers
        self.strategy = strategy
        self._current_index = 0
        self._lock = threading.Lock()
        self._request_counts: Dict[str, int] = {p: 0 for p in providers}
        self._weights: Dict[str, float] = {p: 1.0 for p in providers}

    def get_next_provider(self, available_providers: List[str]) -> Optional[str]:
        """Get next provider based on load balancing strategy"""
        if not available_providers:
            return None

        # Filter to only available providers
        valid = [p for p in available_providers if p in self.providers]
        if not valid:
            return available_providers[0] if available_providers else None

        with self._lock:
            if self.strategy == "round_robin":
                # Simple round robin
                provider = valid[self._current_index % len(valid)]
                self._current_index += 1
                return provider

            elif self.strategy == "least_loaded":
                # Return provider with fewest requests
                counts = {p: self._request_counts.get(p, 0) for p in valid}
                return min(counts, key=counts.get)

            elif self.strategy == "weighted":
                # Weighted selection based on configured weights
                import random
                weights = [self._weights.get(p, 1.0) for p in valid]
                total = sum(weights)
                if total == 0:
                    return valid[0]
                r = random.uniform(0, total)
                cumulative = 0
                for i, p in enumerate(valid):
                    cumulative += weights[i]
                    if r <= cumulative:
                        return p
                return valid[-1]

            else:
                return valid[0]

    def record_request(self, provider: str):
        """Record that a request was made to a provider"""
        with self._lock:
            self._request_counts[provider] = self._request_counts.get(provider, 0) + 1

    def set_weight(self, provider: str, weight: float):
        """Set weight for a provider (used in weighted strategy)"""
        with self._lock:
            self._weights[provider] = weight


class ContentClassifier:
    """Classifies content type for routing decisions"""

    # Patterns for detecting content types
    CODE_PATTERNS = [
        r'\bdef\s+\w+\s*\(',  # Python function
        r'\bfunction\s+\w+\s*\(',  # JavaScript function
        r'\bclass\s+\w+',  # Class definition
        r'```\w*\n',  # Code blocks
        r'\bimport\s+\w+',  # Import statements
        r'\bfrom\s+\w+\s+import',  # Python imports
        r'[{}\[\]];',  # Code syntax
        r'#include\s*<',  # C/C++ includes
        r'public\s+class',  # Java class
    ]

    MATH_PATTERNS = [
        r'\d+\s*[\+\-\*\/\^]\s*\d+',  # Math operations
        r'\bsolve\b.*equation',
        r'\bcalculate\b',
        r'\bderivative\b',
        r'\bintegral\b',
        r'\bprobability\b',
        r'\bstatistics\b',
    ]

    CREATIVE_PATTERNS = [
        r'\bwrite\s+(a\s+)?(story|poem|song|essay)',
        r'\bcreative\b',
        r'\bimagine\b',
        r'\bfiction\b',
    ]

    ANALYSIS_PATTERNS = [
        r'\banalyze\b',
        r'\bcompare\b.*\band\b',
        r'\bevaluate\b',
        r'\bexplain\b.*\bdifference',
        r'\bsummarize\b',
    ]

    @classmethod
    def classify(cls, text: str) -> ContentType:
        """Classify the content type of the given text"""
        text_lower = text.lower()

        # Check for code patterns first (highest priority for specialized routing)
        for pattern in cls.CODE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return ContentType.CODE

        # Check for math
        for pattern in cls.MATH_PATTERNS:
            if re.search(pattern, text_lower):
                return ContentType.MATH

        # Check for creative
        for pattern in cls.CREATIVE_PATTERNS:
            if re.search(pattern, text_lower):
                return ContentType.CREATIVE

        # Check for analysis
        for pattern in cls.ANALYSIS_PATTERNS:
            if re.search(pattern, text_lower):
                return ContentType.ANALYSIS

        return ContentType.GENERAL


class BaseLLMAdapter(ABC):
    """Base class for LLM provider adapters"""

    def __init__(self, config: LLMConfig, provider: LLMProvider):
        self.config = config
        self.provider = provider
        self.status = ProviderStatus.INITIALIZING
        self.last_error: Optional[str] = None
        self.last_error_time: Optional[datetime] = None
        self.rate_limiter = RateLimiter(
            config.requests_per_minute.get(provider.value, 60)
        )
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self._active_requests = 0

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from this LLM provider"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available"""
        pass

    def get_default_model(self) -> str:
        """Get the default model for this provider"""
        return self.config.default_models.get(self.provider.value, "default")

    def get_load(self) -> float:
        """Get current load (0-1) based on rate limiter and active requests"""
        available_tokens = self.rate_limiter.get_available_tokens()
        max_tokens = self.config.requests_per_minute.get(self.provider.value, 60)
        token_load = 1.0 - (available_tokens / max_tokens) if max_tokens > 0 else 0
        return token_load


class AnthropicAdapter(BaseLLMAdapter):
    """Adapter for Anthropic Claude API"""

    def __init__(self, config: LLMConfig):
        super().__init__(config, LLMProvider.ANTHROPIC)
        self.api_key = config.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = None

        if self.api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                self.status = ProviderStatus.AVAILABLE
                logger.info("Anthropic adapter initialized successfully")
            except ImportError:
                logger.warning("anthropic package not installed")
                self.status = ProviderStatus.UNAVAILABLE
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
                self.status = ProviderStatus.ERROR
                self.last_error = str(e)
        else:
            logger.warning("ANTHROPIC_API_KEY not found")
            self.status = ProviderStatus.UNAVAILABLE

    def is_available(self) -> bool:
        return self.client is not None and self.status == ProviderStatus.AVAILABLE

    async def generate(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        start_time = time.time()
        self.total_requests += 1
        self._active_requests += 1

        try:
            if not self.is_available():
                self.failed_requests += 1
                return LLMResponse(
                    success=False,
                    text="",
                    provider=self.provider,
                    model=model or self.get_default_model(),
                    error="Anthropic client not available",
                    error_type="UnavailableError",
                )

            # Rate limiting
            if not await self.rate_limiter.wait_for_token(timeout=30.0):
                self.failed_requests += 1
                self.status = ProviderStatus.RATE_LIMITED
                return LLMResponse(
                    success=False,
                    text="",
                    provider=self.provider,
                    model=model or self.get_default_model(),
                    error="Rate limit exceeded",
                    error_type="RateLimitError",
                )

            model = model or self.get_default_model()

            # Build messages
            if messages:
                api_messages = messages
            else:
                api_messages = [{"role": "user", "content": prompt}]

            # Build request parameters
            request_params = {
                "model": model,
                "max_tokens": kwargs.get("max_tokens", 4096),
                "messages": api_messages,
            }

            if system_prompt:
                request_params["system"] = system_prompt

            if "temperature" in kwargs:
                request_params["temperature"] = kwargs["temperature"]

            if "top_p" in kwargs:
                request_params["top_p"] = kwargs["top_p"]

            # Make API call
            response = self.client.messages.create(**request_params)

            latency_ms = (time.time() - start_time) * 1000

            self.successful_requests += 1
            self.status = ProviderStatus.AVAILABLE

            return LLMResponse(
                success=True,
                text=response.content[0].text,
                provider=self.provider,
                model=model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                },
                finish_reason=response.stop_reason,
                latency_ms=latency_ms,
                raw_response={"id": response.id, "model": response.model},
            )

        except Exception as e:
            self.failed_requests += 1
            self.last_error = str(e)
            self.last_error_time = datetime.now()

            # Check for rate limit errors
            if "rate" in str(e).lower() or "429" in str(e):
                self.status = ProviderStatus.RATE_LIMITED
                error_type = "RateLimitError"
            else:
                self.status = ProviderStatus.ERROR
                error_type = "APIError"

            return LLMResponse(
                success=False,
                text="",
                provider=self.provider,
                model=model or self.get_default_model(),
                error=str(e),
                error_type=error_type,
                latency_ms=(time.time() - start_time) * 1000,
            )
        finally:
            self._active_requests -= 1


class OpenAIAdapter(BaseLLMAdapter):
    """Adapter for OpenAI GPT API"""

    def __init__(self, config: LLMConfig):
        super().__init__(config, LLMProvider.OPENAI)
        self.api_key = config.openai_api_key or os.getenv("OPENAI_API_KEY")
        self.client = None

        if self.api_key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
                self.status = ProviderStatus.AVAILABLE
                logger.info("OpenAI adapter initialized successfully")
            except ImportError:
                logger.warning("openai package not installed")
                self.status = ProviderStatus.UNAVAILABLE
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.status = ProviderStatus.ERROR
                self.last_error = str(e)
        else:
            logger.warning("OPENAI_API_KEY not found")
            self.status = ProviderStatus.UNAVAILABLE

    def is_available(self) -> bool:
        return self.client is not None and self.status == ProviderStatus.AVAILABLE

    async def generate(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        start_time = time.time()
        self.total_requests += 1
        self._active_requests += 1

        try:
            if not self.is_available():
                self.failed_requests += 1
                return LLMResponse(
                    success=False,
                    text="",
                    provider=self.provider,
                    model=model or self.get_default_model(),
                    error="OpenAI client not available",
                    error_type="UnavailableError",
                )

            # Rate limiting
            if not await self.rate_limiter.wait_for_token(timeout=30.0):
                self.failed_requests += 1
                self.status = ProviderStatus.RATE_LIMITED
                return LLMResponse(
                    success=False,
                    text="",
                    provider=self.provider,
                    model=model or self.get_default_model(),
                    error="Rate limit exceeded",
                    error_type="RateLimitError",
                )

            model = model or self.get_default_model()

            # Build messages
            api_messages = []
            if system_prompt:
                api_messages.append({"role": "system", "content": system_prompt})

            if messages:
                api_messages.extend(messages)
            else:
                api_messages.append({"role": "user", "content": prompt})

            # Build request parameters
            request_params = {
                "model": model,
                "messages": api_messages,
                "max_tokens": kwargs.get("max_tokens", 4096),
            }

            if "temperature" in kwargs:
                request_params["temperature"] = kwargs["temperature"]

            if "top_p" in kwargs:
                request_params["top_p"] = kwargs["top_p"]

            # Make API call
            response = self.client.chat.completions.create(**request_params)

            latency_ms = (time.time() - start_time) * 1000

            self.successful_requests += 1
            self.status = ProviderStatus.AVAILABLE

            return LLMResponse(
                success=True,
                text=response.choices[0].message.content,
                provider=self.provider,
                model=model,
                usage={
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                finish_reason=response.choices[0].finish_reason,
                latency_ms=latency_ms,
                raw_response={"id": response.id, "model": response.model},
            )

        except Exception as e:
            self.failed_requests += 1
            self.last_error = str(e)
            self.last_error_time = datetime.now()

            if "rate" in str(e).lower() or "429" in str(e):
                self.status = ProviderStatus.RATE_LIMITED
                error_type = "RateLimitError"
            else:
                self.status = ProviderStatus.ERROR
                error_type = "APIError"

            return LLMResponse(
                success=False,
                text="",
                provider=self.provider,
                model=model or self.get_default_model(),
                error=str(e),
                error_type=error_type,
                latency_ms=(time.time() - start_time) * 1000,
            )
        finally:
            self._active_requests -= 1


class GoogleAdapter(BaseLLMAdapter):
    """Adapter for Google Gemini API"""

    def __init__(self, config: LLMConfig):
        super().__init__(config, LLMProvider.GOOGLE)
        self.api_key = config.google_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.initialized = False

        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.genai = genai
                self.initialized = True
                self.status = ProviderStatus.AVAILABLE
                logger.info("Google Gemini adapter initialized successfully")
            except ImportError:
                logger.warning("google-generativeai package not installed")
                self.status = ProviderStatus.UNAVAILABLE
            except Exception as e:
                logger.error(f"Failed to initialize Google Gemini: {e}")
                self.status = ProviderStatus.ERROR
                self.last_error = str(e)
        else:
            logger.warning("GEMINI_API_KEY/GOOGLE_API_KEY not found")
            self.status = ProviderStatus.UNAVAILABLE

    def is_available(self) -> bool:
        return self.initialized and self.status == ProviderStatus.AVAILABLE

    async def generate(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        start_time = time.time()
        self.total_requests += 1
        self._active_requests += 1

        try:
            if not self.is_available():
                self.failed_requests += 1
                return LLMResponse(
                    success=False,
                    text="",
                    provider=self.provider,
                    model=model or self.get_default_model(),
                    error="Google Gemini not available",
                    error_type="UnavailableError",
                )

            # Rate limiting
            if not await self.rate_limiter.wait_for_token(timeout=30.0):
                self.failed_requests += 1
                self.status = ProviderStatus.RATE_LIMITED
                return LLMResponse(
                    success=False,
                    text="",
                    provider=self.provider,
                    model=model or self.get_default_model(),
                    error="Rate limit exceeded",
                    error_type="RateLimitError",
                )

            model_name = model or self.get_default_model()

            # Initialize model
            gemini_model = self.genai.GenerativeModel(model_name)

            # Build content
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt

            if messages:
                # Convert messages to Gemini format
                for msg in messages:
                    if msg["role"] == "user":
                        full_prompt += f"\nUser: {msg['content']}"
                    else:
                        full_prompt += f"\nAssistant: {msg['content']}"

            # Configure generation
            generation_config = self.genai.types.GenerationConfig(
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
                max_output_tokens=kwargs.get("max_tokens", 4096),
            )

            # Generate response
            response = gemini_model.generate_content(
                full_prompt,
                generation_config=generation_config,
            )

            latency_ms = (time.time() - start_time) * 1000

            self.successful_requests += 1
            self.status = ProviderStatus.AVAILABLE

            # Extract usage if available
            usage = {}
            if hasattr(response, "usage_metadata"):
                usage = {
                    "input_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
                    "output_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
                    "total_tokens": getattr(response.usage_metadata, "total_token_count", 0),
                }

            return LLMResponse(
                success=True,
                text=response.text,
                provider=self.provider,
                model=model_name,
                usage=usage,
                latency_ms=latency_ms,
            )

        except Exception as e:
            self.failed_requests += 1
            self.last_error = str(e)
            self.last_error_time = datetime.now()

            if "rate" in str(e).lower() or "429" in str(e):
                self.status = ProviderStatus.RATE_LIMITED
                error_type = "RateLimitError"
            else:
                self.status = ProviderStatus.ERROR
                error_type = "APIError"

            return LLMResponse(
                success=False,
                text="",
                provider=self.provider,
                model=model or self.get_default_model(),
                error=str(e),
                error_type=error_type,
                latency_ms=(time.time() - start_time) * 1000,
            )
        finally:
            self._active_requests -= 1


class LocalAdapter(BaseLLMAdapter):
    """
    Adapter for local models (Code Llama, HuggingFace models, etc.)

    This is a mock implementation that can be extended for actual local inference.
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config, LLMProvider.LOCAL)
        self.status = ProviderStatus.AVAILABLE
        logger.info("Local model adapter initialized (mock mode)")

    def is_available(self) -> bool:
        return self.status == ProviderStatus.AVAILABLE

    async def generate(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        start_time = time.time()
        self.total_requests += 1
        self._active_requests += 1

        try:
            model_name = model or self.get_default_model()

            # Mock response for local model
            mock_response = f"""[Local Model Response]
Model: {model_name}
Prompt received: {prompt[:100]}...

This is a mock response from the local adapter.
In production, this would use a local transformer model or Code Llama.

To enable actual local inference:
1. Install transformers and torch
2. Extend LocalAdapter.generate() with model loading/inference logic
3. Consider using quantization for memory efficiency
"""

            await asyncio.sleep(0.1)  # Simulate processing time

            latency_ms = (time.time() - start_time) * 1000
            self.successful_requests += 1

            return LLMResponse(
                success=True,
                text=mock_response,
                provider=self.provider,
                model=model_name,
                usage={
                    "input_tokens": len(prompt.split()),
                    "output_tokens": len(mock_response.split()),
                    "total_tokens": len(prompt.split()) + len(mock_response.split()),
                },
                latency_ms=latency_ms,
            )

        except Exception as e:
            self.failed_requests += 1
            return LLMResponse(
                success=False,
                text="",
                provider=self.provider,
                model=model or self.get_default_model(),
                error=str(e),
                error_type="LocalError",
                latency_ms=(time.time() - start_time) * 1000,
            )
        finally:
            self._active_requests -= 1


class MultiLLMClient:
    """
    Unified client for multiple LLM providers.

    Provides:
    - Single generate() interface for all providers
    - Automatic fallback when primary provider fails
    - Content-based routing to specialized models
    - Council of experts for critical queries
    - Rate limiting and quota management
    - Health monitoring and status tracking
    - Load balancing for high-throughput scenarios
    - Batching for parallel requests
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize the multi-LLM client.

        Args:
            config: Configuration for the client
        """
        self.config = config or LLMConfig()
        self.adapters: Dict[LLMProvider, BaseLLMAdapter] = {}

        # Initialize adapters
        self._initialize_adapters()

        # Load balancer
        available = self.get_available_providers()
        self.load_balancer = LoadBalancer(
            available,
            strategy=self.config.load_balance_strategy
        )

        # Content classifier
        self.content_classifier = ContentClassifier()

        # Request queue for batching
        self._request_queue: deque = deque()
        self._batch_lock = asyncio.Lock()

        # Health check state
        self._last_health_check = datetime.now()
        self._health_check_task = None

        logger.info("MultiLLMClient initialized")

    def _initialize_adapters(self):
        """Initialize all LLM adapters"""
        self.adapters[LLMProvider.ANTHROPIC] = AnthropicAdapter(self.config)
        self.adapters[LLMProvider.OPENAI] = OpenAIAdapter(self.config)
        self.adapters[LLMProvider.GOOGLE] = GoogleAdapter(self.config)
        self.adapters[LLMProvider.LOCAL] = LocalAdapter(self.config)

        available = [p.value for p, a in self.adapters.items() if a.is_available()]
        logger.info(f"Available LLM providers: {available}")

    def _get_model_for_content(
        self,
        content_type: ContentType,
        provider: LLMProvider
    ) -> Optional[str]:
        """Get the appropriate model for content type and provider"""
        content_key = content_type.value
        if content_key in self.config.content_routing:
            routing = self.config.content_routing[content_key]
            if provider.value in routing:
                return routing[provider.value]
        return self.config.default_models.get(provider.value)

    async def generate(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
        enable_fallback: bool = True,
        enable_content_routing: bool = True,
        use_council: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response using the specified or default LLM provider.

        Args:
            prompt: The main prompt text
            messages: Optional conversation messages
            system_prompt: Optional system prompt
            provider: Preferred LLM provider (will use fallback if unavailable)
            model: Specific model to use
            enable_fallback: Whether to try fallback providers on failure
            enable_content_routing: Whether to route based on content type
            use_council: Whether to use council of experts pattern
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Dictionary with 'text' and 'usage' keys (plus error info if failed)
        """
        # Use council of experts if requested
        if use_council and self.config.enable_council:
            return await self._generate_with_council(
                prompt=prompt,
                messages=messages,
                system_prompt=system_prompt,
                **kwargs
            )

        # Classify content for routing
        content_type = ContentType.GENERAL
        if enable_content_routing:
            content_type = self.content_classifier.classify(prompt)
            logger.debug(f"Content classified as: {content_type.value}")

        # Determine provider order
        providers = self._get_provider_order(provider, content_type)

        last_error = None

        for llm_provider in providers:
            adapter = self.adapters.get(llm_provider)

            if not adapter or not adapter.is_available():
                continue

            # Get content-appropriate model if not specified
            effective_model = model
            if not effective_model and enable_content_routing:
                effective_model = self._get_model_for_content(content_type, llm_provider)

            # Record request for load balancing
            if self.config.enable_load_balancing:
                self.load_balancer.record_request(llm_provider.value)

            response = await adapter.generate(
                prompt=prompt,
                messages=messages,
                system_prompt=system_prompt,
                model=effective_model,
                **kwargs
            )

            if response.success:
                return {
                    "text": response.text,
                    "usage": response.usage,
                    "provider": response.provider.value,
                    "model": response.model,
                    "latency_ms": response.latency_ms,
                    "content_type": content_type.value,
                }

            last_error = response.error

            if not enable_fallback:
                break

            logger.warning(f"Provider {llm_provider.value} failed: {response.error}")

        # All providers failed
        return {
            "text": "",
            "usage": {},
            "error": last_error or "All LLM providers failed",
            "success": False,
        }

    async def _generate_with_council(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate using council of experts pattern.

        Queries multiple LLMs and returns consensus or best response.

        Args:
            prompt: The prompt to send to all providers
            messages: Optional messages
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Returns:
            Response dictionary with council metadata
        """
        available_providers = [
            p for p, a in self.adapters.items()
            if a.is_available()
        ]

        if len(available_providers) < self.config.council_min_providers:
            logger.warning(
                f"Not enough providers for council ({len(available_providers)} < "
                f"{self.config.council_min_providers}). Using single provider."
            )
            return await self.generate(
                prompt=prompt,
                messages=messages,
                system_prompt=system_prompt,
                use_council=False,
                **kwargs
            )

        # Query all available providers in parallel
        tasks = []
        for provider in available_providers:
            adapter = self.adapters[provider]
            task = adapter.generate(
                prompt=prompt,
                messages=messages,
                system_prompt=system_prompt,
                **kwargs
            )
            tasks.append((provider, task))

        # Gather results
        responses: Dict[str, LLMResponse] = {}
        for provider, task in tasks:
            try:
                response = await task
                if response.success:
                    responses[provider.value] = response
            except Exception as e:
                logger.warning(f"Council member {provider.value} failed: {e}")

        if not responses:
            return {
                "text": "",
                "usage": {},
                "error": "All council members failed",
                "success": False,
            }

        # Analyze responses for consensus
        council_result = self._analyze_council_responses(responses)

        # Build response
        primary_response = responses.get(council_result.majority_provider)
        if not primary_response:
            primary_response = list(responses.values())[0]

        return {
            "text": council_result.consensus_text,
            "usage": primary_response.usage,
            "provider": council_result.majority_provider,
            "model": primary_response.model,
            "latency_ms": primary_response.latency_ms,
            "is_council_response": True,
            "council_votes": {p: r.text[:100] for p, r in responses.items()},
            "council_agreement": council_result.agreement_score,
            "all_agree": council_result.all_agree,
        }

    def _analyze_council_responses(
        self,
        responses: Dict[str, LLMResponse]
    ) -> CouncilResult:
        """
        Analyze council responses to find consensus.

        Uses simple text similarity to determine agreement.
        More sophisticated implementations could use semantic similarity.

        Args:
            responses: Dictionary of provider -> response

        Returns:
            CouncilResult with consensus information
        """
        texts = {p: r.text for p, r in responses.items()}
        providers = list(texts.keys())

        if len(providers) == 1:
            return CouncilResult(
                responses=responses,
                consensus_text=list(texts.values())[0],
                agreement_score=1.0,
                majority_provider=providers[0],
                all_agree=True,
            )

        # Calculate pairwise similarity (simple jaccard on words)
        def similarity(text1: str, text2: str) -> float:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            return intersection / union if union > 0 else 0.0

        # Calculate average similarity
        similarities = []
        for i, p1 in enumerate(providers):
            for p2 in providers[i+1:]:
                sim = similarity(texts[p1], texts[p2])
                similarities.append(sim)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

        # Find response with highest average similarity to others (most "central")
        best_provider = providers[0]
        best_avg_sim = 0.0

        for p1 in providers:
            sims = [similarity(texts[p1], texts[p2]) for p2 in providers if p2 != p1]
            avg = sum(sims) / len(sims) if sims else 0.0
            if avg > best_avg_sim:
                best_avg_sim = avg
                best_provider = p1

        return CouncilResult(
            responses=responses,
            consensus_text=texts[best_provider],
            agreement_score=avg_similarity,
            majority_provider=best_provider,
            all_agree=avg_similarity >= self.config.council_agreement_threshold,
        )

    def generate_sync(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for generate().
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.generate(prompt, messages, system_prompt, provider, model, **kwargs)
        )

    async def generate_batch(
        self,
        requests: List[Dict[str, Any]],
        provider: Optional[LLMProvider] = None,
        concurrency: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for multiple requests in parallel.

        Args:
            requests: List of request dictionaries with 'prompt' and optional params
            provider: LLM provider to use
            concurrency: Maximum concurrent requests

        Returns:
            List of response dictionaries
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_generate(request: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                return await self.generate(
                    prompt=request.get("prompt", ""),
                    messages=request.get("messages"),
                    system_prompt=request.get("system_prompt"),
                    provider=provider,
                    model=request.get("model"),
                    **request.get("params", {})
                )

        tasks = [bounded_generate(req) for req in requests]
        return await asyncio.gather(*tasks)

    def _get_provider_order(
        self,
        preferred: Optional[LLMProvider] = None,
        content_type: ContentType = ContentType.GENERAL
    ) -> List[LLMProvider]:
        """
        Get the order of providers to try based on content type and preferences.

        Args:
            preferred: Preferred provider to try first
            content_type: Type of content being processed

        Returns:
            Ordered list of providers
        """
        order = []

        # Add preferred provider first
        if preferred and preferred in self.adapters:
            order.append(preferred)

        # Get content-specific fallback order if available
        content_key = content_type.value
        if content_key in self.config.content_fallback_order:
            fallback_names = self.config.content_fallback_order[content_key]
        else:
            fallback_names = self.config.fallback_order

        # Apply load balancing if enabled
        if self.config.enable_load_balancing:
            available = [
                p for p in fallback_names
                if LLMProvider(p) in self.adapters and
                self.adapters[LLMProvider(p)].is_available()
            ]
            next_provider = self.load_balancer.get_next_provider(available)
            if next_provider and LLMProvider(next_provider) not in order:
                order.append(LLMProvider(next_provider))

        # Add fallback order
        for provider_name in fallback_names:
            try:
                provider = LLMProvider(provider_name)
                if provider not in order and provider in self.adapters:
                    order.append(provider)
            except ValueError:
                continue

        return order

    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all providers.

        Returns:
            Dictionary of provider status information
        """
        status = {}
        for provider, adapter in self.adapters.items():
            status[provider.value] = {
                "status": adapter.status.value,
                "is_available": adapter.is_available(),
                "total_requests": adapter.total_requests,
                "successful_requests": adapter.successful_requests,
                "failed_requests": adapter.failed_requests,
                "success_rate": (
                    adapter.successful_requests / max(1, adapter.total_requests)
                ),
                "current_load": adapter.get_load(),
                "last_error": adapter.last_error,
                "last_error_time": (
                    adapter.last_error_time.isoformat() if adapter.last_error_time else None
                ),
            }
        return status

    def get_available_providers(self) -> List[str]:
        """Get list of available provider names"""
        return [p.value for p, a in self.adapters.items() if a.is_available()]

    def is_provider_available(self, provider: LLMProvider) -> bool:
        """Check if a specific provider is available"""
        adapter = self.adapters.get(provider)
        return adapter.is_available() if adapter else False

    async def health_check(self) -> Dict[str, bool]:
        """
        Perform health check on all providers.

        Returns:
            Dictionary of provider health status
        """
        health = {}

        for provider, adapter in self.adapters.items():
            try:
                if adapter.is_available():
                    # Simple test prompt
                    response = await adapter.generate(
                        prompt="Hello",
                        max_tokens=10,
                    )
                    health[provider.value] = response.success
                else:
                    health[provider.value] = False
            except Exception:
                health[provider.value] = False

        self._last_health_check = datetime.now()
        return health


# Convenience function for quick access
def create_llm_client(
    anthropic_key: Optional[str] = None,
    openai_key: Optional[str] = None,
    google_key: Optional[str] = None,
    enable_council: bool = True,
    enable_load_balancing: bool = True,
    **kwargs
) -> MultiLLMClient:
    """
    Create a configured MultiLLMClient.

    Args:
        anthropic_key: Anthropic API key
        openai_key: OpenAI API key
        google_key: Google API key
        enable_council: Enable council of experts
        enable_load_balancing: Enable load balancing
        **kwargs: Additional config options

    Returns:
        Configured MultiLLMClient instance
    """
    config = LLMConfig(
        anthropic_api_key=anthropic_key,
        openai_api_key=openai_key,
        google_api_key=google_key,
        enable_council=enable_council,
        enable_load_balancing=enable_load_balancing,
        **kwargs
    )
    return MultiLLMClient(config)
