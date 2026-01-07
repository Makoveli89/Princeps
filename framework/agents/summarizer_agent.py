"""
SummarizerAgent - Text Summarization for Knowledge Distillation

This module implements the SummarizerAgent that generates concise summaries:
- Multi-level summaries (one-sentence, executive, detailed)
- Support for BART, T5, and LLM-based summarization
- Automatic model fallback chain
- Integration with Brain Layer for storing summaries
- Batch processing for efficiency

The SummarizerAgent converts lengthy documents and agent run transcripts
into concise, queryable knowledge atoms suitable for retrieval.

Strategic Intent:
After a document is ingested or a complex task finishes, the SummarizerAgent
produces summaries that capture the main points. These summaries are stored
in the Brain's document_summaries table and can be retrieved quickly without
reading full documents.

Adapted from patterns in:
- brain_layer/03_knowledge_distillation/summarization_agent.py (BART pipeline)
- brain_layer/03_knowledge_distillation/librarian_agent.py (orchestration)
"""

import asyncio
import hashlib
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from framework.agents.base_agent import (
    AgentConfig,
    AgentContext,
    AgentResponse,
    AgentTask,
    BaseAgent,
    TaskStatus,
)
from framework.agents.brain_logger import BrainLogger, get_brain_logger
from framework.agents.schemas.agent_run import (
    AgentRunCreate,
    AgentRunUpdate,
    RunStatus,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Configuration
# =============================================================================


class SummarizationModel(Enum):
    """Supported summarization models."""

    BART_LARGE_CNN = "facebook/bart-large-cnn"
    DISTILBART_CNN = "sshleifer/distilbart-cnn-12-6"
    T5_BASE = "t5-base"
    T5_SMALL = "t5-small"
    PEGASUS = "google/pegasus-xsum"
    LLM = "llm"  # Use configured LLM provider


class SummaryLevel(Enum):
    """Levels of summary detail."""

    ONE_SENTENCE = "one_sentence"  # ~20-30 words
    EXECUTIVE = "executive"  # ~50-100 words
    DETAILED = "detailed"  # ~150-300 words
    BULLET_POINTS = "bullet_points"  # Key points as list
    ABSTRACT = "abstract"  # Academic-style abstract


@dataclass
class SummarizerConfig:
    """Configuration for the SummarizerAgent."""

    # Model settings
    model: SummarizationModel = SummarizationModel.BART_LARGE_CNN
    fallback_models: list[SummarizationModel] = field(
        default_factory=lambda: [
            SummarizationModel.DISTILBART_CNN,
            SummarizationModel.T5_SMALL,
            SummarizationModel.LLM,
        ]
    )

    # Summary length settings
    one_sentence_max_length: int = 60
    one_sentence_min_length: int = 20
    executive_max_length: int = 150
    executive_min_length: int = 50
    detailed_max_length: int = 400
    detailed_min_length: int = 150

    # Processing settings
    min_input_words: int = 50  # Minimum words to trigger summarization
    max_input_tokens: int = 1024  # Max input tokens for model
    chunk_overlap: int = 100  # Token overlap for chunking long texts

    # Quality settings
    do_sample: bool = False  # Deterministic output
    temperature: float = 0.7  # For LLM-based summarization
    num_beams: int = 4  # Beam search

    # Caching
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600

    # Brain Layer integration
    enable_brain_logging: bool = True
    store_summaries: bool = True


@dataclass
class SummaryResult:
    """Result from summarization."""

    source_id: str  # Document or run ID
    one_sentence: str | None = None
    executive: str | None = None
    detailed: str | None = None
    bullet_points: list[str] | None = None
    abstract: str | None = None
    model_used: str = ""
    processing_time_ms: float = 0.0
    input_word_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source_id": self.source_id,
            "one_sentence": self.one_sentence,
            "executive": self.executive,
            "detailed": self.detailed,
            "bullet_points": self.bullet_points,
            "abstract": self.abstract,
            "model_used": self.model_used,
            "processing_time_ms": self.processing_time_ms,
            "input_word_count": self.input_word_count,
            "metadata": self.metadata,
        }


# =============================================================================
# Summary Cache
# =============================================================================


class SummaryCache:
    """Cache for computed summaries."""

    def __init__(
        self,
        max_size: int = 500,
        ttl_seconds: int = 3600,
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[SummaryResult, float]] = {}
        self._lock = asyncio.Lock()

    def _compute_key(self, content: str, levels: list[SummaryLevel]) -> str:
        """Compute cache key from content and requested levels."""
        levels_str = ",".join(sorted(l.value for l in levels))
        key_content = f"{content[:1000]}:{levels_str}"
        return hashlib.sha256(key_content.encode()).hexdigest()

    async def get(
        self,
        content: str,
        levels: list[SummaryLevel],
    ) -> SummaryResult | None:
        """Get cached summary if available."""
        key = self._compute_key(content, levels)

        async with self._lock:
            if key not in self._cache:
                return None

            result, timestamp = self._cache[key]

            if time.time() - timestamp > self.ttl_seconds:
                del self._cache[key]
                return None

            return result

    async def set(
        self,
        content: str,
        levels: list[SummaryLevel],
        result: SummaryResult,
    ) -> None:
        """Cache a summary result."""
        key = self._compute_key(content, levels)

        async with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= self.max_size:
                oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]

            self._cache[key] = (result, time.time())

    async def clear(self) -> None:
        """Clear the cache."""
        async with self._lock:
            self._cache.clear()


# =============================================================================
# SummarizerAgent Implementation
# =============================================================================


class SummarizerAgent(BaseAgent):
    """
    Knowledge Distillation Agent for text summarization.

    The SummarizerAgent generates multi-level summaries from text content:
    - One-sentence: Quick overview (~25 words)
    - Executive: Medium summary (~75 words)
    - Detailed: Comprehensive summary (~200 words)
    - Bullet points: Key takeaways as list
    - Abstract: Academic-style abstract

    Usage:
        summarizer = SummarizerAgent(config=config)
        result = await summarizer.summarize(
            content="Long document text...",
            source_id="doc_123",
            levels=[SummaryLevel.ONE_SENTENCE, SummaryLevel.EXECUTIVE],
        )
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        summarizer_config: SummarizerConfig | None = None,
        context: AgentContext | None = None,
        brain_logger: BrainLogger | None = None,
    ):
        super().__init__(config=config, context=context)

        self.summarizer_config = summarizer_config or SummarizerConfig()
        self.brain_logger = brain_logger or get_brain_logger()

        # Initialize cache
        self._cache = (
            SummaryCache(
                ttl_seconds=self.summarizer_config.cache_ttl_seconds,
            )
            if self.summarizer_config.enable_cache
            else None
        )

        # Lazy-loaded summarization pipeline
        self._pipeline = None
        self._pipeline_model = None
        self._pipeline_lock = asyncio.Lock()

        # Statistics
        self._stats = {
            "total_summaries": 0,
            "cache_hits": 0,
            "model_fallbacks": 0,
            "llm_summaries": 0,
            "average_time_ms": 0.0,
        }

    async def _get_pipeline(self, model: SummarizationModel):
        """Lazy load the summarization pipeline."""
        if model == SummarizationModel.LLM:
            return None  # LLM doesn't use pipeline

        async with self._pipeline_lock:
            if self._pipeline is not None and self._pipeline_model == model:
                return self._pipeline

            try:
                from transformers import pipeline

                hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")

                logger.info(f"Loading summarization model: {model.value}")

                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                self._pipeline = await loop.run_in_executor(
                    None,
                    lambda: pipeline(
                        "summarization",
                        model=model.value,
                        token=hf_token,
                    ),
                )
                self._pipeline_model = model

                logger.info(f"Loaded summarization model: {model.value}")
                return self._pipeline

            except ImportError:
                logger.warning("transformers not available for summarization")
                return None
            except Exception as e:
                logger.warning(f"Failed to load model {model.value}: {e}")
                return None

    # =========================================================================
    # Core Summarization Interface
    # =========================================================================

    async def summarize(
        self,
        content: str,
        source_id: str | None = None,
        levels: list[SummaryLevel] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SummaryResult:
        """
        Generate summaries for the given content.

        Args:
            content: Text content to summarize
            source_id: Optional ID of source document/run
            levels: Summary levels to generate (default: one_sentence, executive)
            metadata: Optional metadata to attach to result

        Returns:
            SummaryResult with requested summaries
        """
        start_time = time.time()

        # Default levels
        if levels is None:
            levels = [SummaryLevel.ONE_SENTENCE, SummaryLevel.EXECUTIVE]

        # Generate source ID if not provided
        if source_id is None:
            source_id = hashlib.md5(content.encode()).hexdigest()[:16]

        # Check cache
        if self._cache:
            cached = await self._cache.get(content, levels)
            if cached:
                self._stats["cache_hits"] += 1
                return cached

        # Check minimum input length
        word_count = len(content.split())
        if word_count < self.summarizer_config.min_input_words:
            return SummaryResult(
                source_id=source_id,
                one_sentence=content[:150] if SummaryLevel.ONE_SENTENCE in levels else None,
                executive=content if SummaryLevel.EXECUTIVE in levels else None,
                model_used="passthrough",
                input_word_count=word_count,
                metadata=metadata or {},
            )

        # Log start
        run_id = str(uuid.uuid4())
        if self.summarizer_config.enable_brain_logging and self.brain_logger:
            await self._log_summary_start(run_id, source_id, levels)

        try:
            # Try models in order
            result = await self._generate_summaries(
                content=content,
                source_id=source_id,
                levels=levels,
                metadata=metadata or {},
            )

            processing_time = (time.time() - start_time) * 1000
            result.processing_time_ms = processing_time
            result.input_word_count = word_count

            # Update cache
            if self._cache:
                await self._cache.set(content, levels, result)

            # Update stats
            self._update_stats(result)

            # Store in Brain Layer
            if self.summarizer_config.store_summaries and self.brain_logger:
                await self._store_summary(source_id, result)

            # Log completion
            if self.summarizer_config.enable_brain_logging and self.brain_logger:
                await self._log_summary_complete(run_id, result)

            return result

        except Exception as e:
            logger.error(f"Summarization failed: {e}")

            if self.summarizer_config.enable_brain_logging and self.brain_logger:
                await self._log_summary_error(run_id, str(e))

            # Return truncation fallback
            return SummaryResult(
                source_id=source_id,
                one_sentence=content[:150] if SummaryLevel.ONE_SENTENCE in levels else None,
                executive=content[:500] if SummaryLevel.EXECUTIVE in levels else None,
                model_used="truncation",
                processing_time_ms=(time.time() - start_time) * 1000,
                input_word_count=word_count,
                metadata={"error": str(e), **(metadata or {})},
            )

    async def _generate_summaries(
        self,
        content: str,
        source_id: str,
        levels: list[SummaryLevel],
        metadata: dict[str, Any],
    ) -> SummaryResult:
        """Generate summaries using available models."""
        # Try models in priority order
        models_to_try = [self.summarizer_config.model] + self.summarizer_config.fallback_models
        model_used = None

        for model in models_to_try:
            if model == SummarizationModel.LLM:
                # Use LLM-based summarization
                result = await self._summarize_with_llm(content, source_id, levels, metadata)
                if result:
                    self._stats["llm_summaries"] += 1
                    return result
            else:
                # Use transformer pipeline
                pipeline = await self._get_pipeline(model)
                if pipeline:
                    result = await self._summarize_with_pipeline(
                        content, source_id, levels, metadata, pipeline, model
                    )
                    if result:
                        return result

            self._stats["model_fallbacks"] += 1

        # Fallback to truncation
        return SummaryResult(
            source_id=source_id,
            one_sentence=content[:150] if SummaryLevel.ONE_SENTENCE in levels else None,
            executive=content[:500] if SummaryLevel.EXECUTIVE in levels else None,
            model_used="truncation",
            metadata=metadata,
        )

    async def _summarize_with_pipeline(
        self,
        content: str,
        source_id: str,
        levels: list[SummaryLevel],
        metadata: dict[str, Any],
        pipeline: Any,
        model: SummarizationModel,
    ) -> SummaryResult | None:
        """Generate summaries using transformer pipeline."""
        try:
            result = SummaryResult(
                source_id=source_id,
                model_used=model.value,
                metadata=metadata,
            )

            loop = asyncio.get_event_loop()

            # Generate each requested level
            if SummaryLevel.ONE_SENTENCE in levels:
                summary = await loop.run_in_executor(
                    None,
                    lambda: pipeline(
                        content,
                        max_length=self.summarizer_config.one_sentence_max_length,
                        min_length=self.summarizer_config.one_sentence_min_length,
                        do_sample=self.summarizer_config.do_sample,
                        num_beams=self.summarizer_config.num_beams,
                    ),
                )
                result.one_sentence = summary[0]["summary_text"]

            if SummaryLevel.EXECUTIVE in levels:
                summary = await loop.run_in_executor(
                    None,
                    lambda: pipeline(
                        content,
                        max_length=self.summarizer_config.executive_max_length,
                        min_length=self.summarizer_config.executive_min_length,
                        do_sample=self.summarizer_config.do_sample,
                        num_beams=self.summarizer_config.num_beams,
                    ),
                )
                result.executive = summary[0]["summary_text"]

            if SummaryLevel.DETAILED in levels:
                summary = await loop.run_in_executor(
                    None,
                    lambda: pipeline(
                        content,
                        max_length=self.summarizer_config.detailed_max_length,
                        min_length=self.summarizer_config.detailed_min_length,
                        do_sample=self.summarizer_config.do_sample,
                        num_beams=self.summarizer_config.num_beams,
                    ),
                )
                result.detailed = summary[0]["summary_text"]

            return result

        except Exception as e:
            logger.warning(f"Pipeline summarization failed: {e}")
            return None

    async def _summarize_with_llm(
        self,
        content: str,
        source_id: str,
        levels: list[SummaryLevel],
        metadata: dict[str, Any],
    ) -> SummaryResult | None:
        """Generate summaries using LLM."""
        try:
            result = SummaryResult(
                source_id=source_id,
                model_used="llm",
                metadata=metadata,
            )

            # Build prompts for each level
            if SummaryLevel.ONE_SENTENCE in levels:
                prompt = f"""Summarize the following text in exactly one sentence (20-30 words):

{content[:4000]}

One-sentence summary:"""
                response = await self._call_llm(prompt)
                if response and response.output:
                    result.one_sentence = response.output.strip()

            if SummaryLevel.EXECUTIVE in levels:
                prompt = f"""Write an executive summary of the following text in 2-3 sentences (50-100 words):

{content[:4000]}

Executive summary:"""
                response = await self._call_llm(prompt)
                if response and response.output:
                    result.executive = response.output.strip()

            if SummaryLevel.DETAILED in levels:
                prompt = f"""Write a detailed summary of the following text in 1-2 paragraphs (150-300 words):

{content[:6000]}

Detailed summary:"""
                response = await self._call_llm(prompt)
                if response and response.output:
                    result.detailed = response.output.strip()

            if SummaryLevel.BULLET_POINTS in levels:
                prompt = f"""Extract the 5-7 most important points from the following text as bullet points:

{content[:4000]}

Key points:"""
                response = await self._call_llm(prompt)
                if response and response.output:
                    # Parse bullet points
                    lines = response.output.strip().split("\n")
                    result.bullet_points = [
                        line.lstrip("•-*").strip()
                        for line in lines
                        if line.strip() and len(line.strip()) > 5
                    ]

            if SummaryLevel.ABSTRACT in levels:
                prompt = f"""Write an academic-style abstract for the following text (150-250 words). Include: context, main findings, and implications:

{content[:6000]}

Abstract:"""
                response = await self._call_llm(prompt)
                if response and response.output:
                    result.abstract = response.output.strip()

            return result

        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}")
            return None

    # =========================================================================
    # Batch Processing
    # =========================================================================

    async def summarize_batch(
        self,
        items: list[tuple[str, str]],  # [(content, source_id), ...]
        levels: list[SummaryLevel] | None = None,
        max_concurrent: int = 5,
    ) -> list[SummaryResult]:
        """
        Summarize multiple documents in parallel.

        Args:
            items: List of (content, source_id) tuples
            levels: Summary levels to generate
            max_concurrent: Maximum concurrent summarizations

        Returns:
            List of SummaryResult in same order as input
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_one(content: str, source_id: str) -> SummaryResult:
            async with semaphore:
                return await self.summarize(content, source_id, levels)

        tasks = [process_one(content, source_id) for content, source_id in items]

        return await asyncio.gather(*tasks)

    # =========================================================================
    # Brain Layer Integration
    # =========================================================================

    async def _store_summary(
        self,
        source_id: str,
        result: SummaryResult,
    ) -> None:
        """Store summary in Brain Layer."""
        try:
            # Import dynamically to avoid circular imports
            import importlib

            try:
                supabase_module = importlib.import_module(
                    "brain_layer.08_supabase_pgvector.supabase_client"
                )
                client = getattr(supabase_module, "get_supabase_client", lambda: None)()

                if client:
                    # Insert into document_summaries table
                    summary_id = hashlib.md5(f"{source_id}:{time.time()}".encode()).hexdigest()

                    await client.table("document_summaries").upsert(
                        {
                            "id": summary_id,
                            "document_id": source_id,
                            "one_sentence": result.one_sentence,
                            "executive": result.executive,
                            "model_used": result.model_used,
                            "created_at": datetime.utcnow().isoformat(),
                        }
                    ).execute()

            except (ImportError, AttributeError):
                pass  # Brain Layer not available

        except Exception as e:
            logger.warning(f"Failed to store summary: {e}")

    async def _log_summary_start(
        self,
        run_id: str,
        source_id: str,
        levels: list[SummaryLevel],
    ) -> None:
        """Log summarization start."""
        try:
            run_create = AgentRunCreate(
                agent_id=f"summarizer_{self.agent_id}",
                agent_type="summarizer",
                tenant_id=self.context.tenant_id if self.context else None,
                input_summary=f"Summarizing {source_id}",
                metadata={
                    "source_id": source_id,
                    "levels": [l.value for l in levels],
                },
            )
            await self.brain_logger.log_run_start(run_create)
        except Exception as e:
            logger.warning(f"Failed to log summary start: {e}")

    async def _log_summary_complete(
        self,
        run_id: str,
        result: SummaryResult,
    ) -> None:
        """Log summarization completion."""
        try:
            run_update = AgentRunUpdate(
                status=RunStatus.COMPLETED,
                output_summary=result.one_sentence or "Summary generated",
                metrics={
                    "processing_time_ms": result.processing_time_ms,
                    "input_word_count": result.input_word_count,
                    "model_used": result.model_used,
                },
            )
            await self.brain_logger.log_run_update(run_id, run_update)
        except Exception as e:
            logger.warning(f"Failed to log summary complete: {e}")

    async def _log_summary_error(
        self,
        run_id: str,
        error: str,
    ) -> None:
        """Log summarization error."""
        try:
            run_update = AgentRunUpdate(
                status=RunStatus.FAILED,
                error_message=error,
            )
            await self.brain_logger.log_run_update(run_id, run_update)
        except Exception as e:
            logger.warning(f"Failed to log summary error: {e}")

    def _update_stats(self, result: SummaryResult) -> None:
        """Update internal statistics."""
        self._stats["total_summaries"] += 1

        n = self._stats["total_summaries"]
        old_avg = self._stats["average_time_ms"]
        self._stats["average_time_ms"] = old_avg + (result.processing_time_ms - old_avg) / n

    # =========================================================================
    # BaseAgent Interface
    # =========================================================================

    async def _execute_task(self, task: AgentTask) -> AgentResponse:
        """Execute a summarization task."""
        content = task.input_data.get("content", "")
        source_id = task.input_data.get("source_id")
        levels_str = task.input_data.get("levels", ["one_sentence", "executive"])

        levels = []
        for level_str in levels_str:
            try:
                levels.append(SummaryLevel(level_str))
            except ValueError:
                pass

        if not levels:
            levels = [SummaryLevel.ONE_SENTENCE, SummaryLevel.EXECUTIVE]

        result = await self.summarize(
            content=content,
            source_id=source_id,
            levels=levels,
        )

        return AgentResponse(
            task_id=task.task_id,
            status=TaskStatus.COMPLETED,
            output=result.to_dict(),
            confidence=1.0 if result.one_sentence else 0.5,
            metadata={
                "processing_time_ms": result.processing_time_ms,
                "model_used": result.model_used,
            },
        )

    def get_stats(self) -> dict[str, Any]:
        """Get summarizer statistics."""
        return self._stats.copy()


# =============================================================================
# Factory Function
# =============================================================================


def create_summarizer_agent(
    config: AgentConfig | None = None,
    summarizer_config: SummarizerConfig | None = None,
    context: AgentContext | None = None,
) -> SummarizerAgent:
    """
    Factory function to create a SummarizerAgent.

    Args:
        config: Base agent configuration
        summarizer_config: Summarizer-specific configuration
        context: Agent execution context

    Returns:
        Configured SummarizerAgent instance
    """
    return SummarizerAgent(
        config=config,
        summarizer_config=summarizer_config,
        context=context,
    )
