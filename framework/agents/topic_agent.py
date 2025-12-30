"""
TopicAgent - Topic Classification for Knowledge Distillation

This module implements the TopicAgent that identifies document topics:
- Topic modeling using BERTopic, KeyBERT, or LLM
- Zero-shot classification for predefined topics
- Multi-label topic assignment
- Topic hierarchy support
- Integration with Brain Layer for storing topics

The TopicAgent determines main topics/categories present in text, enabling
filtered retrieval and knowledge organization.

Strategic Intent:
TopicAgent labels documents and agent runs with topics (e.g., "Database Migration",
"DevOps", "Security"). These topic labels help classify knowledge for filtered
retrieval - if someone queries about security, we can prioritize security-labeled
documents.

Adapted from patterns in:
- brain_layer/03_knowledge_distillation/topic_modeling_agent.py (BERTopic)
- brain_layer/04_data_models/models.py (DocumentTopic table)
- brain_layer/03_knowledge_distillation/librarian_agent.py (orchestration)
"""

import asyncio
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from framework.agents.base_agent import (
    BaseAgent,
    AgentTask,
    AgentResponse,
    AgentConfig,
    AgentContext,
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

class TopicModelType(Enum):
    """Supported topic modeling approaches."""
    BERTOPIC = "bertopic"
    KEYBERT = "keybert"
    ZERO_SHOT = "zero_shot"  # Zero-shot classification
    LLM = "llm"  # LLM-based classification
    TFIDF = "tfidf"  # Simple TF-IDF keywords


class ClassificationMode(Enum):
    """How to assign topics."""
    SINGLE = "single"  # Assign one primary topic
    MULTI = "multi"    # Assign multiple topics
    HIERARCHICAL = "hierarchical"  # Topic with subtopics


@dataclass
class TopicConfig:
    """Configuration for TopicAgent."""

    # Model settings
    model: TopicModelType = TopicModelType.KEYBERT
    fallback_models: List[TopicModelType] = field(
        default_factory=lambda: [
            TopicModelType.TFIDF,
            TopicModelType.LLM,
        ]
    )

    # Classification settings
    mode: ClassificationMode = ClassificationMode.MULTI
    max_topics: int = 5
    min_probability: float = 0.3
    min_keyword_score: float = 0.3

    # Predefined topic categories (for zero-shot/LLM)
    predefined_topics: Optional[List[str]] = None

    # KeyBERT settings
    keyphrase_ngram_range: Tuple[int, int] = (1, 3)
    use_mmr: bool = True
    diversity: float = 0.5
    top_n_keywords: int = 10

    # Processing
    min_content_length: int = 50
    max_content_length: int = 50000

    # Brain Layer integration
    enable_brain_logging: bool = True
    store_topics: bool = True


@dataclass
class Topic:
    """A single identified topic."""

    topic_id: int
    name: str
    keywords: List[str]
    probability: float = 1.0
    parent_topic: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "topic_id": self.topic_id,
            "name": self.name,
            "keywords": self.keywords,
            "probability": self.probability,
            "parent_topic": self.parent_topic,
            "metadata": self.metadata,
        }


@dataclass
class TopicExtractionResult:
    """Result from topic extraction."""

    source_id: str
    topics: List[Topic]
    primary_topic: Optional[Topic] = None
    all_keywords: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    model_used: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source_id": self.source_id,
            "topics": [t.to_dict() for t in self.topics],
            "primary_topic": self.primary_topic.to_dict() if self.primary_topic else None,
            "all_keywords": self.all_keywords,
            "processing_time_ms": self.processing_time_ms,
            "model_used": self.model_used,
            "metadata": self.metadata,
        }


# =============================================================================
# TopicAgent Implementation
# =============================================================================

class TopicAgent(BaseAgent):
    """
    Knowledge Distillation Agent for topic classification.

    The TopicAgent identifies and classifies document topics:
    - Uses KeyBERT for keyword-based topics
    - BERTopic for unsupervised topic modeling
    - Zero-shot classification for predefined categories
    - LLM-based classification as fallback

    Usage:
        topic_agent = TopicAgent(config=config)
        result = await topic_agent.analyze_topics(
            content="This document discusses database migration strategies...",
            source_id="doc_123",
        )
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        topic_config: Optional[TopicConfig] = None,
        context: Optional[AgentContext] = None,
        brain_logger: Optional[BrainLogger] = None,
    ):
        super().__init__(config=config, context=context)

        self.topic_config = topic_config or TopicConfig()
        self.brain_logger = brain_logger or get_brain_logger()

        # Lazy-loaded models
        self._keybert = None
        self._bertopic = None
        self._classifier = None
        self._model_lock = asyncio.Lock()

        # Statistics
        self._stats = {
            "total_analyses": 0,
            "topics_extracted": 0,
            "average_time_ms": 0.0,
            "topic_frequency": {},
        }

    async def _get_keybert(self):
        """Lazy load KeyBERT model."""
        if self._keybert is not None:
            return self._keybert

        async with self._model_lock:
            if self._keybert is not None:
                return self._keybert

            try:
                from keybert import KeyBERT

                logger.info("Loading KeyBERT model")

                loop = asyncio.get_event_loop()
                self._keybert = await loop.run_in_executor(None, KeyBERT)

                logger.info("KeyBERT model loaded")
                return self._keybert

            except ImportError:
                logger.warning("KeyBERT not available")
                return None
            except Exception as e:
                logger.warning(f"Failed to load KeyBERT: {e}")
                return None

    async def _get_bertopic(self):
        """Lazy load BERTopic model."""
        if self._bertopic is not None:
            return self._bertopic

        async with self._model_lock:
            if self._bertopic is not None:
                return self._bertopic

            try:
                from bertopic import BERTopic
                from sklearn.feature_extraction.text import CountVectorizer

                logger.info("Loading BERTopic model")

                vectorizer = CountVectorizer(stop_words="english")

                loop = asyncio.get_event_loop()
                self._bertopic = await loop.run_in_executor(
                    None,
                    lambda: BERTopic(vectorizer_model=vectorizer)
                )

                logger.info("BERTopic model loaded")
                return self._bertopic

            except ImportError:
                logger.warning("BERTopic not available")
                return None
            except Exception as e:
                logger.warning(f"Failed to load BERTopic: {e}")
                return None

    # =========================================================================
    # Core Topic Analysis Interface
    # =========================================================================

    async def analyze_topics(
        self,
        content: str,
        source_id: Optional[str] = None,
        predefined_topics: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TopicExtractionResult:
        """
        Analyze and extract topics from content.

        Args:
            content: Text content to analyze
            source_id: Optional ID of source document/run
            predefined_topics: Optional list of predefined topic categories
            metadata: Optional metadata to attach

        Returns:
            TopicExtractionResult with identified topics
        """
        start_time = time.time()

        # Generate source ID if not provided
        if source_id is None:
            source_id = hashlib.md5(content.encode()).hexdigest()[:16]

        # Check content length
        if len(content.strip()) < self.topic_config.min_content_length:
            return TopicExtractionResult(
                source_id=source_id,
                topics=[],
                processing_time_ms=(time.time() - start_time) * 1000,
                model_used="none",
                metadata={"reason": "content_too_short"},
            )

        # Truncate if too long
        if len(content) > self.topic_config.max_content_length:
            content = content[:self.topic_config.max_content_length]

        # Log start
        run_id = str(uuid.uuid4())
        if self.topic_config.enable_brain_logging and self.brain_logger:
            await self._log_analysis_start(run_id, source_id)

        try:
            # Use predefined topics if provided
            topics_to_use = predefined_topics or self.topic_config.predefined_topics

            # Try models in order
            topics, keywords, model_used = await self._analyze_with_fallback(
                content, topics_to_use
            )

            # Determine primary topic
            primary_topic = topics[0] if topics else None

            processing_time = (time.time() - start_time) * 1000

            result = TopicExtractionResult(
                source_id=source_id,
                topics=topics,
                primary_topic=primary_topic,
                all_keywords=keywords,
                processing_time_ms=processing_time,
                model_used=model_used,
                metadata=metadata or {},
            )

            # Update stats
            self._update_stats(result)

            # Store in Brain Layer
            if self.topic_config.store_topics and self.brain_logger:
                await self._store_topics(source_id, topics)

            # Log completion
            if self.topic_config.enable_brain_logging and self.brain_logger:
                await self._log_analysis_complete(run_id, result)

            return result

        except Exception as e:
            logger.error(f"Topic analysis failed: {e}")

            if self.topic_config.enable_brain_logging and self.brain_logger:
                await self._log_analysis_error(run_id, str(e))

            return TopicExtractionResult(
                source_id=source_id,
                topics=[],
                processing_time_ms=(time.time() - start_time) * 1000,
                model_used="error",
                metadata={"error": str(e), **(metadata or {})},
            )

    async def _analyze_with_fallback(
        self,
        content: str,
        predefined_topics: Optional[List[str]],
    ) -> Tuple[List[Topic], List[str], str]:
        """Try topic analysis models in priority order."""
        models_to_try = [self.topic_config.model] + self.topic_config.fallback_models

        for model in models_to_try:
            try:
                if model == TopicModelType.KEYBERT:
                    result = await self._analyze_with_keybert(content)
                    if result[0]:
                        return result

                elif model == TopicModelType.BERTOPIC:
                    result = await self._analyze_with_bertopic(content)
                    if result[0]:
                        return result

                elif model == TopicModelType.ZERO_SHOT and predefined_topics:
                    result = await self._analyze_with_zero_shot(content, predefined_topics)
                    if result[0]:
                        return result

                elif model == TopicModelType.LLM:
                    result = await self._analyze_with_llm(content, predefined_topics)
                    if result[0]:
                        return result

                elif model == TopicModelType.TFIDF:
                    result = await self._analyze_with_tfidf(content)
                    if result[0]:
                        return result

            except Exception as e:
                logger.warning(f"Topic analysis with {model.value} failed: {e}")
                continue

        return [], [], "none"

    async def _analyze_with_keybert(
        self,
        content: str,
    ) -> Tuple[List[Topic], List[str], str]:
        """Extract topics using KeyBERT keywords."""
        keybert = await self._get_keybert()
        if keybert is None:
            return [], [], ""

        try:
            loop = asyncio.get_event_loop()
            keywords = await loop.run_in_executor(
                None,
                lambda: keybert.extract_keywords(
                    content,
                    keyphrase_ngram_range=self.topic_config.keyphrase_ngram_range,
                    stop_words="english",
                    use_mmr=self.topic_config.use_mmr,
                    diversity=self.topic_config.diversity,
                    top_n=self.topic_config.top_n_keywords,
                )
            )

            # Filter by score
            filtered_keywords = [
                (kw, score) for kw, score in keywords
                if score >= self.topic_config.min_keyword_score
            ]

            if not filtered_keywords:
                return [], [], ""

            # Create topics from keywords
            topics = []
            all_keywords = []

            for i, (kw, score) in enumerate(filtered_keywords[:self.topic_config.max_topics]):
                topics.append(Topic(
                    topic_id=i,
                    name=kw.title(),
                    keywords=[kw],
                    probability=score,
                ))
                all_keywords.append(kw)

            return topics, all_keywords, "keybert"

        except Exception as e:
            logger.warning(f"KeyBERT analysis failed: {e}")
            return [], [], ""

    async def _analyze_with_bertopic(
        self,
        content: str,
    ) -> Tuple[List[Topic], List[str], str]:
        """Extract topics using BERTopic."""
        bertopic = await self._get_bertopic()
        if bertopic is None:
            return [], [], ""

        try:
            loop = asyncio.get_event_loop()

            # BERTopic needs multiple documents, use sentences as documents
            sentences = [s.strip() for s in content.split(".") if len(s.strip()) > 20]

            if len(sentences) < 3:
                return [], [], ""

            topic_ids, _ = await loop.run_in_executor(
                None,
                lambda: bertopic.fit_transform(sentences)
            )

            topic_info = bertopic.get_topic_info()

            topics = []
            all_keywords = []

            for _, row in topic_info.iterrows():
                if row["Topic"] == -1:  # Outlier topic
                    continue

                keywords = row.get("Representation", [])
                if isinstance(keywords, str):
                    keywords = [keywords]

                topics.append(Topic(
                    topic_id=int(row["Topic"]),
                    name=row.get("Name", f"Topic {row['Topic']}"),
                    keywords=keywords[:5] if keywords else [],
                    probability=1.0,
                ))

                all_keywords.extend(keywords[:5] if keywords else [])

                if len(topics) >= self.topic_config.max_topics:
                    break

            return topics, list(set(all_keywords)), "bertopic"

        except Exception as e:
            logger.warning(f"BERTopic analysis failed: {e}")
            return [], [], ""

    async def _analyze_with_zero_shot(
        self,
        content: str,
        categories: List[str],
    ) -> Tuple[List[Topic], List[str], str]:
        """Classify into predefined categories using zero-shot."""
        try:
            from transformers import pipeline

            loop = asyncio.get_event_loop()

            classifier = await loop.run_in_executor(
                None,
                lambda: pipeline("zero-shot-classification")
            )

            result = await loop.run_in_executor(
                None,
                lambda: classifier(
                    content[:1000],  # Limit input length
                    categories,
                    multi_label=self.topic_config.mode == ClassificationMode.MULTI,
                )
            )

            topics = []
            for i, (label, score) in enumerate(zip(result["labels"], result["scores"])):
                if score >= self.topic_config.min_probability:
                    topics.append(Topic(
                        topic_id=i,
                        name=label,
                        keywords=[label.lower()],
                        probability=score,
                    ))

                if len(topics) >= self.topic_config.max_topics:
                    break

            keywords = [t.name.lower() for t in topics]
            return topics, keywords, "zero_shot"

        except Exception as e:
            logger.warning(f"Zero-shot classification failed: {e}")
            return [], [], ""

    async def _analyze_with_llm(
        self,
        content: str,
        predefined_topics: Optional[List[str]],
    ) -> Tuple[List[Topic], List[str], str]:
        """Extract topics using LLM."""
        try:
            if predefined_topics:
                topics_hint = f"Choose from these categories if applicable: {', '.join(predefined_topics)}"
            else:
                topics_hint = "Identify the main topics/themes"

            prompt = f"""Analyze the following text and identify the main topics.
{topics_hint}

For each topic, provide:
- Topic name (2-4 words)
- 3-5 keywords
- Confidence score (0.0-1.0)

Text:
{content[:3000]}

Format your response as:
TOPIC_NAME | keyword1, keyword2, keyword3 | SCORE

Example:
Machine Learning | neural networks, deep learning, training | 0.9"""

            response = await self._call_llm(prompt)

            if not response or not response.output:
                return [], [], ""

            topics = []
            all_keywords = []

            for i, line in enumerate(response.output.strip().split("\n")):
                if "|" not in line:
                    continue

                parts = line.split("|")
                if len(parts) < 2:
                    continue

                name = parts[0].strip()
                keywords_str = parts[1].strip() if len(parts) > 1 else ""
                score_str = parts[2].strip() if len(parts) > 2 else "0.8"

                try:
                    score = float(score_str)
                except ValueError:
                    score = 0.8

                if score < self.topic_config.min_probability:
                    continue

                keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]

                topics.append(Topic(
                    topic_id=i,
                    name=name,
                    keywords=keywords,
                    probability=score,
                ))

                all_keywords.extend(keywords)

                if len(topics) >= self.topic_config.max_topics:
                    break

            return topics, list(set(all_keywords)), "llm"

        except Exception as e:
            logger.warning(f"LLM topic analysis failed: {e}")
            return [], [], ""

    async def _analyze_with_tfidf(
        self,
        content: str,
    ) -> Tuple[List[Topic], List[str], str]:
        """Simple TF-IDF keyword extraction as fallback."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=20,
            )

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: vectorizer.fit_transform([content])
            )

            feature_names = vectorizer.get_feature_names_out().tolist()

            if not feature_names:
                return [], [], ""

            # Create a single topic from keywords
            topics = [Topic(
                topic_id=0,
                name="Keywords",
                keywords=feature_names[:self.topic_config.top_n_keywords],
                probability=1.0,
            )]

            return topics, feature_names, "tfidf"

        except Exception as e:
            logger.warning(f"TF-IDF analysis failed: {e}")
            return [], [], ""

    # =========================================================================
    # Brain Layer Integration
    # =========================================================================

    async def _store_topics(
        self,
        source_id: str,
        topics: List[Topic],
    ) -> None:
        """Store topics in Brain Layer."""
        try:
            import importlib

            try:
                supabase_module = importlib.import_module(
                    "brain_layer.08_supabase_pgvector.supabase_client"
                )
                client = getattr(supabase_module, "get_supabase_client", lambda: None)()

                if client and topics:
                    records = []
                    for topic in topics:
                        topic_id_hash = hashlib.md5(
                            f"{source_id}:{topic.name}".encode()
                        ).hexdigest()

                        records.append({
                            "id": topic_id_hash,
                            "document_id": source_id,
                            "topic_id": topic.topic_id,
                            "name": topic.name,
                            "keywords": topic.keywords,
                            "probability": topic.probability,
                            "created_at": datetime.utcnow().isoformat(),
                        })

                    await client.table("document_topics").upsert(records).execute()

            except (ImportError, AttributeError):
                pass

        except Exception as e:
            logger.warning(f"Failed to store topics: {e}")

    async def _log_analysis_start(
        self,
        run_id: str,
        source_id: str,
    ) -> None:
        """Log analysis start."""
        try:
            run_create = AgentRunCreate(
                agent_id=f"topic_{self.agent_id}",
                agent_type="topic_analysis",
                tenant_id=self.context.tenant_id if self.context else None,
                input_summary=f"Analyzing topics for {source_id}",
                metadata={"source_id": source_id},
            )
            await self.brain_logger.log_run_start(run_create)
        except Exception as e:
            logger.warning(f"Failed to log analysis start: {e}")

    async def _log_analysis_complete(
        self,
        run_id: str,
        result: TopicExtractionResult,
    ) -> None:
        """Log analysis completion."""
        try:
            run_update = AgentRunUpdate(
                status=RunStatus.COMPLETED,
                output_summary=f"Found {len(result.topics)} topics",
                metrics={
                    "topic_count": len(result.topics),
                    "processing_time_ms": result.processing_time_ms,
                    "primary_topic": result.primary_topic.name if result.primary_topic else None,
                },
            )
            await self.brain_logger.log_run_update(run_id, run_update)
        except Exception as e:
            logger.warning(f"Failed to log analysis complete: {e}")

    async def _log_analysis_error(
        self,
        run_id: str,
        error: str,
    ) -> None:
        """Log analysis error."""
        try:
            run_update = AgentRunUpdate(
                status=RunStatus.FAILED,
                error_message=error,
            )
            await self.brain_logger.log_run_update(run_id, run_update)
        except Exception as e:
            logger.warning(f"Failed to log analysis error: {e}")

    def _update_stats(self, result: TopicExtractionResult) -> None:
        """Update internal statistics."""
        self._stats["total_analyses"] += 1
        self._stats["topics_extracted"] += len(result.topics)

        # Update topic frequency
        for topic in result.topics:
            if topic.name not in self._stats["topic_frequency"]:
                self._stats["topic_frequency"][topic.name] = 0
            self._stats["topic_frequency"][topic.name] += 1

        # Update average time
        n = self._stats["total_analyses"]
        old_avg = self._stats["average_time_ms"]
        self._stats["average_time_ms"] = old_avg + (result.processing_time_ms - old_avg) / n

    # =========================================================================
    # BaseAgent Interface
    # =========================================================================

    async def _execute_task(self, task: AgentTask) -> AgentResponse:
        """Execute a topic analysis task."""
        content = task.input_data.get("content", "")
        source_id = task.input_data.get("source_id")
        predefined_topics = task.input_data.get("predefined_topics")

        result = await self.analyze_topics(
            content=content,
            source_id=source_id,
            predefined_topics=predefined_topics,
        )

        return AgentResponse(
            task_id=task.task_id,
            status=TaskStatus.COMPLETED,
            output=result.to_dict(),
            confidence=1.0 if result.topics else 0.5,
            metadata={
                "topic_count": len(result.topics),
                "processing_time_ms": result.processing_time_ms,
            },
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get topic analysis statistics."""
        return self._stats.copy()


# =============================================================================
# Factory Function
# =============================================================================

def create_topic_agent(
    config: Optional[AgentConfig] = None,
    topic_config: Optional[TopicConfig] = None,
    context: Optional[AgentContext] = None,
) -> TopicAgent:
    """
    Factory function to create a TopicAgent.

    Args:
        config: Base agent configuration
        topic_config: Topic analysis configuration
        context: Agent execution context

    Returns:
        Configured TopicAgent instance
    """
    return TopicAgent(
        config=config,
        topic_config=topic_config,
        context=context,
    )
