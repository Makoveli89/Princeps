"""
EntityExtractionAgent - Named Entity Recognition for Knowledge Distillation

This module implements the EntityExtractionAgent that extracts named entities:
- Person, Organization, Location, Date, Money, etc.
- Support for spaCy NER and LLM-based extraction
- Entity deduplication and normalization
- Confidence scoring
- Integration with Brain Layer for storing entities

The EntityExtractionAgent identifies and extracts named entities from documents
and agent run transcripts, storing them for later search and analysis.

Strategic Intent:
When text is ingested, EntityExtractionAgent pulls out entities (Person, Org,
Tool, etc.). These entities help in search (e.g., find all documents mentioning
a specific person) and can be linked to knowledge nodes for graph traversal.

Adapted from patterns in:
- brain_layer/03_knowledge_distillation/ner_agent.py (spaCy NER)
- brain_layer/04_data_models/models.py (EntityLabelEnum)
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


class EntityLabel(Enum):
    """Named entity types (aligned with spaCy and models.py)."""

    PERSON = "PERSON"
    ORG = "ORG"
    GPE = "GPE"  # Geo-political entity (country, city, state)
    LOC = "LOC"  # Non-GPE locations
    DATE = "DATE"
    TIME = "TIME"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    WORK_OF_ART = "WORK_OF_ART"
    LAW = "LAW"
    LANGUAGE = "LANGUAGE"
    QUANTITY = "QUANTITY"
    ORDINAL = "ORDINAL"
    CARDINAL = "CARDINAL"
    FAC = "FAC"  # Facility (building, airport, etc.)
    NORP = "NORP"  # Nationalities, religious/political groups
    # Custom types for technical domains
    TECHNOLOGY = "TECHNOLOGY"
    API = "API"
    CODE = "CODE"
    ERROR = "ERROR"
    FILE_PATH = "FILE_PATH"
    URL = "URL"


class NERModel(Enum):
    """Supported NER models."""

    SPACY_SM = "en_core_web_sm"
    SPACY_MD = "en_core_web_md"
    SPACY_LG = "en_core_web_lg"
    SPACY_TRF = "en_core_web_trf"  # Transformer-based
    LLM = "llm"  # Use LLM for extraction


@dataclass
class EntityConfig:
    """Configuration for EntityExtractionAgent."""

    # Model settings
    model: NERModel = NERModel.SPACY_LG
    fallback_models: list[NERModel] = field(
        default_factory=lambda: [
            NERModel.SPACY_MD,
            NERModel.SPACY_SM,
            NERModel.LLM,
        ]
    )

    # Entity filtering
    min_confidence: float = 0.5
    min_entity_length: int = 2
    max_entity_length: int = 100
    deduplicate: bool = True
    normalize_case: bool = True

    # Labels to extract (None = all)
    include_labels: list[EntityLabel] | None = None
    exclude_labels: list[EntityLabel] | None = field(
        default_factory=lambda: [EntityLabel.CARDINAL, EntityLabel.ORDINAL]
    )

    # PII handling
    flag_pii: bool = True
    pii_labels: list[EntityLabel] = field(
        default_factory=lambda: [EntityLabel.PERSON, EntityLabel.MONEY]
    )

    # Processing
    max_text_length: int = 100000  # Max characters to process

    # Brain Layer integration
    enable_brain_logging: bool = True
    store_entities: bool = True


@dataclass
class ExtractedEntity:
    """A single extracted entity."""

    text: str
    label: EntityLabel
    start_char: int
    end_char: int
    confidence: float = 1.0
    is_pii: bool = False
    normalized_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "text": self.text,
            "label": self.label.value,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "confidence": self.confidence,
            "is_pii": self.is_pii,
            "normalized_text": self.normalized_text,
            "metadata": self.metadata,
        }


@dataclass
class EntityExtractionResult:
    """Result from entity extraction."""

    source_id: str
    entities: list[ExtractedEntity]
    entities_by_label: dict[str, list[str]]  # label -> [entity texts]
    total_entities: int
    unique_entities: int
    processing_time_ms: float = 0.0
    model_used: str = ""
    pii_detected: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source_id": self.source_id,
            "entities": [e.to_dict() for e in self.entities],
            "entities_by_label": self.entities_by_label,
            "total_entities": self.total_entities,
            "unique_entities": self.unique_entities,
            "processing_time_ms": self.processing_time_ms,
            "model_used": self.model_used,
            "pii_detected": self.pii_detected,
            "metadata": self.metadata,
        }


# =============================================================================
# EntityExtractionAgent Implementation
# =============================================================================


class EntityExtractionAgent(BaseAgent):
    """
    Knowledge Distillation Agent for Named Entity Recognition.

    The EntityExtractionAgent identifies and extracts named entities:
    - Uses spaCy for accurate NER
    - Falls back to LLM-based extraction
    - Normalizes and deduplicates entities
    - Flags potential PII
    - Stores entities in Brain Layer

    Usage:
        entity_agent = EntityExtractionAgent(config=config)
        result = await entity_agent.extract_entities(
            content="John Smith from Microsoft announced...",
            source_id="doc_123",
        )
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        entity_config: EntityConfig | None = None,
        context: AgentContext | None = None,
        brain_logger: BrainLogger | None = None,
    ):
        super().__init__(config=config, context=context)

        self.entity_config = entity_config or EntityConfig()
        self.brain_logger = brain_logger or get_brain_logger()

        # Lazy-loaded spaCy model
        self._nlp = None
        self._nlp_model = None
        self._nlp_lock = asyncio.Lock()

        # Statistics
        self._stats = {
            "total_extractions": 0,
            "total_entities": 0,
            "entities_by_label": {},
            "pii_detected_count": 0,
            "average_time_ms": 0.0,
        }

    async def _get_nlp(self, model: NERModel):
        """Lazy load spaCy model."""
        if model == NERModel.LLM:
            return None

        async with self._nlp_lock:
            if self._nlp is not None and self._nlp_model == model:
                return self._nlp

            try:
                import spacy

                logger.info(f"Loading spaCy model: {model.value}")

                loop = asyncio.get_event_loop()
                self._nlp = await loop.run_in_executor(None, lambda: spacy.load(model.value))
                self._nlp_model = model

                logger.info(f"Loaded spaCy model: {model.value}")
                return self._nlp

            except OSError as e:
                logger.warning(f"spaCy model {model.value} not found: {e}")
                return None
            except ImportError:
                logger.warning("spaCy not available")
                return None
            except Exception as e:
                logger.warning(f"Failed to load spaCy model: {e}")
                return None

    # =========================================================================
    # Core Extraction Interface
    # =========================================================================

    async def extract_entities(
        self,
        content: str,
        source_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EntityExtractionResult:
        """
        Extract named entities from content.

        Args:
            content: Text content to analyze
            source_id: Optional ID of source document/run
            metadata: Optional metadata to attach

        Returns:
            EntityExtractionResult with extracted entities
        """
        start_time = time.time()

        # Generate source ID if not provided
        if source_id is None:
            source_id = hashlib.md5(content.encode()).hexdigest()[:16]

        # Truncate if too long
        if len(content) > self.entity_config.max_text_length:
            content = content[: self.entity_config.max_text_length]

        # Log start
        run_id = str(uuid.uuid4())
        if self.entity_config.enable_brain_logging and self.brain_logger:
            await self._log_extraction_start(run_id, source_id)

        try:
            # Try models in order
            entities = await self._extract_with_fallback(content)

            # Filter entities
            entities = self._filter_entities(entities)

            # Deduplicate if configured
            if self.entity_config.deduplicate:
                entities = self._deduplicate_entities(entities)

            # Group by label
            entities_by_label = self._group_by_label(entities)

            # Check for PII
            pii_detected = any(e.is_pii for e in entities)

            processing_time = (time.time() - start_time) * 1000

            result = EntityExtractionResult(
                source_id=source_id,
                entities=entities,
                entities_by_label=entities_by_label,
                total_entities=len(entities),
                unique_entities=len(set(e.text.lower() for e in entities)),
                processing_time_ms=processing_time,
                model_used=self._nlp_model.value if self._nlp_model else "llm",
                pii_detected=pii_detected,
                metadata=metadata or {},
            )

            # Update stats
            self._update_stats(result)

            # Store in Brain Layer
            if self.entity_config.store_entities and self.brain_logger:
                await self._store_entities(source_id, entities)

            # Log completion
            if self.entity_config.enable_brain_logging and self.brain_logger:
                await self._log_extraction_complete(run_id, result)

            return result

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")

            if self.entity_config.enable_brain_logging and self.brain_logger:
                await self._log_extraction_error(run_id, str(e))

            return EntityExtractionResult(
                source_id=source_id,
                entities=[],
                entities_by_label={},
                total_entities=0,
                unique_entities=0,
                processing_time_ms=(time.time() - start_time) * 1000,
                model_used="error",
                metadata={"error": str(e), **(metadata or {})},
            )

    async def _extract_with_fallback(
        self,
        content: str,
    ) -> list[ExtractedEntity]:
        """Try extraction models in priority order."""
        models_to_try = [self.entity_config.model] + self.entity_config.fallback_models

        for model in models_to_try:
            if model == NERModel.LLM:
                entities = await self._extract_with_llm(content)
                if entities:
                    return entities
            else:
                nlp = await self._get_nlp(model)
                if nlp:
                    entities = await self._extract_with_spacy(content, nlp)
                    if entities:
                        return entities

        return []

    async def _extract_with_spacy(
        self,
        content: str,
        nlp: Any,
    ) -> list[ExtractedEntity]:
        """Extract entities using spaCy."""
        try:
            loop = asyncio.get_event_loop()
            doc = await loop.run_in_executor(None, nlp, content)

            entities = []
            for ent in doc.ents:
                try:
                    label = EntityLabel(ent.label_)
                except ValueError:
                    # Unknown label, skip
                    continue

                is_pii = label in self.entity_config.pii_labels

                entities.append(
                    ExtractedEntity(
                        text=ent.text,
                        label=label,
                        start_char=ent.start_char,
                        end_char=ent.end_char,
                        confidence=1.0,  # spaCy doesn't provide confidence
                        is_pii=is_pii,
                        normalized_text=(
                            ent.text.lower().strip() if self.entity_config.normalize_case else None
                        ),
                    )
                )

            return entities

        except Exception as e:
            logger.warning(f"spaCy extraction failed: {e}")
            return []

    async def _extract_with_llm(
        self,
        content: str,
    ) -> list[ExtractedEntity]:
        """Extract entities using LLM."""
        try:
            prompt = f"""Extract named entities from the following text. For each entity, provide:
- The entity text
- The type (PERSON, ORG, GPE, LOC, DATE, TIME, MONEY, PRODUCT, EVENT, TECHNOLOGY)

Text:
{content[:4000]}

Format your response as a list, one entity per line:
ENTITY_TEXT | TYPE

Examples:
John Smith | PERSON
Microsoft | ORG
New York | GPE"""

            response = await self._call_llm(prompt)

            if not response or not response.output:
                return []

            entities = []
            for line in response.output.strip().split("\n"):
                if "|" not in line:
                    continue

                parts = line.split("|")
                if len(parts) != 2:
                    continue

                text = parts[0].strip()
                label_str = parts[1].strip().upper()

                try:
                    label = EntityLabel(label_str)
                except ValueError:
                    continue

                is_pii = label in self.entity_config.pii_labels

                # Find position in original text (approximate)
                start_char = content.find(text)
                end_char = start_char + len(text) if start_char >= 0 else 0

                entities.append(
                    ExtractedEntity(
                        text=text,
                        label=label,
                        start_char=start_char,
                        end_char=end_char,
                        confidence=0.8,  # LLM extractions have lower confidence
                        is_pii=is_pii,
                        normalized_text=(
                            text.lower().strip() if self.entity_config.normalize_case else None
                        ),
                    )
                )

            return entities

        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
            return []

    # =========================================================================
    # Entity Processing
    # =========================================================================

    def _filter_entities(
        self,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedEntity]:
        """Filter entities based on configuration."""
        filtered = []

        for entity in entities:
            # Check length
            if len(entity.text) < self.entity_config.min_entity_length:
                continue
            if len(entity.text) > self.entity_config.max_entity_length:
                continue

            # Check confidence
            if entity.confidence < self.entity_config.min_confidence:
                continue

            # Check label inclusion
            if self.entity_config.include_labels:
                if entity.label not in self.entity_config.include_labels:
                    continue

            # Check label exclusion
            if self.entity_config.exclude_labels:
                if entity.label in self.entity_config.exclude_labels:
                    continue

            filtered.append(entity)

        return filtered

    def _deduplicate_entities(
        self,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedEntity]:
        """Deduplicate entities, keeping highest confidence."""
        seen: dict[tuple[str, EntityLabel], ExtractedEntity] = {}

        for entity in entities:
            key = (entity.text.lower(), entity.label)

            if key not in seen or entity.confidence > seen[key].confidence:
                seen[key] = entity

        return list(seen.values())

    def _group_by_label(
        self,
        entities: list[ExtractedEntity],
    ) -> dict[str, list[str]]:
        """Group entity texts by label."""
        grouped: dict[str, list[str]] = {}

        for entity in entities:
            label = entity.label.value
            if label not in grouped:
                grouped[label] = []
            if entity.text not in grouped[label]:
                grouped[label].append(entity.text)

        return grouped

    # =========================================================================
    # Batch Processing
    # =========================================================================

    async def extract_entities_batch(
        self,
        items: list[tuple[str, str]],  # [(content, source_id), ...]
        max_concurrent: int = 5,
    ) -> list[EntityExtractionResult]:
        """
        Extract entities from multiple documents in parallel.

        Args:
            items: List of (content, source_id) tuples
            max_concurrent: Maximum concurrent extractions

        Returns:
            List of EntityExtractionResult in same order as input
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_one(content: str, source_id: str) -> EntityExtractionResult:
            async with semaphore:
                return await self.extract_entities(content, source_id)

        tasks = [process_one(content, source_id) for content, source_id in items]

        return await asyncio.gather(*tasks)

    # =========================================================================
    # Brain Layer Integration
    # =========================================================================

    async def _store_entities(
        self,
        source_id: str,
        entities: list[ExtractedEntity],
    ) -> None:
        """Store entities in Brain Layer."""
        try:
            import importlib

            try:
                supabase_module = importlib.import_module(
                    "brain_layer.08_supabase_pgvector.supabase_client"
                )
                client = getattr(supabase_module, "get_supabase_client", lambda: None)()

                if client and entities:
                    # Batch insert entities
                    records = []
                    for entity in entities:
                        entity_id = hashlib.md5(
                            f"{source_id}:{entity.text}:{entity.label.value}".encode()
                        ).hexdigest()

                        records.append(
                            {
                                "id": entity_id,
                                "document_id": source_id,
                                "text": entity.text,
                                "label": entity.label.value,
                                "start_char": entity.start_char,
                                "end_char": entity.end_char,
                                "confidence": entity.confidence,
                                "created_at": datetime.utcnow().isoformat(),
                            }
                        )

                    await client.table("document_entities").upsert(records).execute()

            except (ImportError, AttributeError):
                pass

        except Exception as e:
            logger.warning(f"Failed to store entities: {e}")

    async def _log_extraction_start(
        self,
        run_id: str,
        source_id: str,
    ) -> None:
        """Log extraction start."""
        try:
            run_create = AgentRunCreate(
                agent_id=f"entity_{self.agent_id}",
                agent_type="entity_extraction",
                tenant_id=self.context.tenant_id if self.context else None,
                input_summary=f"Extracting entities from {source_id}",
                metadata={"source_id": source_id},
            )
            await self.brain_logger.log_run_start(run_create)
        except Exception as e:
            logger.warning(f"Failed to log extraction start: {e}")

    async def _log_extraction_complete(
        self,
        run_id: str,
        result: EntityExtractionResult,
    ) -> None:
        """Log extraction completion."""
        try:
            run_update = AgentRunUpdate(
                status=RunStatus.COMPLETED,
                output_summary=f"Extracted {result.total_entities} entities",
                metrics={
                    "total_entities": result.total_entities,
                    "unique_entities": result.unique_entities,
                    "processing_time_ms": result.processing_time_ms,
                    "pii_detected": result.pii_detected,
                },
            )
            await self.brain_logger.log_run_update(run_id, run_update)
        except Exception as e:
            logger.warning(f"Failed to log extraction complete: {e}")

    async def _log_extraction_error(
        self,
        run_id: str,
        error: str,
    ) -> None:
        """Log extraction error."""
        try:
            run_update = AgentRunUpdate(
                status=RunStatus.FAILED,
                error_message=error,
            )
            await self.brain_logger.log_run_update(run_id, run_update)
        except Exception as e:
            logger.warning(f"Failed to log extraction error: {e}")

    def _update_stats(self, result: EntityExtractionResult) -> None:
        """Update internal statistics."""
        self._stats["total_extractions"] += 1
        self._stats["total_entities"] += result.total_entities

        if result.pii_detected:
            self._stats["pii_detected_count"] += 1

        # Update label counts
        for label, entities in result.entities_by_label.items():
            if label not in self._stats["entities_by_label"]:
                self._stats["entities_by_label"][label] = 0
            self._stats["entities_by_label"][label] += len(entities)

        # Update average time
        n = self._stats["total_extractions"]
        old_avg = self._stats["average_time_ms"]
        self._stats["average_time_ms"] = old_avg + (result.processing_time_ms - old_avg) / n

    # =========================================================================
    # BaseAgent Interface
    # =========================================================================

    async def _execute_task(self, task: AgentTask) -> AgentResponse:
        """Execute an entity extraction task."""
        content = task.input_data.get("content", "")
        source_id = task.input_data.get("source_id")

        result = await self.extract_entities(
            content=content,
            source_id=source_id,
        )

        return AgentResponse(
            task_id=task.task_id,
            status=TaskStatus.COMPLETED,
            output=result.to_dict(),
            confidence=1.0 if result.entities else 0.5,
            metadata={
                "total_entities": result.total_entities,
                "processing_time_ms": result.processing_time_ms,
            },
        )

    def get_stats(self) -> dict[str, Any]:
        """Get entity extraction statistics."""
        return self._stats.copy()


# =============================================================================
# Factory Function
# =============================================================================


def create_entity_agent(
    config: AgentConfig | None = None,
    entity_config: EntityConfig | None = None,
    context: AgentContext | None = None,
) -> EntityExtractionAgent:
    """
    Factory function to create an EntityExtractionAgent.

    Args:
        config: Base agent configuration
        entity_config: Entity extraction configuration
        context: Agent execution context

    Returns:
        Configured EntityExtractionAgent instance
    """
    return EntityExtractionAgent(
        config=config,
        entity_config=entity_config,
        context=context,
    )
