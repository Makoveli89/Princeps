"""
ConceptAgent - Concept Graph Building for Knowledge Distillation

This module implements the ConceptAgent that builds and updates concept graphs:
- Key concept extraction using KeyBERT
- Concept relationship mapping (co-occurrence, semantic similarity)
- Graph structure with nodes and edges
- Integration with Brain Layer's concept_nodes and concept_relations tables
- Incremental graph updates

The ConceptAgent identifies key concepts and how they relate, building a
knowledge graph that enables understanding of relationships between ideas.

Strategic Intent:
ConceptAgent enriches the knowledge graph by identifying concepts and their
relationships. For example, linking "PostgreSQL" with "pgvector extension"
when they appear together. This enables semantic navigation and relationship-
aware retrieval.

Adapted from patterns in:
- brain_layer/03_knowledge_distillation/concept_graph_agent.py (KeyBERT + NetworkX)
- brain_layer/04_data_models/models.py (ConceptNode, ConceptRelation)
- brain_layer/04_data_models/knowledge_network.py (graph dataclasses)
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


class ConceptExtractionModel(Enum):
    """Supported concept extraction models."""

    KEYBERT = "keybert"
    YAKE = "yake"
    RAKE = "rake"
    LLM = "llm"
    TFIDF = "tfidf"


class RelationType(Enum):
    """Types of relationships between concepts."""

    CO_OCCURRENCE = "co_occurrence"  # Appear together in same document
    SEMANTIC_SIMILARITY = "semantic"  # Semantically related
    HIERARCHICAL = "hierarchical"  # Parent-child relationship
    CAUSAL = "causal"  # Cause-effect relationship
    PART_OF = "part_of"  # Component relationship
    RELATED_TO = "related_to"  # General relationship


class NodeType(Enum):
    """Types of nodes in the concept graph."""

    CONCEPT = "concept"
    DOCUMENT = "document"
    ENTITY = "entity"
    TOPIC = "topic"


@dataclass
class ConceptConfig:
    """Configuration for ConceptAgent."""

    # Extraction settings
    model: ConceptExtractionModel = ConceptExtractionModel.KEYBERT
    fallback_models: list[ConceptExtractionModel] = field(
        default_factory=lambda: [
            ConceptExtractionModel.TFIDF,
            ConceptExtractionModel.LLM,
        ]
    )

    # KeyBERT settings
    keyphrase_ngram_range: tuple[int, int] = (1, 3)
    use_mmr: bool = True
    diversity: float = 0.7
    top_n_concepts: int = 15
    min_relevance: float = 0.3

    # Graph settings
    build_relationships: bool = True
    min_edge_weight: float = 0.1
    max_edges_per_node: int = 10

    # Relationship detection
    detect_semantic_relations: bool = True
    semantic_threshold: float = 0.5

    # Processing
    min_content_length: int = 100
    max_content_length: int = 50000

    # Brain Layer integration
    enable_brain_logging: bool = True
    store_concepts: bool = True
    update_global_graph: bool = True


@dataclass
class ConceptNode:
    """A node in the concept graph."""

    id: str
    name: str
    node_type: NodeType = NodeType.CONCEPT
    relevance: float = 1.0
    frequency: int = 1
    documents: list[str] = field(default_factory=list)  # Document IDs containing this concept
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "node_type": self.node_type.value,
            "relevance": self.relevance,
            "frequency": self.frequency,
            "documents": self.documents,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ConceptRelation:
    """A relationship between concepts."""

    source_id: str
    target_id: str
    source_name: str
    target_name: str
    relation_type: RelationType = RelationType.CO_OCCURRENCE
    weight: float = 1.0
    documents: list[str] = field(default_factory=list)  # Documents where relation appears
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "source_name": self.source_name,
            "target_name": self.target_name,
            "relation_type": self.relation_type.value,
            "weight": self.weight,
            "documents": self.documents,
            "metadata": self.metadata,
        }


@dataclass
class ConceptGraphResult:
    """Result from concept extraction and graph building."""

    source_id: str
    nodes: list[ConceptNode]
    relations: list[ConceptRelation]
    total_concepts: int
    total_relations: int
    processing_time_ms: float = 0.0
    model_used: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source_id": self.source_id,
            "nodes": [n.to_dict() for n in self.nodes],
            "relations": [r.to_dict() for r in self.relations],
            "total_concepts": self.total_concepts,
            "total_relations": self.total_relations,
            "processing_time_ms": self.processing_time_ms,
            "model_used": self.model_used,
            "metadata": self.metadata,
        }


# =============================================================================
# In-Memory Graph Store
# =============================================================================


class ConceptGraph:
    """
    In-memory concept graph using NetworkX.

    Maintains a global graph that gets updated as documents are processed.
    """

    def __init__(self):
        self._graph = None
        self._lock = asyncio.Lock()

    async def _ensure_graph(self):
        """Lazy initialize NetworkX graph."""
        if self._graph is not None:
            return

        async with self._lock:
            if self._graph is not None:
                return

            try:
                import networkx as nx

                self._graph = nx.DiGraph()
            except ImportError:
                logger.warning("NetworkX not available, graph features disabled")
                self._graph = None

    async def add_node(
        self,
        node: ConceptNode,
    ) -> None:
        """Add or update a node in the graph."""
        await self._ensure_graph()
        if self._graph is None:
            return

        async with self._lock:
            if self._graph.has_node(node.id):
                # Update existing node
                existing = self._graph.nodes[node.id]
                existing["frequency"] = existing.get("frequency", 0) + 1
                docs = existing.get("documents", [])
                for doc_id in node.documents:
                    if doc_id not in docs:
                        docs.append(doc_id)
                existing["documents"] = docs
            else:
                # Add new node
                self._graph.add_node(
                    node.id,
                    name=node.name,
                    node_type=node.node_type.value,
                    relevance=node.relevance,
                    frequency=node.frequency,
                    documents=node.documents,
                    created_at=node.created_at.isoformat(),
                )

    async def add_relation(
        self,
        relation: ConceptRelation,
    ) -> None:
        """Add or update a relation in the graph."""
        await self._ensure_graph()
        if self._graph is None:
            return

        async with self._lock:
            if self._graph.has_edge(relation.source_id, relation.target_id):
                # Update weight
                self._graph[relation.source_id][relation.target_id]["weight"] += relation.weight
            else:
                # Add new edge
                self._graph.add_edge(
                    relation.source_id,
                    relation.target_id,
                    relation_type=relation.relation_type.value,
                    weight=relation.weight,
                    documents=relation.documents,
                )

    async def get_related_concepts(
        self,
        concept_id: str,
        max_depth: int = 2,
        max_results: int = 10,
    ) -> list[tuple[str, str, float]]:
        """Get concepts related to a given concept."""
        await self._ensure_graph()
        if self._graph is None or not self._graph.has_node(concept_id):
            return []

        results = []
        visited = {concept_id}

        async with self._lock:
            # BFS to find related concepts
            queue = [(concept_id, 0)]

            while queue and len(results) < max_results:
                current_id, depth = queue.pop(0)

                if depth >= max_depth:
                    continue

                # Get neighbors
                for neighbor in self._graph.neighbors(current_id):
                    if neighbor in visited:
                        continue

                    visited.add(neighbor)
                    weight = self._graph[current_id][neighbor].get("weight", 1.0)
                    name = self._graph.nodes[neighbor].get("name", neighbor)

                    results.append((neighbor, name, weight))
                    queue.append((neighbor, depth + 1))

        # Sort by weight
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:max_results]

    async def get_stats(self) -> dict[str, Any]:
        """Get graph statistics."""
        await self._ensure_graph()
        if self._graph is None:
            return {"available": False}

        async with self._lock:
            return {
                "available": True,
                "node_count": self._graph.number_of_nodes(),
                "edge_count": self._graph.number_of_edges(),
            }


# Global concept graph
_global_graph = ConceptGraph()


def get_concept_graph() -> ConceptGraph:
    """Get the global concept graph."""
    return _global_graph


# =============================================================================
# ConceptAgent Implementation
# =============================================================================


class ConceptAgent(BaseAgent):
    """
    Knowledge Distillation Agent for concept graph building.

    The ConceptAgent extracts key concepts and their relationships:
    - Uses KeyBERT for concept extraction
    - Builds co-occurrence and semantic relationships
    - Updates global concept graph
    - Stores concepts in Brain Layer

    Usage:
        concept_agent = ConceptAgent(config=config)
        result = await concept_agent.extract_concepts(
            content="PostgreSQL with pgvector enables vector search...",
            source_id="doc_123",
        )
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        concept_config: ConceptConfig | None = None,
        context: AgentContext | None = None,
        brain_logger: BrainLogger | None = None,
        graph: ConceptGraph | None = None,
    ):
        super().__init__(config=config, context=context)

        self.concept_config = concept_config or ConceptConfig()
        self.brain_logger = brain_logger or get_brain_logger()
        self.graph = graph or get_concept_graph()

        # Lazy-loaded models
        self._keybert = None
        self._embedding_model = None
        self._model_lock = asyncio.Lock()

        # Statistics
        self._stats = {
            "total_extractions": 0,
            "concepts_extracted": 0,
            "relations_created": 0,
            "average_time_ms": 0.0,
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

                logger.info("Loading KeyBERT model for concept extraction")

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

    async def _get_embedding_model(self):
        """Lazy load embedding model for semantic similarity."""
        if self._embedding_model is not None:
            return self._embedding_model

        async with self._model_lock:
            if self._embedding_model is not None:
                return self._embedding_model

            try:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading embedding model for semantic relations")

                loop = asyncio.get_event_loop()
                self._embedding_model = await loop.run_in_executor(
                    None, lambda: SentenceTransformer("all-MiniLM-L6-v2")
                )

                return self._embedding_model

            except ImportError:
                logger.warning("sentence-transformers not available")
                return None
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                return None

    # =========================================================================
    # Core Concept Extraction Interface
    # =========================================================================

    async def extract_concepts(
        self,
        content: str,
        source_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ConceptGraphResult:
        """
        Extract concepts and build relationships from content.

        Args:
            content: Text content to analyze
            source_id: Optional ID of source document/run
            metadata: Optional metadata to attach

        Returns:
            ConceptGraphResult with nodes and relations
        """
        start_time = time.time()

        # Generate source ID if not provided
        if source_id is None:
            source_id = hashlib.md5(content.encode()).hexdigest()[:16]

        # Check content length
        if len(content.strip()) < self.concept_config.min_content_length:
            return ConceptGraphResult(
                source_id=source_id,
                nodes=[],
                relations=[],
                total_concepts=0,
                total_relations=0,
                processing_time_ms=(time.time() - start_time) * 1000,
                model_used="none",
                metadata={"reason": "content_too_short"},
            )

        # Truncate if too long
        if len(content) > self.concept_config.max_content_length:
            content = content[: self.concept_config.max_content_length]

        # Log start
        run_id = str(uuid.uuid4())
        if self.concept_config.enable_brain_logging and self.brain_logger:
            await self._log_extraction_start(run_id, source_id)

        try:
            # Extract concepts
            concepts, model_used = await self._extract_with_fallback(content, source_id)

            # Build nodes
            nodes = [
                ConceptNode(
                    id=hashlib.md5(c["concept"].lower().encode()).hexdigest()[:16],
                    name=c["concept"],
                    relevance=c["relevance"],
                    documents=[source_id],
                )
                for c in concepts
            ]

            # Build relations
            relations = []
            if self.concept_config.build_relationships and len(nodes) > 1:
                relations = await self._build_relations(nodes, source_id)

            # Update global graph
            if self.concept_config.update_global_graph:
                for node in nodes:
                    await self.graph.add_node(node)
                for relation in relations:
                    await self.graph.add_relation(relation)

            processing_time = (time.time() - start_time) * 1000

            result = ConceptGraphResult(
                source_id=source_id,
                nodes=nodes,
                relations=relations,
                total_concepts=len(nodes),
                total_relations=len(relations),
                processing_time_ms=processing_time,
                model_used=model_used,
                metadata=metadata or {},
            )

            # Update stats
            self._update_stats(result)

            # Store in Brain Layer
            if self.concept_config.store_concepts and self.brain_logger:
                await self._store_concepts(source_id, nodes, relations)

            # Log completion
            if self.concept_config.enable_brain_logging and self.brain_logger:
                await self._log_extraction_complete(run_id, result)

            return result

        except Exception as e:
            logger.error(f"Concept extraction failed: {e}")

            if self.concept_config.enable_brain_logging and self.brain_logger:
                await self._log_extraction_error(run_id, str(e))

            return ConceptGraphResult(
                source_id=source_id,
                nodes=[],
                relations=[],
                total_concepts=0,
                total_relations=0,
                processing_time_ms=(time.time() - start_time) * 1000,
                model_used="error",
                metadata={"error": str(e), **(metadata or {})},
            )

    async def _extract_with_fallback(
        self,
        content: str,
        source_id: str,
    ) -> tuple[list[dict[str, Any]], str]:
        """Try extraction models in priority order."""
        models_to_try = [self.concept_config.model] + self.concept_config.fallback_models

        for model in models_to_try:
            try:
                if model == ConceptExtractionModel.KEYBERT:
                    result = await self._extract_with_keybert(content)
                    if result:
                        return result, "keybert"

                elif model == ConceptExtractionModel.LLM:
                    result = await self._extract_with_llm(content)
                    if result:
                        return result, "llm"

                elif model == ConceptExtractionModel.TFIDF:
                    result = await self._extract_with_tfidf(content)
                    if result:
                        return result, "tfidf"

            except Exception as e:
                logger.warning(f"Concept extraction with {model.value} failed: {e}")
                continue

        return [], "none"

    async def _extract_with_keybert(
        self,
        content: str,
    ) -> list[dict[str, Any]]:
        """Extract concepts using KeyBERT."""
        keybert = await self._get_keybert()
        if keybert is None:
            return []

        try:
            loop = asyncio.get_event_loop()
            keywords = await loop.run_in_executor(
                None,
                lambda: keybert.extract_keywords(
                    content,
                    keyphrase_ngram_range=self.concept_config.keyphrase_ngram_range,
                    stop_words="english",
                    use_mmr=self.concept_config.use_mmr,
                    diversity=self.concept_config.diversity,
                    top_n=self.concept_config.top_n_concepts,
                ),
            )

            concepts = [
                {"concept": kw, "relevance": score}
                for kw, score in keywords
                if score >= self.concept_config.min_relevance
            ]

            return concepts

        except Exception as e:
            logger.warning(f"KeyBERT extraction failed: {e}")
            return []

    async def _extract_with_llm(
        self,
        content: str,
    ) -> list[dict[str, Any]]:
        """Extract concepts using LLM."""
        try:
            prompt = f"""Extract the {self.concept_config.top_n_concepts} most important concepts, terms, or key phrases from the following text.

Text:
{content[:4000]}

For each concept, rate its importance from 0.0 to 1.0.

Format your response as:
CONCEPT | SCORE

Examples:
machine learning | 0.9
neural networks | 0.85
data preprocessing | 0.7"""

            response = await self._call_llm(prompt)

            if not response or not response.output:
                return []

            concepts = []
            for line in response.output.strip().split("\n"):
                if "|" not in line:
                    continue

                parts = line.split("|")
                if len(parts) != 2:
                    continue

                concept = parts[0].strip()
                try:
                    score = float(parts[1].strip())
                except ValueError:
                    score = 0.5

                if score >= self.concept_config.min_relevance:
                    concepts.append({"concept": concept, "relevance": score})

            return concepts[: self.concept_config.top_n_concepts]

        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
            return []

    async def _extract_with_tfidf(
        self,
        content: str,
    ) -> list[dict[str, Any]]:
        """Extract concepts using TF-IDF."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=self.concept_config.keyphrase_ngram_range,
                max_features=self.concept_config.top_n_concepts * 2,
            )

            loop = asyncio.get_event_loop()
            tfidf_matrix = await loop.run_in_executor(
                None, lambda: vectorizer.fit_transform([content])
            )

            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]

            # Get top concepts
            concept_scores = list(zip(feature_names, scores))
            concept_scores.sort(key=lambda x: x[1], reverse=True)

            concepts = [
                {"concept": concept, "relevance": float(score)}
                for concept, score in concept_scores[: self.concept_config.top_n_concepts]
                if score >= self.concept_config.min_relevance
            ]

            return concepts

        except Exception as e:
            logger.warning(f"TF-IDF extraction failed: {e}")
            return []

    # =========================================================================
    # Relationship Building
    # =========================================================================

    async def _build_relations(
        self,
        nodes: list[ConceptNode],
        source_id: str,
    ) -> list[ConceptRelation]:
        """Build relationships between concepts."""
        relations = []

        # Co-occurrence relations (all concepts in same doc are related)
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i + 1 :]:
                # Weight based on combined relevance
                weight = (node1.relevance + node2.relevance) / 2

                if weight >= self.concept_config.min_edge_weight:
                    relations.append(
                        ConceptRelation(
                            source_id=node1.id,
                            target_id=node2.id,
                            source_name=node1.name,
                            target_name=node2.name,
                            relation_type=RelationType.CO_OCCURRENCE,
                            weight=weight,
                            documents=[source_id],
                        )
                    )

                if len(relations) >= self.concept_config.max_edges_per_node * len(nodes):
                    break

        # Semantic relations (if enabled and embedding model available)
        if self.concept_config.detect_semantic_relations:
            semantic_relations = await self._build_semantic_relations(nodes, source_id)
            relations.extend(semantic_relations)

        return relations

    async def _build_semantic_relations(
        self,
        nodes: list[ConceptNode],
        source_id: str,
    ) -> list[ConceptRelation]:
        """Build semantic similarity-based relations."""
        embedding_model = await self._get_embedding_model()
        if embedding_model is None:
            return []

        try:
            import numpy as np

            loop = asyncio.get_event_loop()

            # Get embeddings for all concept names
            concept_names = [node.name for node in nodes]
            embeddings = await loop.run_in_executor(
                None, lambda: embedding_model.encode(concept_names)
            )

            relations = []

            # Calculate pairwise cosine similarities
            for i, node1 in enumerate(nodes):
                for j, node2 in enumerate(nodes):
                    if i >= j:
                        continue

                    # Cosine similarity
                    sim = float(
                        np.dot(embeddings[i], embeddings[j])
                        / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                    )

                    if sim >= self.concept_config.semantic_threshold:
                        relations.append(
                            ConceptRelation(
                                source_id=node1.id,
                                target_id=node2.id,
                                source_name=node1.name,
                                target_name=node2.name,
                                relation_type=RelationType.SEMANTIC_SIMILARITY,
                                weight=sim,
                                documents=[source_id],
                            )
                        )

            return relations

        except Exception as e:
            logger.warning(f"Semantic relation building failed: {e}")
            return []

    # =========================================================================
    # Graph Queries
    # =========================================================================

    async def get_related_concepts(
        self,
        concept: str,
        max_depth: int = 2,
        max_results: int = 10,
    ) -> list[tuple[str, str, float]]:
        """
        Get concepts related to a given concept name.

        Args:
            concept: Concept name to find relations for
            max_depth: Maximum graph traversal depth
            max_results: Maximum number of results

        Returns:
            List of (concept_id, concept_name, weight) tuples
        """
        concept_id = hashlib.md5(concept.lower().encode()).hexdigest()[:16]
        return await self.graph.get_related_concepts(concept_id, max_depth, max_results)

    async def get_graph_stats(self) -> dict[str, Any]:
        """Get global concept graph statistics."""
        return await self.graph.get_stats()

    # =========================================================================
    # Brain Layer Integration
    # =========================================================================

    async def _store_concepts(
        self,
        source_id: str,
        nodes: list[ConceptNode],
        relations: list[ConceptRelation],
    ) -> None:
        """Store concepts and relations in Brain Layer."""
        try:
            import importlib

            try:
                supabase_module = importlib.import_module(
                    "brain_layer.08_supabase_pgvector.supabase_client"
                )
                client = getattr(supabase_module, "get_supabase_client", lambda: None)()

                if client:
                    # Store document_concepts
                    if nodes:
                        concept_records = []
                        for node in nodes:
                            record_id = hashlib.md5(f"{source_id}:{node.name}".encode()).hexdigest()

                            concept_records.append(
                                {
                                    "id": record_id,
                                    "document_id": source_id,
                                    "concept": node.name,
                                    "relevance": node.relevance,
                                    "created_at": datetime.utcnow().isoformat(),
                                }
                            )

                        await client.table("document_concepts").upsert(concept_records).execute()

                    # Store concept_nodes (global)
                    if nodes:
                        node_records = []
                        for node in nodes:
                            node_records.append(
                                {
                                    "id": node.id,
                                    "name": node.name,
                                    "node_type": node.node_type.value,
                                    "frequency": node.frequency,
                                    "created_at": datetime.utcnow().isoformat(),
                                }
                            )

                        await client.table("concept_nodes").upsert(node_records).execute()

                    # Store concept_relations
                    if relations:
                        relation_records = []
                        for rel in relations:
                            rel_id = hashlib.md5(
                                f"{rel.source_id}:{rel.target_id}:{rel.relation_type.value}".encode()
                            ).hexdigest()

                            relation_records.append(
                                {
                                    "id": rel_id,
                                    "source_id": rel.source_id,
                                    "target_id": rel.target_id,
                                    "relation_type": rel.relation_type.value,
                                    "weight": rel.weight,
                                    "created_at": datetime.utcnow().isoformat(),
                                }
                            )

                        await client.table("concept_relations").upsert(relation_records).execute()

            except (ImportError, AttributeError):
                pass

        except Exception as e:
            logger.warning(f"Failed to store concepts: {e}")

    async def _log_extraction_start(
        self,
        run_id: str,
        source_id: str,
    ) -> None:
        """Log extraction start."""
        try:
            run_create = AgentRunCreate(
                agent_id=f"concept_{self.agent_id}",
                agent_type="concept_extraction",
                tenant_id=self.context.tenant_id if self.context else None,
                input_summary=f"Extracting concepts from {source_id}",
                metadata={"source_id": source_id},
            )
            await self.brain_logger.log_run_start(run_create)
        except Exception as e:
            logger.warning(f"Failed to log extraction start: {e}")

    async def _log_extraction_complete(
        self,
        run_id: str,
        result: ConceptGraphResult,
    ) -> None:
        """Log extraction completion."""
        try:
            run_update = AgentRunUpdate(
                status=RunStatus.COMPLETED,
                output_summary=f"Extracted {result.total_concepts} concepts, {result.total_relations} relations",
                metrics={
                    "concept_count": result.total_concepts,
                    "relation_count": result.total_relations,
                    "processing_time_ms": result.processing_time_ms,
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

    def _update_stats(self, result: ConceptGraphResult) -> None:
        """Update internal statistics."""
        self._stats["total_extractions"] += 1
        self._stats["concepts_extracted"] += result.total_concepts
        self._stats["relations_created"] += result.total_relations

        # Update average time
        n = self._stats["total_extractions"]
        old_avg = self._stats["average_time_ms"]
        self._stats["average_time_ms"] = old_avg + (result.processing_time_ms - old_avg) / n

    # =========================================================================
    # BaseAgent Interface
    # =========================================================================

    async def _execute_task(self, task: AgentTask) -> AgentResponse:
        """Execute a concept extraction task."""
        content = task.input_data.get("content", "")
        source_id = task.input_data.get("source_id")

        result = await self.extract_concepts(
            content=content,
            source_id=source_id,
        )

        return AgentResponse(
            task_id=task.task_id,
            status=TaskStatus.COMPLETED,
            output=result.to_dict(),
            confidence=1.0 if result.nodes else 0.5,
            metadata={
                "concept_count": result.total_concepts,
                "relation_count": result.total_relations,
                "processing_time_ms": result.processing_time_ms,
            },
        )

    def get_stats(self) -> dict[str, Any]:
        """Get concept extraction statistics."""
        return self._stats.copy()


# =============================================================================
# Factory Function
# =============================================================================


def create_concept_agent(
    config: AgentConfig | None = None,
    concept_config: ConceptConfig | None = None,
    context: AgentContext | None = None,
) -> ConceptAgent:
    """
    Factory function to create a ConceptAgent.

    Args:
        config: Base agent configuration
        concept_config: Concept extraction configuration
        context: Agent execution context

    Returns:
        Configured ConceptAgent instance
    """
    return ConceptAgent(
        config=config,
        concept_config=concept_config,
        context=context,
    )
