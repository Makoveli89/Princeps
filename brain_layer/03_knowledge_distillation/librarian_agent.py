"""
Librarian Agent - Knowledge Management System for Mothership
Stores, retrieves, and manages knowledge from all agents
"""

import hashlib
import json
import logging
import sqlite3
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import networkx as nx
import requests  # for URL ingestion
from bs4 import BeautifulSoup  # for parsing HTML

from .concept_graph_agent import ConceptGraphAgent
from .ner_agent import NERAgent
from .summarization_agent import SummarizationAgent
from .topic_modeling_agent import TopicModelingAgent

# Optional imports for enhanced features
_VECTOR_IMPORT_ERROR: Optional[Exception] = None
try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    VECTOR_SUPPORT = True
except Exception as exc:  # pragma: no cover - optional dependency handling
    VECTOR_SUPPORT = False
    _VECTOR_IMPORT_ERROR = exc
    np = None  # type: ignore[assignment]
    SentenceTransformer = None  # type: ignore[assignment]
    faiss = None  # type: ignore[assignment]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if _VECTOR_IMPORT_ERROR:
    logger.warning(
        "Vector search features disabled: %s. "
        "Run 'pip install sentence-transformers faiss-cpu numpy tf-keras' to enable.",
        _VECTOR_IMPORT_ERROR,
    )


class KnowledgeType(Enum):
    """Types of knowledge the librarian can store"""

    AGENT_OUTPUT = "agent_output"
    TASK_RESULT = "task_result"
    DATASET = "dataset"
    CODE_SNIPPET = "code_snippet"
    DOCUMENTATION = "documentation"
    ERROR_SOLUTION = "error_solution"
    BEST_PRACTICE = "best_practice"
    RESEARCH_PAPER = "research_paper"
    CONVERSATION = "conversation"
    EXPERIMENT = "experiment"


class SecurityLevel(Enum):
    """Security levels for knowledge entries"""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class KnowledgeEntry:
    """A single piece of knowledge"""

    id: str
    type: KnowledgeType
    title: str
    content: str
    source: str  # Which agent/system created this
    timestamp: datetime
    tags: List[str]
    metadata: Dict[str, Any]
    version: int = 1
    parent_id: Optional[str] = None  # For versioning
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    category: Optional[str] = None
    checksum: Optional[str] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None


@dataclass
class SearchResult:
    """Search result with relevance score"""

    entry: KnowledgeEntry
    score: float
    snippet: str


class LibrarianAgent:
    """
    The Librarian Agent manages all knowledge across the Mothership system
    """

    def __init__(self, library_dir: str = "./library", enable_vectors: bool = True):
        """
        Initialize the Librarian Agent

        Args:
            library_dir: Directory to store the knowledge library
            enable_vectors: Whether to use vector embeddings for search
        """
        self.library_dir = Path(library_dir)
        self.library_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.datasets_dir = self.library_dir / "datasets"
        self.metadata_dir = self.library_dir / "metadata"
        self.vector_index_dir = self.library_dir / "vector_index"

        for dir_path in [
            self.datasets_dir,
            self.metadata_dir,
            self.vector_index_dir,
        ]:
            dir_path.mkdir(exist_ok=True)

        # Initialize database
        self.db_path = self.metadata_dir / "knowledge.db"
        self._init_database()

        # Initialize NER Agent
        try:
            self.ner_agent = NERAgent()
            logger.info("NER Agent initialized.")
        except Exception as e:
            self.ner_agent = None
            logger.warning(f"Could not initialize NER Agent: {e}")

        # Initialize Topic Modeling Agent
        try:
            self.topic_modeling_agent = TopicModelingAgent()
            logger.info("Topic Modeling Agent initialized.")
        except Exception as e:
            self.topic_modeling_agent = None
            logger.warning(f"Could not initialize Topic Modeling Agent: {e}")

        # Initialize Concept Graph Agent
        try:
            self.concept_graph_agent = ConceptGraphAgent()
            self.concept_graph_path = self.library_dir / "concept_graph.gml"
            self._load_concept_graph()
            logger.info("Concept Graph Agent initialized.")
        except Exception as e:
            self.concept_graph_agent = None
            logger.warning(f"Could not initialize Concept Graph Agent: {e}")

        # Initialize Summarization Agent
        try:
            self.summarization_agent = SummarizationAgent()
            logger.info("Summarization Agent initialized.")
        except Exception as e:
            self.summarization_agent = None
            logger.warning(f"Could not initialize Summarization Agent: {e}")

        # Initialize vector search if available and enabled
        self.vector_enabled = VECTOR_SUPPORT and enable_vectors
        if self.vector_enabled:
            self._init_vector_search()
        else:
            logger.info("Vector search disabled or dependencies not available")
            self.model = None
            self.index = None

    @contextmanager
    def _db_connection(self):
        """Yield a SQLite connection and ensure it closes to avoid file locks."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _init_database(self):
        """Initialize SQLite database for metadata"""
        with self._db_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    tags TEXT,
                    metadata TEXT,
                    version INTEGER DEFAULT 1,
                    parent_id TEXT,
                    security_level TEXT DEFAULT 'internal',
                    category TEXT,
                    checksum TEXT,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    embedding_id INTEGER
                )
            """
            )

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_type ON knowledge(type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON knowledge(source)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON knowledge(category)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON knowledge(timestamp)")

            # New table for entities
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    knowledge_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    label TEXT NOT NULL,
                    start_char INTEGER,
                    end_char INTEGER,
                    FOREIGN KEY (knowledge_id) REFERENCES knowledge (id)
                )
            """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_knowledge_id ON entities(knowledge_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_label ON entities(label)")

            # New table for topics
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS topics (
                    id TEXT PRIMARY KEY,
                    knowledge_id TEXT NOT NULL,
                    topic_id INTEGER NOT NULL,
                    name TEXT,
                    keywords TEXT,
                    FOREIGN KEY (knowledge_id) REFERENCES knowledge (id)
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_topics_knowledge_id ON topics(knowledge_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_topics_topic_id ON topics(topic_id)")

            # New table for concepts
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS concepts (
                    id TEXT PRIMARY KEY,
                    knowledge_id TEXT NOT NULL,
                    concept TEXT NOT NULL,
                    relevance REAL,
                    FOREIGN KEY (knowledge_id) REFERENCES knowledge (id)
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_concepts_knowledge_id ON concepts(knowledge_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_concepts_concept ON concepts(concept)")

            # New table for summaries
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS summaries (
                    id TEXT PRIMARY KEY,
                    knowledge_id TEXT NOT NULL,
                    one_sentence TEXT,
                    executive TEXT,
                    FOREIGN KEY (knowledge_id) REFERENCES knowledge (id)
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_summaries_knowledge_id ON summaries(knowledge_id)")

            conn.commit()

    def _load_concept_graph(self):
        """Load the concept graph from a file."""
        if self.concept_graph_path.exists() and self.concept_graph_agent:
            try:
                self.concept_graph_agent.graph = nx.read_gml(self.concept_graph_path)
                logger.info(
                    f"Loaded concept graph with {self.concept_graph_agent.graph.number_of_nodes()} nodes and {self.concept_graph_agent.graph.number_of_edges()} edges."
                )
            except Exception as e:
                logger.error(f"Could not load concept graph: {e}")

    def _save_concept_graph(self):
        """Save the concept graph to a file."""
        if self.concept_graph_agent:
            try:
                nx.write_gml(self.concept_graph_agent.graph, self.concept_graph_path)
            except Exception as e:
                logger.error(f"Could not save concept graph: {e}")

    def _init_vector_search(self):
        """Initialize vector embedding model and index"""
        try:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")

            # Load or create FAISS index
            index_path = self.vector_index_dir / "knowledge.index"
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))
                logger.info(f"Loaded vector index with {self.index.ntotal} vectors")
            else:
                # Create new index
                dimension = 384  # all-MiniLM-L6-v2 dimension
                # Inner product for cosine similarity
                self.index = faiss.IndexFlatIP(dimension)
                logger.info("Created new vector index")
        except Exception as e:
            logger.error(f"Failed to initialize vector search: {e}")
            self.vector_enabled = False
            self.model = None
            self.index = None

    def store_agent_knowledge(
        self,
        agent_name: str,
        task: str,
        output: Any = None,
        content: Any = None,  # Alias for output
        knowledge_type: KnowledgeType = KnowledgeType.AGENT_OUTPUT,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
        security_level: SecurityLevel = SecurityLevel.INTERNAL,
        category: str = None,
    ) -> str:
        """
        Store knowledge from an agent's task execution

        Args:
            agent_name: Name of the agent storing knowledge
            task: Description of the task performed
            output: The output/result of the task.
            content: Alias for output. For backward compatibility or clearer intent.
            knowledge_type: Type of knowledge being stored
            tags: List of tags for categorization
            metadata: Additional metadata
            security_level: Security classification
            category: Category for organization

        Returns:
            The ID of the stored knowledge entry
        """
        # Use content if output is not provided
        if output is None and content is not None:
            output = content

        # Generate ID
        timestamp = datetime.now()
        content_str = str(output) if not isinstance(output, str) else output
        entry_id = self._generate_id(agent_name, task, timestamp)

        # Calculate checksum
        checksum = hashlib.sha256(content_str.encode()).hexdigest()

        # Create knowledge entry
        entry = KnowledgeEntry(
            id=entry_id,
            type=knowledge_type,
            title=f"{task[:100]}",  # Truncate long tasks
            content=content_str,
            source=agent_name,
            timestamp=timestamp,
            tags=tags or [],
            metadata=metadata or {},
            security_level=security_level,
            category=category,
            checksum=checksum,
        )

        # Check for existing similar entries (versioning)
        existing = self._find_similar_entry(agent_name, task, checksum)
        if existing:
            entry.parent_id = existing["id"]
            entry.version = existing["version"] + 1

        # Store in database
        self._store_entry(entry)

        # Extract and store entities
        if self.ner_agent:
            entities = self.ner_agent.extract_entities(content_str)
            self._store_entities(entry.id, entities)

        # Analyze and store topics
        if self.topic_modeling_agent:
            topics = self.topic_modeling_agent.analyze_topics(content_str)
            self._store_topics(entry.id, topics)

        # Extract, store, and link concepts
        if self.concept_graph_agent:
            concepts = self.concept_graph_agent.extract_concepts(content_str)
            self._store_concepts(entry.id, concepts)
            self.concept_graph_agent.update_graph(entry.id, concepts)
            self._save_concept_graph()

        # Generate and store summaries
        if self.summarization_agent:
            summaries = self.summarization_agent.generate_summaries(content_str)
            self._store_summaries(entry.id, summaries)

        # Update vector index if enabled
        if self.vector_enabled and self.model and self.index:
            self._update_vector_index(entry)

        logger.info(f"Stored knowledge: {entry_id} from {agent_name}")
        return entry_id

    def get_knowledge_by_id(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """
        Retrieve a knowledge entry by its ID.

        Args:
            entry_id: The ID of the entry to retrieve.

        Returns:
            A KnowledgeEntry object or None if not found.
        """
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM knowledge WHERE id=?", (entry_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_entry(row)
        return None

    def _row_to_entry(self, row: tuple) -> KnowledgeEntry:
        """Convert a database row to a KnowledgeEntry object."""
        (
            id_val,
            type_str,
            title,
            content,
            source,
            timestamp_str,
            tags_str,
            metadata_str,
            version,
            parent_id,
            security_level_str,
            category,
            checksum,
            access_count,
            last_accessed_str,
            _,
        ) = row

        return KnowledgeEntry(
            id=id_val,
            type=KnowledgeType(type_str),
            title=title,
            content=content,
            source=source,
            timestamp=datetime.fromisoformat(timestamp_str),
            tags=json.loads(tags_str) if tags_str else [],
            metadata=json.loads(metadata_str) if metadata_str else {},
            version=version,
            parent_id=parent_id,
            security_level=SecurityLevel(security_level_str),
            category=category,
            checksum=checksum,
            access_count=access_count,
            last_accessed=datetime.fromisoformat(last_accessed_str) if last_accessed_str else None,
        )

    def _store_summaries(self, knowledge_id: str, summaries: dict):
        """Store summaries for a knowledge entry."""
        if not summaries:
            return

        with self._db_connection() as conn:
            cursor = conn.cursor()
            summary_id = self._generate_id(knowledge_id, "summary")
            cursor.execute(
                "INSERT INTO summaries (id, knowledge_id, one_sentence, executive) VALUES (?, ?, ?, ?)",
                (
                    summary_id,
                    knowledge_id,
                    summaries.get("one_sentence"),
                    summaries.get("executive"),
                ),
            )
            conn.commit()

    def _store_concepts(self, knowledge_id: str, concepts: list):
        """Store concepts for a knowledge entry."""
        if not concepts:
            return

        with self._db_connection() as conn:
            cursor = conn.cursor()
            for concept_item in concepts:
                concept_id = self._generate_id(knowledge_id, concept_item["concept"])
                cursor.execute(
                    "INSERT INTO concepts (id, knowledge_id, concept, relevance) VALUES (?, ?, ?, ?)",
                    (concept_id, knowledge_id, concept_item["concept"], concept_item["relevance"]),
                )
            conn.commit()

    def _store_topics(self, knowledge_id: str, topics: list):
        """Store topics for a knowledge entry"""
        if not topics:
            return

        with self._db_connection() as conn:
            cursor = conn.cursor()
            for topic in topics:
                topic_entry_id = self._generate_id(knowledge_id, topic.get("name", "topic"))
                cursor.execute(
                    """
                    INSERT INTO topics (id, knowledge_id, topic_id, name, keywords)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        topic_entry_id,
                        knowledge_id,
                        topic.get("topic_id"),
                        topic.get("name"),
                        json.dumps(topic.get("keywords")),
                    ),
                )
            conn.commit()

    def import_dataset(
        self,
        dataset_path: str,
        dataset_name: str,
        source: str,
        description: str = "",
        tags: List[str] = None,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """
        Import a 3rd party dataset into the library

        Args:
            dataset_path: Path to the dataset file
            dataset_name: Name for the dataset
            source: Source of the dataset (e.g., "huggingface", "kaggle")
            description: Description of the dataset
            tags: Tags for categorization
            validate: Whether to validate the dataset

        Returns:
            Import results including ID and statistics
        """
        results = {"success": False, "id": None, "stats": {}, "errors": []}

        try:
            # Validate dataset if requested
            if validate:
                validation = self._validate_dataset(dataset_path)
                if not validation["valid"]:
                    results["errors"] = validation["errors"]
                    return results

            # Copy dataset to library
            dataset_file = Path(dataset_path)
            if not dataset_file.exists():
                results["errors"].append(f"Dataset file not found: {dataset_path}")
                return results

            # Generate unique name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stored_name = f"{dataset_name}_{timestamp}{dataset_file.suffix}"
            stored_path = self.datasets_dir / stored_name

            # Copy or move file
            import shutil

            shutil.copy2(dataset_path, stored_path)

            # Calculate checksum
            checksum = self._calculate_file_checksum(stored_path)

            # Get dataset statistics
            stats = self._analyze_dataset(stored_path)

            # Store metadata
            entry_id = self.store_agent_knowledge(
                agent_name="LibrarianAgent",
                task=f"Import dataset: {dataset_name}",
                output=json.dumps(
                    {
                        "name": dataset_name,
                        "source": source,
                        "description": description,
                        "path": str(stored_path),
                        "original_path": dataset_path,
                        "checksum": checksum,
                        "stats": stats,
                    }
                ),
                knowledge_type=KnowledgeType.DATASET,
                tags=tags or [],
                metadata={
                    "file_size": stored_path.stat().st_size,
                    "file_type": dataset_file.suffix,
                    "import_date": timestamp,
                },
                category="datasets",
            )

            results["success"] = True
            results["id"] = entry_id
            results["stats"] = stats

            logger.info(f"Successfully imported dataset: {dataset_name}")

        except Exception as e:
            results["errors"].append(str(e))
            logger.error(f"Failed to import dataset: {e}")

        return results

    def query_knowledge(
        self,
        query: str,
        knowledge_types: List[KnowledgeType] = None,
        sources: List[str] = None,
        category: str = None,
        tags: List[str] = None,
        limit: int = 10,
        use_vectors: bool = True,
    ) -> List[SearchResult]:
        """
        Query the knowledge library

        Args:
            query: Search query
            knowledge_types: Filter by knowledge types
            sources: Filter by source agents
            category: Filter by category
            tags: Filter by tags
            limit: Maximum number of results
            use_vectors: Whether to use vector search if available

        Returns:
            List of search results with relevance scores
        """
        results = []

        # Use vector search if available and requested
        if use_vectors and self.vector_enabled and self.model and self.index:
            results = self._vector_search(query, knowledge_types, sources, category, tags, limit)

        # Fallback to or combine with keyword search
        if not results or not use_vectors:
            results = self._keyword_search(query, knowledge_types, sources, category, tags, limit)

        # Update access statistics
        for result in results:
            self._update_access_stats(result.entry.id)

        return results

    def _vector_search(
        self,
        query: str,
        knowledge_types: List[KnowledgeType],
        sources: List[str],
        category: str,
        tags: List[str],
        limit: int,
    ) -> List[SearchResult]:
        """Perform vector similarity search"""
        if not self.model or not self.index or self.index.ntotal == 0:
            return []

        # Encode query
        query_vector = self.model.encode([query])
        query_vector = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)

        # Search index
        scores, indices = self.index.search(query_vector, min(limit * 3, self.index.ntotal))

        # Retrieve entries and apply filters
        results = []
        with self._db_connection() as conn:
            cursor = conn.cursor()

            for idx, score in zip(indices[0], scores[0]):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue

                # Get entry by embedding_id
                cursor.execute(
                    "SELECT * FROM knowledge WHERE embedding_id = ?",
                    (int(idx),),
                )

                row = cursor.fetchone()
                if row:
                    entry = self._row_to_entry(row)

                    # Apply filters
                    if knowledge_types and entry.type not in knowledge_types:
                        continue
                    if sources and entry.source not in sources:
                        continue
                    if category and entry.category != category:
                        continue
                    if tags and not any(tag in entry.tags for tag in tags):
                        continue

                    # Create snippet
                    snippet = entry.content[:200] + "..." if len(entry.content) > 200 else entry.content

                    results.append(SearchResult(entry=entry, score=float(score), snippet=snippet))

                    if len(results) >= limit:
                        break

        return results

    def _keyword_search(
        self,
        query: str,
        knowledge_types: List[KnowledgeType],
        sources: List[str],
        category: str,
        tags: List[str],
        limit: int,
    ) -> List[SearchResult]:
        """Perform keyword-based search"""
        with self._db_connection() as conn:
            cursor = conn.cursor()

            # Build query
            sql = "SELECT * FROM knowledge WHERE 1=1"
            params = []

            # Add search condition
            if query:
                sql += " AND (title LIKE ? OR content LIKE ? OR tags LIKE ?)"
                search_term = f"%{query}%"
                params.extend([search_term, search_term, search_term])

            # Add filters
            if knowledge_types:
                type_placeholders = ",".join(["?" for _ in knowledge_types])
                sql += f" AND type IN ({type_placeholders})"
                params.extend([kt.value for kt in knowledge_types])

            if sources:
                source_placeholders = ",".join(["?" for _ in sources])
                sql += f" AND source IN ({source_placeholders})"
                params.extend(sources)

            if category:
                sql += " AND category = ?"
                params.append(category)

            # Order by access count and timestamp
            sql += " ORDER BY access_count DESC, timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(sql, params)
            rows = cursor.fetchall()

            results = []
            for row in rows:
                entry = self._row_to_entry(row)

                # Apply tag filter if specified
                if tags and not any(tag in entry.tags for tag in tags):
                    continue

                # Calculate simple relevance score
                score = 0.0
                if query:
                    query_lower = query.lower()
                    if query_lower in entry.title.lower():
                        score += 2.0
                    if query_lower in entry.content.lower():
                        score += 1.0
                    if any(query_lower in tag.lower() for tag in entry.tags):
                        score += 0.5

                # Create snippet
                snippet = entry.content[:200] + "..." if len(entry.content) > 200 else entry.content

                results.append(SearchResult(entry=entry, score=score, snippet=snippet))

        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    def export_knowledge(
        self,
        output_path: str,
        knowledge_types: List[KnowledgeType] = None,
        sources: List[str] = None,
        export_format: str = "json",
    ) -> bool:
        """
        Export knowledge to a file

        Args:
            output_path: Path for the output file
            knowledge_types: Types to export (None for all)
            sources: Sources to export (None for all)
            export_format: Export format ("json", "jsonl", "zip")

        Returns:
            Success status
        """
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()

                # Build query
                sql = "SELECT * FROM knowledge WHERE 1=1"
                params = []

                if knowledge_types:
                    type_placeholders = ",".join(["?" for _ in knowledge_types])
                    sql += f" AND type IN ({type_placeholders})"
                    params.extend([kt.value for kt in knowledge_types])

                if sources:
                    source_placeholders = ",".join(["?" for _ in sources])
                    sql += f" AND source IN ({source_placeholders})"
                    params.extend(sources)

                cursor.execute(sql, params)
                rows = cursor.fetchall()

                # Convert to entries
                entries = [self._row_to_entry(row) for row in rows]

                # Export based on format
                output = Path(output_path)
                output.parent.mkdir(parents=True, exist_ok=True)

                if export_format == "json":
                    with open(output, "w", encoding="utf-8") as f:
                        json.dump(
                            [self._entry_to_dict(e) for e in entries],
                            f,
                            indent=2,
                            default=str,
                        )
                elif export_format == "jsonl":
                    with open(output, "w", encoding="utf-8") as f:
                        for entry in entries:
                            f.write(json.dumps(self._entry_to_dict(entry), default=str) + "\n")
                elif export_format == "zip":
                    with zipfile.ZipFile(output, "w") as zf:
                        # Add metadata
                        metadata = {
                            "export_date": datetime.now().isoformat(),
                            "entry_count": len(entries),
                            "library_version": "1.0.0",
                        }
                        zf.writestr("metadata.json", json.dumps(metadata, indent=2))

                        # Add entries
                        zf.writestr(
                            "knowledge.json",
                            json.dumps(
                                [self._entry_to_dict(e) for e in entries],
                                indent=2,
                                default=str,
                            ),
                        )

            logger.info("Exported %d entries to %s", len(entries), output_path)
            return True

        except Exception as e:
            logger.error("Export failed: %s", e)
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get library statistics"""
        with self._db_connection() as conn:
            cursor = conn.cursor()

            stats = {}

            # Total entries
            cursor.execute("SELECT COUNT(*) FROM knowledge")
            stats["total_entries"] = cursor.fetchone()[0]

            # By type
            cursor.execute("SELECT type, COUNT(*) FROM knowledge GROUP BY type")
            stats["by_type"] = dict(cursor.fetchall())

            # By source
            cursor.execute("SELECT source, COUNT(*) FROM knowledge GROUP BY source")
            stats["by_source"] = dict(cursor.fetchall())

            # Most accessed
            cursor.execute("SELECT title, access_count FROM knowledge ORDER BY access_count DESC LIMIT 10")
            stats["most_accessed"] = cursor.fetchall()

            # Recent entries
            cursor.execute("SELECT title, timestamp FROM knowledge ORDER BY timestamp DESC LIMIT 10")
            stats["recent_entries"] = cursor.fetchall()

        return stats

    def get_all_entries(self) -> List[KnowledgeEntry]:
        """Retrieve all knowledge entries from the database"""
        entries = []
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM knowledge")
            rows = cursor.fetchall()
            for row in rows:
                entries.append(self._row_to_entry(row))
        return entries

    def delete_knowledge(self, entry_id: str) -> bool:
        """Delete a knowledge entry by ID"""
        deleted = False
        # Remove from database
        with self._db_connection() as conn:
            cursor = conn.cursor()
            for table in ("entities", "topics", "concepts", "summaries"):
                cursor.execute(f"DELETE FROM {table} WHERE knowledge_id = ?", (entry_id,))
            cursor.execute("DELETE FROM knowledge WHERE id = ?", (entry_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
        # Note: vector index removal requires rebuild; disable vector for now
        if deleted and self.vector_enabled:
            logger.info("Entry %s deleted; vector index may need rebuild", entry_id)
        return deleted

    def evaluate_knowledge(self, evaluator: Callable[[KnowledgeEntry], bool]) -> List[str]:
        """
        Evaluate entries with a callback; if evaluator returns False, delete the entry.
        Returns list of deleted entry IDs.
        """
        deleted_ids = []
        for entry in self.get_all_entries():
            if not evaluator(entry):
                if self.delete_knowledge(entry.id):
                    deleted_ids.append(entry.id)
        return deleted_ids

    def ingest_directory(
        self,
        directory: str,
        knowledge_type: KnowledgeType = KnowledgeType.DOCUMENTATION,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
        security_level: SecurityLevel = SecurityLevel.INTERNAL,
        category: Optional[str] = None,
    ) -> None:
        """Bulk ingest text files from a directory as knowledge entries"""
        base_path = Path(directory)
        for file_path in base_path.rglob("*"):
            if file_path.is_file():
                try:
                    content = file_path.read_text(encoding="utf-8")
                    entry_meta = (metadata or {}).copy()
                    entry_meta["source_path"] = str(file_path)
                    self.store_agent_knowledge(
                        agent_name="LibrarianAgent",
                        task=f"Ingest file {file_path.name}",
                        content=content,
                        knowledge_type=knowledge_type,
                        tags=tags or [],
                        metadata=entry_meta,
                        security_level=security_level,
                        category=category or file_path.suffix.lstrip("."),
                    )
                except Exception as e:
                    logger.error("Failed to ingest %s: %s", file_path, e)

    def ingest_url(
        self,
        url: str,
        knowledge_type: KnowledgeType = KnowledgeType.DOCUMENTATION,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
        security_level: SecurityLevel = SecurityLevel.INTERNAL,
        category: Optional[str] = None,
    ) -> str:
        """
        Fetch a webpage and store its main textual content as knowledge.
        Returns the entry ID for the stored content.
        """
        metadata = (metadata or {}).copy()
        metadata["source_url"] = url
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            main_section = soup.find("main") or soup
            text = main_section.get_text(separator="\n").strip()
        except Exception as e:
            logger.error("Failed to ingest URL %s: %s", url, e)
            raise

        entry_id = self.store_agent_knowledge(
            agent_name="LibrarianAgent",
            task=f"Ingest URL: {url}",
            content=text,
            knowledge_type=knowledge_type,
            tags=tags or [],
            metadata=metadata,
            security_level=security_level,
            category=category or "webpage",
        )
        logger.info("Stored URL content as entry %s", entry_id)
        return entry_id

    # Helper methods
    def _generate_id(self, *parts: Any) -> str:
        """Generate a unique ID from a series of parts."""
        content = "_".join(str(p) for p in parts)
        return hashlib.md5(content.encode()).hexdigest()

    def _find_similar_entry(self, source: str, task: str, _checksum: str) -> Optional[Dict]:
        """Find existing similar entry for versioning"""
        with self._db_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT id, version FROM knowledge WHERE source = ? AND title LIKE ? ORDER BY version DESC LIMIT 1",
                (source, f"{task[:50]}%"),
            )

            row = cursor.fetchone()

        if row:
            return {"id": row[0], "version": row[1]}
        return None

    def _store_entry(self, entry: KnowledgeEntry):
        """Store knowledge entry in database"""
        with self._db_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO knowledge (
                    id, type, title, content, source, timestamp,
                    tags, metadata, version, parent_id, security_level,
                    category, checksum, access_count, last_accessed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    entry.id,
                    entry.type.value,
                    entry.title,
                    entry.content,
                    entry.source,
                    entry.timestamp.isoformat(),
                    json.dumps(entry.tags),
                    json.dumps(entry.metadata),
                    entry.version,
                    entry.parent_id,
                    entry.security_level.value,
                    entry.category,
                    entry.checksum,
                    entry.access_count,
                    (entry.last_accessed.isoformat() if entry.last_accessed else None),
                ),
            )

            conn.commit()

    def _store_entities(self, knowledge_id: str, entities: Dict[str, List[str]]):
        """Store extracted entities in the database"""
        if not entities:
            return

        with self._db_connection() as conn:
            cursor = conn.cursor()
            entity_data = []
            for label, texts in entities.items():
                for text in texts:
                    entity_id = self._generate_id(knowledge_id, label, text)
                    entity_data.append((entity_id, knowledge_id, text, label))

            cursor.executemany(
                """
                INSERT INTO entities (id, knowledge_id, text, label)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO NOTHING
                """,
                entity_data,
            )
            conn.commit()
        logger.info(f"Stored {len(entity_data)} entities for knowledge entry {knowledge_id}")

    def _update_vector_index(self, entry: KnowledgeEntry):
        """Update vector index with new entry"""
        if not self.model or not self.index:
            return

        # Create embedding
        text = f"{entry.title} {entry.content}"
        embedding = self.model.encode([text])
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

        # Add to index
        embedding_id = self.index.ntotal
        self.index.add(embedding)

        # Update database with embedding_id
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE knowledge SET embedding_id = ? WHERE id = ?",
                (embedding_id, entry.id),
            )
            conn.commit()

        # Save index
        faiss.write_index(self.index, str(self.vector_index_dir / "knowledge.index"))

    def _generate_id(self, *parts: Any) -> str:
        """Generate a unique ID from a series of parts."""
        content = "_".join(str(p) for p in parts)
        return hashlib.md5(content.encode()).hexdigest()

    def rebuild_vector_index(self) -> int:
        """Rebuild the entire vector index from knowledge entries.

        Returns:
            int: Number of embeddings (entries) indexed.

        Notes:
            - Requires vector dependencies. If unavailable, logs and returns 0.
            - Recomputes embeddings for all entries and resets embedding_id mapping.
        """
        if not VECTOR_SUPPORT:
            logger.warning("Vector dependencies unavailable; cannot rebuild vector index")
            return 0

        # Ensure model and index are initialized
        if not getattr(self, "model", None) or not getattr(self, "index", None):
            self._init_vector_search()
            if not self.model or not self.index:
                logger.warning("Vector search not initialized; rebuild aborted")
                return 0

        try:
            # Reset index and clear embedding_id mappings
            dimension = 384  # all-MiniLM-L6-v2
            import faiss  # local import to satisfy type checkers

            self.index = faiss.IndexFlatIP(dimension)

            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE knowledge SET embedding_id = NULL")
                conn.commit()

            # Load all entries and (re)embed
            count = 0
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM knowledge ORDER BY timestamp ASC")
                rows = cursor.fetchall()

                for row in rows:
                    entry = self._row_to_entry(row)
                    text = f"{entry.title} {entry.content}"
                    embedding = self.model.encode([text])
                    # Normalize for cosine similarity via inner product
                    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
                    embedding_id = self.index.ntotal
                    self.index.add(embedding)

                    cursor.execute(
                        "UPDATE knowledge SET embedding_id = ? WHERE id = ?",
                        (embedding_id, entry.id),
                    )
                    count += 1
                conn.commit()

            # Persist index
            faiss.write_index(self.index, str(self.vector_index_dir / "knowledge.index"))
            logger.info("Rebuilt vector index with %d vectors", count)
            return count
        except Exception as e:
            logger.error("Failed to rebuild vector index: %s", e)
            return 0

    def rebuild_concept_graph(self) -> int:
        """Rebuild the concept graph from stored concept rows."""
        if not self.concept_graph_agent:
            logger.warning("Concept graph agent unavailable; cannot rebuild concept graph")
            return 0

        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT knowledge_id, concept, relevance
                FROM concepts
                ORDER BY knowledge_id
                """
            )
            rows = cursor.fetchall()

        if not rows:
            logger.info("No stored concepts found; concept graph cleared")
            self.concept_graph_agent.graph.clear()
            self._save_concept_graph()
            return 0

        self.concept_graph_agent.graph.clear()
        doc_count = 0
        current_id: Optional[str] = None
        doc_concepts: List[Dict[str, Any]] = []

        for knowledge_id, concept, relevance in rows:
            if knowledge_id != current_id:
                if current_id is not None and doc_concepts:
                    self.concept_graph_agent.update_graph(current_id, doc_concepts)
                    doc_count += 1
                current_id = knowledge_id
                doc_concepts = []

            doc_concepts.append({"concept": concept, "relevance": relevance})

        if current_id is not None and doc_concepts:
            self.concept_graph_agent.update_graph(current_id, doc_concepts)
            doc_count += 1

        self._save_concept_graph()
        logger.info(
            "Rebuilt concept graph with %s documents and %s nodes",
            doc_count,
            self.concept_graph_agent.graph.number_of_nodes(),
        )
        return doc_count

    def vector_status(self) -> Dict[str, Any]:
        """Return vector search status and basic metrics."""
        return {
            "enabled": bool(getattr(self, "vector_enabled", False)),
            "model": (getattr(self, "model", None).__class__.__name__ if getattr(self, "model", None) else None),
            "index_size": (int(getattr(self, "index", None).ntotal) if getattr(self, "index", None) else 0),
            "index_path": str(self.vector_index_dir / "knowledge.index"),
        }

    def get_source_file_index(self) -> Dict[str, str]:
        """Return mapping of source_file metadata to knowledge IDs."""
        index: Dict[str, str] = {}
        with self._db_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
                    SELECT id, json_extract(metadata, '$.source_file') AS source_file
                    FROM knowledge
                    WHERE json_extract(metadata, '$.source_file') IS NOT NULL
                    """
                )
                rows = cursor.fetchall()
            except sqlite3.OperationalError:
                cursor.execute(
                    """
                    SELECT id, metadata
                    FROM knowledge
                    WHERE metadata LIKE '%source_file%'
                    """
                )
                rows = cursor.fetchall()
                parsed_rows = []
                for entry_id, metadata in rows:
                    try:
                        data = json.loads(metadata or "{}")
                    except json.JSONDecodeError:
                        continue
                    source_path = data.get("source_file")
                    if source_path:
                        parsed_rows.append((entry_id, source_path))
                for entry_id, source_path in parsed_rows:
                    index[source_path] = entry_id
                return index

        for entry_id, source_path in rows:
            if source_path:
                index[source_path] = entry_id
        return index

    def has_entry_for_source_file(self, source_file: str) -> bool:
        """Check if an entry already exists for a given source file."""
        with self._db_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
                    SELECT 1
                    FROM knowledge
                    WHERE json_extract(metadata, '$.source_file') = ?
                    LIMIT 1
                    """,
                    (source_file,),
                )
                return cursor.fetchone() is not None
            except sqlite3.OperationalError:
                cursor.execute(
                    """
                    SELECT metadata
                    FROM knowledge
                    WHERE metadata LIKE '%source_file%'
                    """
                )
                rows = cursor.fetchall()

        for (metadata,) in rows:
            try:
                data = json.loads(metadata or "{}")
            except json.JSONDecodeError:
                continue
            if data.get("source_file") == source_file:
                return True
        return False

    def deduplicate_by_source_file(self) -> Dict[str, Any]:
        """Delete older entries when multiple records share the same source_file metadata."""
        summary = {
            "sources_seen": 0,
            "sources_with_duplicates": 0,
            "deleted_entries": 0,
            "deleted_ids": [],
        }

        with self._db_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
                    SELECT
                        id,
                        timestamp,
                        json_extract(metadata, '$.source_file') AS source_file
                    FROM knowledge
                    WHERE json_extract(metadata, '$.source_file') IS NOT NULL
                    ORDER BY source_file, timestamp
                    """
                )
                rows = cursor.fetchall()
                parsed_rows = [
                    (entry_id, timestamp, source_file) for entry_id, timestamp, source_file in rows if source_file
                ]
            except sqlite3.OperationalError:
                cursor.execute(
                    """
                    SELECT id, timestamp, metadata
                    FROM knowledge
                    WHERE metadata LIKE '%source_file%'
                    """
                )
                rows = cursor.fetchall()
                parsed_rows = []
                for entry_id, timestamp, metadata in rows:
                    try:
                        data = json.loads(metadata or "{}")
                    except json.JSONDecodeError:
                        continue
                    source_file = data.get("source_file")
                    if source_file:
                        parsed_rows.append((entry_id, timestamp, source_file))

        grouped: Dict[str, List[tuple]] = defaultdict(list)
        for entry_id, timestamp, source_file in parsed_rows:
            try:
                ts = datetime.fromisoformat(timestamp)
            except ValueError:
                ts = datetime.min
            grouped[source_file].append((entry_id, ts))

        summary["sources_seen"] = len(grouped)

        for source_file, entries in grouped.items():
            if len(entries) <= 1:
                continue
            summary["sources_with_duplicates"] += 1
            entries.sort(key=lambda item: item[1])
            for entry_id, _ in entries[:-1]:
                if self.delete_knowledge(entry_id):
                    summary["deleted_ids"].append(entry_id)
                    summary["deleted_entries"] += 1

        if summary["deleted_entries"] > 0:
            if self.vector_enabled:
                self.rebuild_vector_index()
            self.rebuild_concept_graph()

        return summary

    def _validate_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Validate dataset for security and format"""
        validation = {"valid": True, "errors": [], "warnings": []}

        try:
            path = Path(dataset_path)

            # Check file exists
            if not path.exists():
                validation["valid"] = False
                validation["errors"].append("File does not exist")
                return validation

            # Check file size (max 1GB)
            max_size = 1024 * 1024 * 1024  # 1GB
            if path.stat().st_size > max_size:
                validation["valid"] = False
                validation["errors"].append(f"File too large (max {max_size} bytes)")

            # Check file extension
            allowed_extensions = [
                ".json",
                ".jsonl",
                ".csv",
                ".txt",
                ".tsv",
                ".parquet",
            ]
            if path.suffix.lower() not in allowed_extensions:
                validation["warnings"].append(f"Unusual file extension: {path.suffix}")

            # Basic content validation
            if path.suffix.lower() == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    try:
                        json.load(f)
                    except json.JSONDecodeError as e:
                        validation["valid"] = False
                        validation["errors"].append(f"Invalid JSON: {e}")

            # Security checks
            if path.suffix.lower() in [".exe", ".dll", ".so", ".sh", ".bat"]:
                validation["valid"] = False
                validation["errors"].append("Executable files not allowed")

        except Exception as e:
            validation["valid"] = False
            validation["errors"].append(str(e))

        return validation

    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _analyze_dataset(self, dataset_path: Path) -> Dict[str, Any]:
        """Analyze dataset and return statistics"""
        stats = {
            "file_size": dataset_path.stat().st_size,
            "file_type": dataset_path.suffix,
            "line_count": 0,
            "sample": None,
        }

        try:
            if dataset_path.suffix.lower() in [".json", ".jsonl"]:
                with open(dataset_path, "r", encoding="utf-8") as f:
                    if dataset_path.suffix.lower() == ".json":
                        data = json.load(f)
                        if isinstance(data, list):
                            stats["line_count"] = len(data)
                            stats["sample"] = data[0] if data else None
                    else:  # jsonl
                        lines = f.readlines()
                        stats["line_count"] = len(lines)
                        if lines:
                            stats["sample"] = json.loads(lines[0])

            elif dataset_path.suffix.lower() in [".csv", ".tsv"]:
                import csv

                with open(dataset_path, "r", encoding="utf-8") as f:
                    delimiter = "\t" if dataset_path.suffix.lower() == ".tsv" else ","
                    reader = csv.reader(f, delimiter=delimiter)
                    rows = list(reader)
                    stats["line_count"] = len(rows)
                    if len(rows) > 1:
                        stats["columns"] = rows[0]
                        stats["sample"] = rows[1] if len(rows) > 1 else None

        except Exception as e:
            stats["error"] = str(e)

        return stats

    def _update_access_stats(self, entry_id: str):
        """Update access statistics for an entry"""
        with self._db_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "UPDATE knowledge SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                (datetime.now().isoformat(), entry_id),
            )

            conn.commit()

    def _entry_to_dict(self, entry: KnowledgeEntry) -> Dict[str, Any]:
        """Convert KnowledgeEntry to dictionary"""
        d = asdict(entry)
        d["type"] = entry.type.value
        d["security_level"] = entry.security_level.value
        return d
