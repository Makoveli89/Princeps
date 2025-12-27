#!/usr/bin/env python3
# pylint: disable=invalid-name
"""
Memory System Improvements
===========================
Fixes for memory retrieval, storage, and learning persistence issues

Key Improvements:
1. Semantic memory retrieval with similarity scoring
2. Memory deduplication and consolidation
3. Auto-save persistence with error recovery
4. Priority-based knowledge sharing
5. Recency-weighted pattern analysis
"""

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import shutil
import math

# Optional vector store
try:
    from vector_store import ChromaVectorStore
except Exception:
    ChromaVectorStore = None  # type: ignore


class ImprovedMemoryRetrieval:
    """Enhanced memory retrieval with semantic search and relevance scoring"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_schema()

    def _ensure_schema(self):
        """Ensure database has required schema with new fields"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if table exists
        cursor.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='learnings'
        """
        )
        table_exists = cursor.fetchone() is not None

        if not table_exists:
            # Create enhanced learnings table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS learnings (
                    id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    task_hash TEXT,
                    iteration INTEGER,
                    success BOOLEAN,
                    score INTEGER,
                    solution TEXT,
                    feedback TEXT,
                    timestamp TEXT,
                    relevance_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    consolidated BOOLEAN DEFAULT 0,
                    parent_id TEXT,
                    FOREIGN KEY (parent_id) REFERENCES learnings(id)
                )
            """
            )
        else:
            # Migrate existing table by adding new columns if they don't exist
            cursor.execute("PRAGMA table_info(learnings)")
            columns = {row[1] for row in cursor.fetchall()}

            new_columns = {
                "task_hash": "TEXT",
                "relevance_count": "INTEGER DEFAULT 0",
                "last_accessed": "TEXT",
                "consolidated": "BOOLEAN DEFAULT 0",
                "parent_id": "TEXT",
            }

            for col_name, col_type in new_columns.items():
                if col_name not in columns:
                    try:
                        cursor.execute(f"ALTER TABLE learnings ADD COLUMN {col_name} {col_type}")
                        print(f"  ✅ Added column: {col_name}")
                    except sqlite3.OperationalError as e:
                        print(f"  ⚠️  Could not add column {col_name}: {e}")

        # Create indexes (will skip if already exist)
        try:
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_learnings_success_score
                ON learnings(success, score)
            """
            )
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_learnings_task_hash
                ON learnings(task_hash, consolidated)
            """
            )
        except sqlite3.OperationalError:
            pass

        conn.commit()
        conn.close()

    def _calculate_task_similarity(self, task1: str, task2: str) -> float:
        """Calculate simple similarity score between tasks"""
        # Tokenize and normalize
        words1 = set(task1.lower().split())
        words2 = set(task2.lower().split())

        if not words1 or not words2:
            return 0.0

        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def retrieve_relevant_knowledge(
        self, task: str, limit: int = 5, min_score: int = 70, min_similarity: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant past learnings with similarity scoring"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get successful learnings
            cursor.execute(
                """
                SELECT id, task, feedback, score, timestamp, relevance_count
                FROM learnings
                WHERE success = 1 AND score >= ? AND consolidated = 0
                ORDER BY score DESC, timestamp DESC
                LIMIT 20
            """,
                (min_score,),
            )

            candidates = cursor.fetchall()

            # Calculate similarity and rank
            ranked = []
            for row in candidates:
                similarity = self._calculate_task_similarity(task, row[1])
                if similarity >= min_similarity:
                    ranked.append(
                        {
                            "id": row[0],
                            "task": row[1],
                            "feedback": row[2],
                            "score": row[3],
                            "timestamp": row[4],
                            "relevance_count": row[5],
                            "similarity": similarity,
                            "composite_score": (similarity * 0.6) + (row[3] / 100 * 0.4),
                        }
                    )

            # Sort by composite score
            ranked.sort(key=lambda x: x["composite_score"], reverse=True)

            # Update relevance counts for accessed items
            accessed_ids = [item["id"] for item in ranked[:limit]]
            if accessed_ids:
                placeholders = ",".join("?" * len(accessed_ids))
                cursor.execute(
                    f"""
                    UPDATE learnings
                    SET relevance_count = relevance_count + 1,
                        last_accessed = ?
                    WHERE id IN ({placeholders})
                """,
                    [datetime.now().isoformat()] + accessed_ids,
                )
                conn.commit()

            conn.close()

            return ranked[:limit]

        except Exception as e:
            print(f"⚠️  Memory retrieval error: {e}")
            return []

    def consolidate_memories(self, min_age_days: int = 7) -> int:
        """Consolidate old similar memories to reduce bloat"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cutoff_date = (datetime.now() - timedelta(days=min_age_days)).isoformat()

            # Find duplicate tasks (same hash, not consolidated)
            cursor.execute(
                """
                SELECT task_hash, COUNT(*) as count
                FROM learnings
                WHERE consolidated = 0 AND timestamp < ? AND task_hash IS NOT NULL
                GROUP BY task_hash
                HAVING count > 1
            """,
                (cutoff_date,),
            )

            duplicates = cursor.fetchall()
            consolidated_count = 0

            for task_hash, _ in duplicates:
                # Get all instances
                cursor.execute(
                    """
                    SELECT id, score, timestamp
                    FROM learnings
                    WHERE task_hash = ? AND consolidated = 0
                    ORDER BY score DESC, timestamp DESC
                """,
                    (task_hash,),
                )

                instances = cursor.fetchall()

                if len(instances) > 1:
                    # Keep the best one, mark others as consolidated
                    best_id = instances[0][0]
                    other_ids = [inst[0] for inst in instances[1:]]

                    placeholders = ",".join("?" * len(other_ids))
                    cursor.execute(
                        f"""
                        UPDATE learnings
                        SET consolidated = 1, parent_id = ?
                        WHERE id IN ({placeholders})
                    """,
                        [best_id] + other_ids,
                    )

                    consolidated_count += len(other_ids)

            conn.commit()
            conn.close()

            print(f"🧹 Consolidated {consolidated_count} duplicate memories")
            return consolidated_count

        except Exception as e:
            print(f"⚠️  Consolidation error: {e}")
            return 0


class AutoSavePersistence:
    """Automatic persistence with transaction safety and error recovery"""

    def __init__(self, memory_file: str, backup_file: Optional[str] = None):
        self.memory_file = Path(memory_file)
        self.backup_file = Path(backup_file) if backup_file else self.memory_file.with_suffix(".bak")
        self.auto_save_enabled = True
        self.save_interval = 10  # Save every 10 operations
        self.operations_since_save = 0

    def save_with_transaction(self, data: Dict[str, Any]) -> bool:
        """Save data with transaction safety"""
        try:
            # Create backup of existing file
            if self.memory_file.exists():
                shutil.copy2(self.memory_file, self.backup_file)

            # Write to temporary file first
            temp_file = self.memory_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            # Atomic rename
            temp_file.replace(self.memory_file)

            return True

        except Exception as e:
            print(f"⚠️  Save error: {e}")
            return False

    def load_with_recovery(self) -> Optional[Dict[str, Any]]:
        """Load data with automatic error recovery"""
        try:
            # Try main file
            if self.memory_file.exists():
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    return json.load(f)

        except json.JSONDecodeError as e:
            print(f"⚠️  Corrupted memory file, attempting recovery from backup: {e}")

            # Try backup file
            if self.backup_file.exists():
                try:
                    with open(self.backup_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    print("✅ Recovered from backup")
                    return data
                except Exception as e2:
                    print(f"❌ Backup also corrupted: {e2}")

        except Exception as e:
            print(f"⚠️  Load error: {e}")

        return None

    def auto_save_if_needed(self, data: Dict[str, Any]) -> bool:
        """Auto-save if operation threshold reached"""
        if not self.auto_save_enabled:
            return False

        self.operations_since_save += 1

        if self.operations_since_save >= self.save_interval:
            success = self.save_with_transaction(data)
            if success:
                self.operations_since_save = 0
            return success

        return False


class PriorityKnowledgeSharing:
    """Priority-based knowledge sharing system"""

    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self.knowledge_base: List[Dict[str, Any]] = []
        self.priority_scores: Dict[str, float] = {}

    def calculate_priority(self, knowledge: Dict[str, Any]) -> float:
        """Calculate priority score for knowledge"""
        base_score = 0.0

        # Confidence weight
        base_score += knowledge.get("confidence", 0.0) * 50

        # Recency weight (newer is better)
        timestamp = datetime.fromisoformat(knowledge.get("updated_at", datetime.now().isoformat()))
        age_days = (datetime.now() - timestamp).days
        recency_score = max(0, 50 - age_days)
        base_score += recency_score

        # Usage weight (more used = more important)
        usage_count = knowledge.get("usage_count", 0)
        base_score += min(usage_count * 2, 30)

        return base_score

    def add_knowledge(self, knowledge: Dict[str, Any]) -> bool:
        """Add knowledge with priority management"""
        knowledge_id = knowledge.get("id", str(len(self.knowledge_base)))
        priority = self.calculate_priority(knowledge)

        # Add to base
        self.knowledge_base.append(knowledge)
        self.priority_scores[knowledge_id] = priority

        # Trim if over size
        if len(self.knowledge_base) > self.max_size:
            self._trim_low_priority()

        return True

    def _trim_low_priority(self):
        """Remove lowest priority items"""
        # Sort by priority
        sorted_items = sorted(
            self.knowledge_base,
            key=lambda x: self.priority_scores.get(x.get("id", ""), 0),
            reverse=True,
        )

        # Keep top items
        keep_count = int(self.max_size * 0.8)  # Keep 80%
        self.knowledge_base = sorted_items[:keep_count]

        # Update priority scores
        kept_ids = {item.get("id") for item in self.knowledge_base}
        self.priority_scores = {k: v for k, v in self.priority_scores.items() if k in kept_ids}

    def get_top_knowledge(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get highest priority knowledge"""
        sorted_items = sorted(
            self.knowledge_base,
            key=lambda x: self.priority_scores.get(x.get("id", ""), 0),
            reverse=True,
        )
        return sorted_items[:limit]


class RecencyWeightedPatterns:
    """Pattern analysis with recency weighting"""

    def __init__(self, decay_days: int = 30):
        self.decay_days = decay_days

    def calculate_recency_weight(self, timestamp: str) -> float:
        """Calculate exponential decay weight based on age"""
        try:
            event_time = datetime.fromisoformat(timestamp)
            age_days = (datetime.now() - event_time).days

            # Exponential decay: e^(-age/decay_period)
            weight = math.exp(-age_days / self.decay_days)

            return weight

        except Exception:
            return 0.5  # Default weight for invalid timestamps

    def analyze_patterns_with_recency(
        self, patterns: List[Dict[str, Any]], pattern_type: str = "success"
    ) -> List[Dict[str, Any]]:
        """Analyze patterns with recency weighting"""
        weighted_patterns = []

        for pattern in patterns:
            timestamp = pattern.get("timestamp", datetime.now().isoformat())
            recency_weight = self.calculate_recency_weight(timestamp)

            # Calculate weighted score
            base_score = 1.0 if pattern_type == "success" else 0.0
            weighted_score = base_score * recency_weight

            weighted_patterns.append(
                {
                    **pattern,
                    "recency_weight": recency_weight,
                    "weighted_score": weighted_score,
                    "age_days": (datetime.now() - datetime.fromisoformat(timestamp)).days,
                }
            )

        # Sort by weighted score
        weighted_patterns.sort(key=lambda x: x["weighted_score"], reverse=True)

        return weighted_patterns


class UnifiedMemoryInterface:
    """Unified interface for all memory systems"""

    def __init__(self, db_path: str, memory_file: str):
        self.retrieval = ImprovedMemoryRetrieval(db_path)
        self.persistence = AutoSavePersistence(memory_file)
        self.knowledge_sharing = PriorityKnowledgeSharing()
        self.pattern_analyzer = RecencyWeightedPatterns()
        # Initialize vector store (Chroma on disk) if available
        self.vector_store = None
        try:
            if ChromaVectorStore is not None:
                self.vector_store = ChromaVectorStore(persist_dir=".chroma", collection_name="mothership_learnings")
        except Exception:
            self.vector_store = None

    def store_learning(self, task: str, solution: Dict[str, Any], validation: Dict[str, Any]) -> bool:
        """Store learning with all improvements"""
        try:
            conn = sqlite3.connect(self.retrieval.db_path)
            cursor = conn.cursor()

            learning_id = hashlib.md5(f"{task}_{datetime.now().timestamp()}".encode()).hexdigest()

            task_hash = hashlib.md5(task.encode()).hexdigest()

            cursor.execute(
                """
                INSERT INTO learnings (
                    id, task, task_hash, iteration, success, score,
                    solution, feedback, timestamp, last_accessed
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    learning_id,
                    task,
                    task_hash,
                    1,  # iteration
                    validation.get("success", False),
                    validation.get("score", 0),
                    json.dumps(solution, default=str),
                    validation.get("feedback", ""),
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                ),
            )

            conn.commit()
            conn.close()

            # Upsert into vector store if available
            try:
                if self.vector_store and self.vector_store.is_available():
                    meta = {
                        "success": bool(validation.get("success", False)),
                        "score": int(validation.get("score", 0)),
                        "timestamp": datetime.now().isoformat(),
                    }
                    self.vector_store.upsert_learning(
                        learning_id=learning_id,
                        task=task,
                        feedback=validation.get("feedback", ""),
                        solution=solution,
                        metadata=meta,
                    )
            except Exception:
                pass

            return True

        except Exception as e:
            print(f"⚠️  Failed to store learning: {e}")
            return False

    def retrieve_for_task(self, task: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant knowledge for a task"""
        # Try vector store first
        try:
            if self.vector_store and self.vector_store.is_available():
                vres = self.vector_store.query(task, top_k=limit)
                if vres:
                    # Map to expected structure
                    out: List[Dict[str, Any]] = []
                    for r in vres:
                        out.append(
                            {
                                "id": r.id,
                                "task": r.document.split("\n", 1)[0].replace("Task: ", ""),
                                "feedback": r.metadata.get("feedback", ""),
                                "score": int(r.metadata.get("score", 0)),
                                "timestamp": r.metadata.get("timestamp", datetime.now().isoformat()),
                                "relevance_count": 0,
                                "similarity": r.score,
                                "composite_score": r.score,  # already similarity-like
                            }
                        )
                    return out
        except Exception:
            pass
        # Fallback to SQLite retrieval
        return self.retrieval.retrieve_relevant_knowledge(task, limit=limit)

    def consolidate_old_memories(self, min_age_days: int = 7) -> int:
        """Consolidate old memories"""
        return self.retrieval.consolidate_memories(min_age_days)

    def share_knowledge(self, knowledge: Dict[str, Any]) -> bool:
        """Share knowledge across systems"""
        return self.knowledge_sharing.add_knowledge(knowledge)

    def get_best_practices(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top best practices"""
        return self.knowledge_sharing.get_top_knowledge(limit)

    def analyze_patterns(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze patterns with recency weighting"""
        return self.pattern_analyzer.analyze_patterns_with_recency(patterns)


# Demo and testing
def demo_improvements():
    """Demonstrate all improvements"""
    print("\n" + "=" * 80)
    print("🧠 MEMORY SYSTEM IMPROVEMENTS DEMO")
    print("=" * 80)

    # Initialize unified interface
    interface = UnifiedMemoryInterface(db_path="ai_learning_memory.db", memory_file="agent_memory.json")

    # Demo 1: Store learning
    print("\n📝 Storing sample learnings...")
    for i in range(3):
        interface.store_learning(
            task=f"Fix bug in authentication system {i}",
            solution={"code": f"fix_{i}.py", "tests": f"test_{i}.py"},
            validation={"success": True, "score": 85 + i * 5, "feedback": "Good fix"},
        )

    # Demo 2: Retrieve relevant
    print("\n🔍 Retrieving relevant knowledge...")
    results = interface.retrieve_for_task("Fix bug in authentication", limit=3)
    for idx, result in enumerate(results, 1):
        print(f"\n  {idx}. Similarity: {result['similarity']:.2f}, Score: {result['score']}")
        print(f"     Task: {result['task'][:60]}...")

    # Demo 3: Consolidate
    print("\n🧹 Consolidating old memories...")
    count = interface.consolidate_old_memories(min_age_days=0)
    print(f"   Consolidated {count} memories")

    # Demo 4: Knowledge sharing
    print("\n🤝 Sharing knowledge...")
    interface.share_knowledge(
        {
            "id": "k1",
            "type": "best_practice",
            "confidence": 0.9,
            "updated_at": datetime.now().isoformat(),
            "usage_count": 5,
            "content": "Always use parameterized queries",
        }
    )

    best_practices = interface.get_best_practices(limit=5)
    print(f"   Retrieved {len(best_practices)} best practices")

    print("\n" + "=" * 80)
    print("✅ All improvements demonstrated successfully!")
    print("=" * 80)


if __name__ == "__main__":
    demo_improvements()
