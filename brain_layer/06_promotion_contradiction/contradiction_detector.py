"""
Contradiction Detector - Knowledge Conflict Identification
============================================================
Source: Patterns extracted from mothership/knowledge_network.py (KnowledgeEdge relationships)

Detects and manages contradictions between knowledge items:
- Direct contradictions (opposing statements)
- Temporal contradictions (outdated vs current)
- Source conflicts (conflicting sources)
- Confidence conflicts (high vs low confidence on same topic)

Uses relationship edges with "contradicts" type to track conflicts.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ContradictionType(Enum):
    """Types of contradictions between knowledge items"""
    
    DIRECT = "direct"          # Explicitly opposing statements
    TEMPORAL = "temporal"      # Old vs new information
    SOURCE = "source"          # Different sources disagree
    CONFIDENCE = "confidence"  # High vs low confidence conflict
    PARTIAL = "partial"        # Partially overlapping disagreement
    SEMANTIC = "semantic"      # Semantically similar but different conclusions


class ResolutionStrategy(Enum):
    """Strategies for resolving contradictions"""
    
    PREFER_NEWER = "prefer_newer"       # Choose more recent
    PREFER_CONFIDENT = "prefer_confident"  # Choose higher confidence
    PREFER_POPULAR = "prefer_popular"   # Choose more used
    MERGE = "merge"                      # Combine with caveats
    ESCALATE = "escalate"               # Flag for human review
    DEPRECATE_BOTH = "deprecate_both"   # Mark both as uncertain


@dataclass
class Contradiction:
    """Represents a detected contradiction between knowledge items"""
    
    contradiction_id: str
    source_id: str          # First knowledge item
    target_id: str          # Conflicting knowledge item
    contradiction_type: ContradictionType
    severity: float         # 0.0 (minor) to 1.0 (critical)
    description: str
    detected_at: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution: Optional[str] = None
    resolved_at: Optional[datetime] = None
    
    # Context
    overlap_topics: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "contradiction_id": self.contradiction_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.contradiction_type.value,
            "severity": self.severity,
            "description": self.description,
            "detected_at": self.detected_at.isoformat(),
            "resolved": self.resolved,
            "resolution": self.resolution,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "overlap_topics": self.overlap_topics,
        }


@dataclass
class KnowledgeItem:
    """Simplified knowledge item for contradiction checking"""
    
    id: str
    content: str
    title: str = ""
    confidence: float = 0.5
    updated_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    topics: List[str] = field(default_factory=list)
    source: str = ""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeItem":
        """Create from dictionary."""
        updated_at = data.get("updated_at", datetime.now())
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        
        return cls(
            id=str(data.get("id", "")),
            content=str(data.get("content", "")),
            title=str(data.get("title", "")),
            confidence=float(data.get("confidence", 0.5)),
            updated_at=updated_at,
            usage_count=int(data.get("usage_count", 0)),
            topics=data.get("topics", []),
            source=str(data.get("source", "")),
        )


class ContradictionDetector:
    """
    Detects contradictions between knowledge items.
    
    Uses multiple detection strategies:
    1. Topic overlap analysis
    2. Semantic similarity (if embeddings available)
    3. Keyword negation patterns
    4. Temporal conflict detection
    5. Source credibility comparison
    """
    
    # Common negation patterns
    NEGATION_PATTERNS = [
        ("should", "should not"),
        ("always", "never"),
        ("recommended", "not recommended"),
        ("best practice", "anti-pattern"),
        ("do", "don't"),
        ("enable", "disable"),
        ("increase", "decrease"),
        ("true", "false"),
        ("yes", "no"),
        ("correct", "incorrect"),
    ]
    
    def __init__(
        self,
        topic_overlap_threshold: float = 0.5,
        confidence_conflict_threshold: float = 0.3,
        age_conflict_days: int = 90,
    ):
        """
        Initialize detector.
        
        Args:
            topic_overlap_threshold: Min topic overlap to consider related
            confidence_conflict_threshold: Min confidence diff for conflict
            age_conflict_days: Days difference to flag temporal conflict
        """
        self.topic_overlap_threshold = topic_overlap_threshold
        self.confidence_conflict_threshold = confidence_conflict_threshold
        self.age_conflict_days = age_conflict_days
        
        # Store detected contradictions
        self.contradictions: Dict[str, Contradiction] = {}
        
        # Custom detection rules
        self.custom_rules: List[Callable[[KnowledgeItem, KnowledgeItem], Optional[Contradiction]]] = []
    
    def add_detection_rule(
        self,
        rule: Callable[[KnowledgeItem, KnowledgeItem], Optional[Contradiction]],
    ) -> None:
        """
        Add a custom detection rule.
        
        Args:
            rule: Function taking two items, returning Contradiction or None
        """
        self.custom_rules.append(rule)
    
    def _calculate_topic_overlap(
        self,
        topics1: List[str],
        topics2: List[str],
    ) -> Tuple[float, List[str]]:
        """Calculate Jaccard similarity between topic sets."""
        if not topics1 or not topics2:
            return 0.0, []
        
        set1 = set(t.lower() for t in topics1)
        set2 = set(t.lower() for t in topics2)
        
        intersection = set1 & set2
        union = set1 | set2
        
        overlap = len(intersection) / len(union) if union else 0.0
        return overlap, list(intersection)
    
    def _check_negation_patterns(
        self,
        content1: str,
        content2: str,
    ) -> Tuple[bool, str]:
        """Check for negation patterns between content."""
        c1_lower = content1.lower()
        c2_lower = content2.lower()
        
        for positive, negative in self.NEGATION_PATTERNS:
            # Check if one has positive and other has negative
            if positive in c1_lower and negative in c2_lower:
                return True, f"'{positive}' vs '{negative}'"
            if negative in c1_lower and positive in c2_lower:
                return True, f"'{negative}' vs '{positive}'"
        
        return False, ""
    
    def _detect_direct_contradiction(
        self,
        item1: KnowledgeItem,
        item2: KnowledgeItem,
        overlap_topics: List[str],
    ) -> Optional[Contradiction]:
        """Detect direct contradictions via negation patterns."""
        has_negation, pattern = self._check_negation_patterns(
            item1.content, item2.content
        )
        
        if has_negation:
            return Contradiction(
                contradiction_id=f"contra-{item1.id}-{item2.id}",
                source_id=item1.id,
                target_id=item2.id,
                contradiction_type=ContradictionType.DIRECT,
                severity=0.8,
                description=f"Direct contradiction detected: {pattern}",
                overlap_topics=overlap_topics,
                evidence={"pattern": pattern},
            )
        
        return None
    
    def _detect_temporal_contradiction(
        self,
        item1: KnowledgeItem,
        item2: KnowledgeItem,
        overlap_topics: List[str],
    ) -> Optional[Contradiction]:
        """Detect temporal contradictions (old vs new)."""
        age_diff = abs((item1.updated_at - item2.updated_at).days)
        
        if age_diff >= self.age_conflict_days:
            older = item1 if item1.updated_at < item2.updated_at else item2
            newer = item2 if item1.updated_at < item2.updated_at else item1
            
            return Contradiction(
                contradiction_id=f"temporal-{older.id}-{newer.id}",
                source_id=older.id,
                target_id=newer.id,
                contradiction_type=ContradictionType.TEMPORAL,
                severity=0.5,
                description=f"Temporal conflict: {age_diff} days between items on same topic",
                overlap_topics=overlap_topics,
                evidence={
                    "older_date": older.updated_at.isoformat(),
                    "newer_date": newer.updated_at.isoformat(),
                    "age_diff_days": age_diff,
                },
            )
        
        return None
    
    def _detect_source_contradiction(
        self,
        item1: KnowledgeItem,
        item2: KnowledgeItem,
        overlap_topics: List[str],
    ) -> Optional[Contradiction]:
        """Detect contradictions from different sources."""
        if item1.source and item2.source and item1.source != item2.source:
            # Different sources on same topic might conflict
            # This is a weak signal, so lower severity
            return Contradiction(
                contradiction_id=f"source-{item1.id}-{item2.id}",
                source_id=item1.id,
                target_id=item2.id,
                contradiction_type=ContradictionType.SOURCE,
                severity=0.3,
                description=f"Different sources ({item1.source} vs {item2.source}) on overlapping topics",
                overlap_topics=overlap_topics,
                evidence={
                    "source1": item1.source,
                    "source2": item2.source,
                },
            )
        
        return None
    
    def _detect_confidence_contradiction(
        self,
        item1: KnowledgeItem,
        item2: KnowledgeItem,
        overlap_topics: List[str],
    ) -> Optional[Contradiction]:
        """Detect contradictions in confidence levels."""
        conf_diff = abs(item1.confidence - item2.confidence)
        
        if conf_diff >= self.confidence_conflict_threshold:
            higher = item1 if item1.confidence > item2.confidence else item2
            lower = item2 if item1.confidence > item2.confidence else item1
            
            return Contradiction(
                contradiction_id=f"confidence-{lower.id}-{higher.id}",
                source_id=lower.id,
                target_id=higher.id,
                contradiction_type=ContradictionType.CONFIDENCE,
                severity=conf_diff,  # Severity matches confidence gap
                description=f"Confidence conflict: {lower.confidence:.0%} vs {higher.confidence:.0%}",
                overlap_topics=overlap_topics,
                evidence={
                    "lower_confidence": lower.confidence,
                    "higher_confidence": higher.confidence,
                    "difference": conf_diff,
                },
            )
        
        return None
    
    def detect(
        self,
        item1: KnowledgeItem,
        item2: KnowledgeItem,
    ) -> List[Contradiction]:
        """
        Detect all contradictions between two knowledge items.
        
        Args:
            item1: First knowledge item
            item2: Second knowledge item
            
        Returns:
            List of detected contradictions (may be empty)
        """
        contradictions = []
        
        # Calculate topic overlap
        overlap, overlap_topics = self._calculate_topic_overlap(
            item1.topics, item2.topics
        )
        
        # Skip if no topic overlap
        if overlap < self.topic_overlap_threshold:
            return contradictions
        
        # Run detection strategies
        checks = [
            self._detect_direct_contradiction,
            self._detect_temporal_contradiction,
            self._detect_source_contradiction,
            self._detect_confidence_contradiction,
        ]
        
        for check in checks:
            result = check(item1, item2, overlap_topics)
            if result:
                contradictions.append(result)
                self.contradictions[result.contradiction_id] = result
        
        # Run custom rules
        for rule in self.custom_rules:
            try:
                result = rule(item1, item2)
                if result:
                    contradictions.append(result)
                    self.contradictions[result.contradiction_id] = result
            except Exception as e:
                logger.warning(f"Custom rule failed: {e}")
        
        return contradictions
    
    def detect_all(
        self,
        items: List[KnowledgeItem],
    ) -> List[Contradiction]:
        """
        Detect all pairwise contradictions in a list of items.
        
        Args:
            items: List of knowledge items to check
            
        Returns:
            All detected contradictions
        """
        all_contradictions = []
        
        for i, item1 in enumerate(items):
            for item2 in items[i + 1:]:
                contradictions = self.detect(item1, item2)
                all_contradictions.extend(contradictions)
        
        logger.info(f"Detected {len(all_contradictions)} contradictions in {len(items)} items")
        return all_contradictions
    
    def resolve(
        self,
        contradiction_id: str,
        resolution: str,
        strategy: ResolutionStrategy = ResolutionStrategy.ESCALATE,
    ) -> bool:
        """
        Mark a contradiction as resolved.
        
        Args:
            contradiction_id: ID of contradiction to resolve
            resolution: Description of how it was resolved
            strategy: Strategy used for resolution
            
        Returns:
            True if successfully resolved
        """
        if contradiction_id not in self.contradictions:
            return False
        
        contradiction = self.contradictions[contradiction_id]
        contradiction.resolved = True
        contradiction.resolution = f"[{strategy.value}] {resolution}"
        contradiction.resolved_at = datetime.now()
        
        logger.info(f"Resolved contradiction {contradiction_id}: {resolution}")
        return True
    
    def get_unresolved(self, min_severity: float = 0.0) -> List[Contradiction]:
        """Get all unresolved contradictions above severity threshold."""
        return [
            c for c in self.contradictions.values()
            if not c.resolved and c.severity >= min_severity
        ]
    
    def get_by_item(self, item_id: str) -> List[Contradiction]:
        """Get all contradictions involving a specific item."""
        return [
            c for c in self.contradictions.values()
            if c.source_id == item_id or c.target_id == item_id
        ]


class ContradictionResolver:
    """
    Resolves contradictions using configured strategies.
    """
    
    def __init__(self, detector: ContradictionDetector):
        self.detector = detector
    
    def auto_resolve(
        self,
        contradiction: Contradiction,
        items: Dict[str, KnowledgeItem],
        strategy: Optional[ResolutionStrategy] = None,
    ) -> Tuple[str, str]:
        """
        Auto-resolve a contradiction.
        
        Args:
            contradiction: The contradiction to resolve
            items: Dict of item_id -> KnowledgeItem
            strategy: Override strategy (else auto-select based on type)
            
        Returns:
            Tuple of (winner_id, resolution_description)
        """
        source = items.get(contradiction.source_id)
        target = items.get(contradiction.target_id)
        
        if not source or not target:
            return "", "Missing items for resolution"
        
        # Auto-select strategy based on contradiction type
        if strategy is None:
            strategy = self._select_strategy(contradiction.contradiction_type)
        
        # Apply strategy
        if strategy == ResolutionStrategy.PREFER_NEWER:
            winner = source if source.updated_at > target.updated_at else target
            resolution = f"Preferred newer item: {winner.id}"
            
        elif strategy == ResolutionStrategy.PREFER_CONFIDENT:
            winner = source if source.confidence > target.confidence else target
            resolution = f"Preferred higher confidence: {winner.id} ({winner.confidence:.0%})"
            
        elif strategy == ResolutionStrategy.PREFER_POPULAR:
            winner = source if source.usage_count > target.usage_count else target
            resolution = f"Preferred more used: {winner.id} ({winner.usage_count} uses)"
            
        elif strategy == ResolutionStrategy.MERGE:
            winner = source  # Arbitrary, both should be merged
            resolution = f"Merge recommended: combine {source.id} and {target.id}"
            
        else:  # ESCALATE or DEPRECATE_BOTH
            winner = source  # Arbitrary
            resolution = "Escalated for human review"
        
        # Mark resolved
        self.detector.resolve(
            contradiction.contradiction_id,
            resolution,
            strategy,
        )
        
        return winner.id if winner else "", resolution
    
    def _select_strategy(self, contradiction_type: ContradictionType) -> ResolutionStrategy:
        """Select appropriate resolution strategy based on contradiction type."""
        mapping = {
            ContradictionType.TEMPORAL: ResolutionStrategy.PREFER_NEWER,
            ContradictionType.CONFIDENCE: ResolutionStrategy.PREFER_CONFIDENT,
            ContradictionType.SOURCE: ResolutionStrategy.ESCALATE,
            ContradictionType.DIRECT: ResolutionStrategy.ESCALATE,
            ContradictionType.PARTIAL: ResolutionStrategy.MERGE,
            ContradictionType.SEMANTIC: ResolutionStrategy.PREFER_CONFIDENT,
        }
        return mapping.get(contradiction_type, ResolutionStrategy.ESCALATE)


# Convenience functions
def create_detector(
    topic_threshold: float = 0.5,
    confidence_threshold: float = 0.3,
    age_days: int = 90,
) -> ContradictionDetector:
    """Create configured ContradictionDetector."""
    return ContradictionDetector(
        topic_overlap_threshold=topic_threshold,
        confidence_conflict_threshold=confidence_threshold,
        age_conflict_days=age_days,
    )


def quick_check(item1: Dict, item2: Dict) -> List[Contradiction]:
    """Quick check for contradictions between two items."""
    detector = ContradictionDetector()
    k1 = KnowledgeItem.from_dict(item1)
    k2 = KnowledgeItem.from_dict(item2)
    return detector.detect(k1, k2)


# Demo
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("CONTRADICTION DETECTOR DEMO")
    print("=" * 60)
    
    detector = create_detector()
    
    # Create test items
    items = [
        KnowledgeItem(
            id="k1",
            title="Password Storage Best Practice",
            content="Always use bcrypt for password hashing. Never store passwords in plaintext.",
            confidence=0.95,
            updated_at=datetime.now(),
            topics=["security", "passwords", "authentication"],
            source="security_agent",
        ),
        KnowledgeItem(
            id="k2",
            title="Legacy Password Guide",
            content="MD5 is recommended for password hashing. You should not use bcrypt due to performance.",
            confidence=0.6,
            updated_at=datetime(2020, 1, 1),
            topics=["security", "passwords", "performance"],
            source="legacy_docs",
        ),
        KnowledgeItem(
            id="k3",
            title="API Rate Limiting",
            content="Always enable rate limiting on public endpoints.",
            confidence=0.9,
            topics=["api", "security", "performance"],
            source="api_agent",
        ),
    ]
    
    print("\n🔍 Checking for contradictions...\n")
    
    all_contradictions = detector.detect_all(items)
    
    for i, c in enumerate(all_contradictions, 1):
        print(f"{i}. {c.contradiction_type.value.upper()} Contradiction")
        print(f"   Between: {c.source_id} ↔ {c.target_id}")
        print(f"   Severity: {c.severity:.1%}")
        print(f"   Description: {c.description}")
        print(f"   Topics: {', '.join(c.overlap_topics)}")
        print()
    
    # Resolve
    if all_contradictions:
        print("⚡ Auto-resolving contradictions...\n")
        items_dict = {item.id: item for item in items}
        resolver = ContradictionResolver(detector)
        
        for c in all_contradictions:
            winner, resolution = resolver.auto_resolve(c, items_dict)
            print(f"   ✓ {c.contradiction_id}: {resolution}")
    
    print("\n" + "=" * 60)
    print("✅ Contradiction Detector ready for integration")
