"""
Priority Scorer - Knowledge Value Ranking System
=================================================
Source: Extracted from mothership/memory_system_improvements.py (PriorityKnowledgeSharing)

Calculates composite priority scores for knowledge atoms based on:
- Confidence: How certain the knowledge is (0.0-1.0)
- Recency: How fresh the knowledge is (exponential decay)
- Usage: How often it's been accessed/applied

Formula: priority = (confidence * 0.4) + (recency * 0.3) + (usage_normalized * 0.3)
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Protocol


class Scoreable(Protocol):
    """Protocol for objects that can be scored"""
    
    @property
    def id(self) -> str: ...
    
    @property
    def confidence(self) -> float: ...
    
    @property
    def updated_at(self) -> datetime: ...
    
    @property
    def usage_count(self) -> int: ...


@dataclass
class PriorityScore:
    """Result of priority scoring"""
    
    item_id: str
    total_score: float
    confidence_score: float
    recency_score: float
    usage_score: float
    calculated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "total_score": round(self.total_score, 4),
            "confidence_score": round(self.confidence_score, 4),
            "recency_score": round(self.recency_score, 4),
            "usage_score": round(self.usage_score, 4),
            "calculated_at": self.calculated_at.isoformat(),
        }


@dataclass
class ScoringWeights:
    """Configurable weights for priority calculation"""
    
    confidence: float = 0.4
    recency: float = 0.3
    usage: float = 0.3
    
    def __post_init__(self):
        total = self.confidence + self.recency + self.usage
        if abs(total - 1.0) > 0.001:
            # Normalize weights if they don't sum to 1.0
            self.confidence /= total
            self.recency /= total
            self.usage /= total


class PriorityScorer:
    """
    Priority-based knowledge scoring system.
    
    Calculates composite scores to rank knowledge items for:
    - Promotion to priority tables
    - Cache eviction decisions
    - Result ranking in searches
    - Best practices selection
    """
    
    def __init__(
        self,
        weights: Optional[ScoringWeights] = None,
        decay_days: int = 30,
        usage_cap: int = 100,
    ):
        """
        Initialize the scorer.
        
        Args:
            weights: Scoring weights (default: 0.4/0.3/0.3)
            decay_days: Half-life for recency decay (default: 30 days)
            usage_cap: Maximum usage count for normalization (default: 100)
        """
        self.weights = weights or ScoringWeights()
        self.decay_days = decay_days
        self.usage_cap = usage_cap
    
    def calculate_confidence_score(self, confidence: float) -> float:
        """
        Calculate normalized confidence component.
        
        Args:
            confidence: Raw confidence value (0.0-1.0)
            
        Returns:
            Weighted confidence score
        """
        # Clamp to valid range
        confidence = max(0.0, min(1.0, confidence))
        return confidence * self.weights.confidence
    
    def calculate_recency_score(self, updated_at: datetime) -> float:
        """
        Calculate recency component with exponential decay.
        
        Uses exponential decay: score = e^(-age/decay_period)
        
        Args:
            updated_at: Timestamp of last update
            
        Returns:
            Weighted recency score
        """
        age_days = (datetime.now() - updated_at).total_seconds() / 86400
        
        # Exponential decay
        decay_factor = math.exp(-age_days / self.decay_days)
        
        return decay_factor * self.weights.recency
    
    def calculate_usage_score(self, usage_count: int) -> float:
        """
        Calculate usage component with capped normalization.
        
        Args:
            usage_count: Number of times item has been accessed
            
        Returns:
            Weighted usage score
        """
        # Normalize to 0-1 range, capped at usage_cap
        normalized = min(usage_count / self.usage_cap, 1.0)
        return normalized * self.weights.usage
    
    def score(
        self,
        item_id: str,
        confidence: float,
        updated_at: datetime,
        usage_count: int = 0,
    ) -> PriorityScore:
        """
        Calculate composite priority score for a knowledge item.
        
        Args:
            item_id: Unique identifier for the item
            confidence: Confidence value (0.0-1.0)
            updated_at: Last update timestamp
            usage_count: Access/usage count
            
        Returns:
            PriorityScore with component breakdown
        """
        confidence_score = self.calculate_confidence_score(confidence)
        recency_score = self.calculate_recency_score(updated_at)
        usage_score = self.calculate_usage_score(usage_count)
        
        total = confidence_score + recency_score + usage_score
        
        return PriorityScore(
            item_id=item_id,
            total_score=total,
            confidence_score=confidence_score,
            recency_score=recency_score,
            usage_score=usage_score,
        )
    
    def score_item(self, item: Scoreable) -> PriorityScore:
        """
        Calculate priority score for an object implementing Scoreable protocol.
        
        Args:
            item: Object with id, confidence, updated_at, usage_count properties
            
        Returns:
            PriorityScore
        """
        return self.score(
            item_id=item.id,
            confidence=item.confidence,
            updated_at=item.updated_at,
            usage_count=item.usage_count,
        )
    
    def rank_items(
        self,
        items: List[Dict[str, Any]],
        id_key: str = "id",
        confidence_key: str = "confidence",
        updated_key: str = "updated_at",
        usage_key: str = "usage_count",
    ) -> List[PriorityScore]:
        """
        Rank a list of items by priority score.
        
        Args:
            items: List of dictionaries with scoring fields
            id_key: Key for item ID
            confidence_key: Key for confidence value
            updated_key: Key for timestamp (str or datetime)
            usage_key: Key for usage count
            
        Returns:
            List of PriorityScore objects sorted by total_score descending
        """
        scores = []
        
        for item in items:
            item_id = str(item.get(id_key, "unknown"))
            confidence = float(item.get(confidence_key, 0.5))
            
            # Handle timestamp
            updated_raw = item.get(updated_key, datetime.now())
            if isinstance(updated_raw, str):
                try:
                    updated_at = datetime.fromisoformat(updated_raw)
                except ValueError:
                    updated_at = datetime.now()
            else:
                updated_at = updated_raw
            
            usage_count = int(item.get(usage_key, 0))
            
            score = self.score(item_id, confidence, updated_at, usage_count)
            scores.append(score)
        
        # Sort by total score descending
        scores.sort(key=lambda s: s.total_score, reverse=True)
        
        return scores
    
    def get_top_n(
        self,
        items: List[Dict[str, Any]],
        n: int = 10,
        **kwargs,
    ) -> List[PriorityScore]:
        """
        Get top N items by priority score.
        
        Args:
            items: List of items to score
            n: Number of top items to return
            **kwargs: Field key mappings for rank_items()
            
        Returns:
            Top N PriorityScore objects
        """
        ranked = self.rank_items(items, **kwargs)
        return ranked[:n]
    
    def filter_above_threshold(
        self,
        items: List[Dict[str, Any]],
        threshold: float = 0.5,
        **kwargs,
    ) -> List[PriorityScore]:
        """
        Filter items above a score threshold.
        
        Args:
            items: List of items to score
            threshold: Minimum total_score to include
            **kwargs: Field key mappings for rank_items()
            
        Returns:
            PriorityScore objects above threshold
        """
        ranked = self.rank_items(items, **kwargs)
        return [s for s in ranked if s.total_score >= threshold]


class AdaptiveScorer(PriorityScorer):
    """
    Adaptive priority scorer that adjusts weights based on feedback.
    
    Tracks scoring effectiveness and can auto-tune weights.
    """
    
    def __init__(self, *args, learning_rate: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_rate = learning_rate
        self.feedback_history: List[Dict[str, Any]] = []
    
    def record_feedback(
        self,
        score: PriorityScore,
        was_useful: bool,
        impact: float = 1.0,
    ) -> None:
        """
        Record feedback on a scoring decision.
        
        Args:
            score: The PriorityScore that was used
            was_useful: Whether the high-priority item was actually useful
            impact: Magnitude of the outcome (0.0-1.0+)
        """
        self.feedback_history.append({
            "score": score.to_dict(),
            "was_useful": was_useful,
            "impact": impact,
            "timestamp": datetime.now().isoformat(),
        })
    
    def adapt_weights(self) -> ScoringWeights:
        """
        Analyze feedback and adjust weights.
        
        Simple adaptation: boost weights of components that
        correlate with positive outcomes.
        
        Returns:
            New adjusted weights
        """
        if len(self.feedback_history) < 10:
            return self.weights
        
        # Analyze recent feedback
        recent = self.feedback_history[-50:]
        
        # Calculate correlation between components and success
        adjustments = {"confidence": 0.0, "recency": 0.0, "usage": 0.0}
        
        for fb in recent:
            score = fb["score"]
            success = 1.0 if fb["was_useful"] else -1.0
            
            # Each component contributes to adjustment if successful
            for key in adjustments:
                component = score.get(f"{key}_score", 0)
                adjustments[key] += component * success * self.learning_rate
        
        # Apply adjustments
        new_weights = ScoringWeights(
            confidence=max(0.1, self.weights.confidence + adjustments["confidence"]),
            recency=max(0.1, self.weights.recency + adjustments["recency"]),
            usage=max(0.1, self.weights.usage + adjustments["usage"]),
        )
        
        self.weights = new_weights
        return new_weights


# Convenience functions
def create_scorer(
    confidence_weight: float = 0.4,
    recency_weight: float = 0.3,
    usage_weight: float = 0.3,
    decay_days: int = 30,
) -> PriorityScorer:
    """Create a configured PriorityScorer."""
    weights = ScoringWeights(
        confidence=confidence_weight,
        recency=recency_weight,
        usage=usage_weight,
    )
    return PriorityScorer(weights=weights, decay_days=decay_days)


def quick_score(
    confidence: float,
    days_old: int = 0,
    usage_count: int = 0,
) -> float:
    """Quick single-item scoring without full scorer setup."""
    scorer = PriorityScorer()
    updated_at = datetime.now() - timedelta(days=days_old)
    score = scorer.score("temp", confidence, updated_at, usage_count)
    return score.total_score


# Demo
if __name__ == "__main__":
    print("=" * 60)
    print("PRIORITY SCORER DEMO")
    print("=" * 60)
    
    # Create scorer with default weights
    scorer = create_scorer()
    
    # Sample knowledge items
    items = [
        {
            "id": "k1",
            "title": "SQL Injection Prevention",
            "confidence": 0.95,
            "updated_at": datetime.now() - timedelta(days=5),
            "usage_count": 45,
        },
        {
            "id": "k2",
            "title": "OAuth2 Best Practices",
            "confidence": 0.85,
            "updated_at": datetime.now() - timedelta(days=60),
            "usage_count": 120,
        },
        {
            "id": "k3",
            "title": "New API Rate Limiting",
            "confidence": 0.70,
            "updated_at": datetime.now() - timedelta(days=1),
            "usage_count": 5,
        },
    ]
    
    print("\n📊 Scoring knowledge items...\n")
    
    ranked = scorer.rank_items(
        items,
        id_key="id",
        confidence_key="confidence",
        updated_key="updated_at",
        usage_key="usage_count",
    )
    
    for i, score in enumerate(ranked, 1):
        item = next(x for x in items if x["id"] == score.item_id)
        print(f"{i}. {item['title']}")
        print(f"   Total Score: {score.total_score:.3f}")
        print(f"   - Confidence: {score.confidence_score:.3f}")
        print(f"   - Recency:    {score.recency_score:.3f}")
        print(f"   - Usage:      {score.usage_score:.3f}")
        print()
    
    print("=" * 60)
    print("✅ Priority Scorer ready for integration")
