"""
Promotion Service - Knowledge Value Promotion & Management
============================================================
Source: Combines patterns from PriorityKnowledgeSharing and KnowledgeNetwork

Manages the promotion of high-value knowledge items:
- Score-based promotion to priority tables
- Auto-trimming of low-value items
- Contradiction-aware promotion
- Cross-agent knowledge sharing
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

from .priority_scorer import PriorityScorer, PriorityScore
from .contradiction_detector import ContradictionDetector

logger = logging.getLogger(__name__)


@dataclass
class PromotionResult:
    """Result of a promotion operation"""
    
    promoted: bool
    item_id: str
    score: float
    tier: str  # "priority", "standard", "deprecated"
    reason: str
    contradictions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "promoted": self.promoted,
            "item_id": self.item_id,
            "score": round(self.score, 4),
            "tier": self.tier,
            "reason": self.reason,
            "contradictions": self.contradictions,
        }


@dataclass
class PromotionConfig:
    """Configuration for promotion thresholds"""
    
    priority_threshold: float = 0.7     # Score needed for priority tier
    deprecation_threshold: float = 0.2  # Score below which to deprecate
    max_priority_items: int = 100       # Maximum items in priority tier
    trim_percentage: float = 0.2        # Trim bottom 20% when over limit
    require_no_contradictions: bool = True  # Block promotion if contradictions exist


class PromotionService:
    """
    Service for promoting high-value knowledge items.
    
    Features:
    - Automatic tier assignment based on priority scores
    - Contradiction checking before promotion
    - Capacity management with intelligent trimming
    - Event hooks for integration
    """
    
    def __init__(
        self,
        config: Optional[PromotionConfig] = None,
        scorer: Optional[PriorityScorer] = None,
        detector: Optional[ContradictionDetector] = None,
    ):
        self.config = config or PromotionConfig()
        self.scorer = scorer or PriorityScorer()
        self.detector = detector or ContradictionDetector()
        
        # Storage by tier
        self.priority_items: Dict[str, Dict[str, Any]] = {}
        self.standard_items: Dict[str, Dict[str, Any]] = {}
        self.deprecated_items: Dict[str, Dict[str, Any]] = {}
        
        # Score cache
        self.scores: Dict[str, PriorityScore] = {}
        
        # Event hooks
        self.on_promoted: List[Callable[[PromotionResult], None]] = []
        self.on_demoted: List[Callable[[str, str, str], None]] = []
        self.on_trimmed: List[Callable[[List[str]], None]] = []
    
    def add_hook(
        self,
        event: str,
        callback: Callable,
    ) -> None:
        """
        Add an event hook.
        
        Events:
        - "promoted": Called with PromotionResult
        - "demoted": Called with (item_id, old_tier, new_tier)
        - "trimmed": Called with list of trimmed item IDs
        """
        if event == "promoted":
            self.on_promoted.append(callback)
        elif event == "demoted":
            self.on_demoted.append(callback)
        elif event == "trimmed":
            self.on_trimmed.append(callback)
    
    def _get_item_for_scoring(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize item dict for scoring."""
        return {
            "id": item.get("id", ""),
            "confidence": item.get("confidence", 0.5),
            "updated_at": item.get("updated_at", datetime.now()),
            "usage_count": item.get("usage_count", 0),
        }
    
    def _get_contradictions(self, item_id: str) -> List[str]:
        """Get unresolved contradictions for an item."""
        contradictions = self.detector.get_by_item(item_id)
        return [
            c.contradiction_id
            for c in contradictions
            if not c.resolved
        ]
    
    def _determine_tier(self, score: float) -> str:
        """Determine tier based on score."""
        if score >= self.config.priority_threshold:
            return "priority"
        elif score <= self.config.deprecation_threshold:
            return "deprecated"
        return "standard"
    
    def evaluate(self, item: Dict[str, Any]) -> PromotionResult:
        """
        Evaluate an item for promotion without making changes.
        
        Args:
            item: Knowledge item to evaluate
            
        Returns:
            PromotionResult indicating what would happen
        """
        item_id = str(item.get("id", ""))
        
        # Calculate score
        score_item = self._get_item_for_scoring(item)
        priority_score = self.scorer.score(
            item_id=item_id,
            confidence=score_item["confidence"],
            updated_at=score_item["updated_at"],
            usage_count=score_item["usage_count"],
        )
        
        # Check contradictions
        contradictions = self._get_contradictions(item_id)
        
        # Determine tier
        tier = self._determine_tier(priority_score.total_score)
        
        # Check if promotion would be blocked
        promoted = tier == "priority"
        reason = f"Score {priority_score.total_score:.3f} qualifies for {tier} tier"
        
        if promoted and contradictions and self.config.require_no_contradictions:
            promoted = False
            tier = "standard"
            reason = f"Blocked: {len(contradictions)} unresolved contradiction(s)"
        
        return PromotionResult(
            promoted=promoted,
            item_id=item_id,
            score=priority_score.total_score,
            tier=tier,
            reason=reason,
            contradictions=contradictions,
        )
    
    def promote(self, item: Dict[str, Any]) -> PromotionResult:
        """
        Promote an item to its appropriate tier.
        
        Args:
            item: Knowledge item to promote
            
        Returns:
            PromotionResult with outcome
        """
        result = self.evaluate(item)
        item_id = result.item_id
        
        # Remove from any existing tier
        for tier_dict in [self.priority_items, self.standard_items, self.deprecated_items]:
            tier_dict.pop(item_id, None)
        
        # Add to appropriate tier
        if result.tier == "priority":
            self.priority_items[item_id] = item
        elif result.tier == "deprecated":
            self.deprecated_items[item_id] = item
        else:
            self.standard_items[item_id] = item
        
        # Cache score
        score_item = self._get_item_for_scoring(item)
        self.scores[item_id] = self.scorer.score(
            item_id=item_id,
            confidence=score_item["confidence"],
            updated_at=score_item["updated_at"],
            usage_count=score_item["usage_count"],
        )
        
        # Trigger hooks
        for hook in self.on_promoted:
            try:
                hook(result)
            except Exception as e:
                logger.warning(f"Promotion hook failed: {e}")
        
        # Trim if needed
        self._trim_if_needed()
        
        logger.info(f"Promoted {item_id} to {result.tier}: {result.reason}")
        return result
    
    def promote_batch(self, items: List[Dict[str, Any]]) -> List[PromotionResult]:
        """Promote multiple items at once."""
        results = []
        for item in items:
            result = self.promote(item)
            results.append(result)
        return results
    
    def demote(self, item_id: str, to_tier: str = "standard") -> bool:
        """
        Demote an item to a lower tier.
        
        Args:
            item_id: ID of item to demote
            to_tier: Target tier ("standard" or "deprecated")
            
        Returns:
            True if successfully demoted
        """
        item = None
        old_tier = None
        
        # Find current location
        if item_id in self.priority_items:
            item = self.priority_items.pop(item_id)
            old_tier = "priority"
        elif item_id in self.standard_items:
            item = self.standard_items.pop(item_id)
            old_tier = "standard"
        elif item_id in self.deprecated_items:
            item = self.deprecated_items.pop(item_id)
            old_tier = "deprecated"
        
        if not item:
            return False
        
        # Move to new tier
        if to_tier == "deprecated":
            self.deprecated_items[item_id] = item
        else:
            self.standard_items[item_id] = item
        
        # Trigger hooks
        for hook in self.on_demoted:
            try:
                hook(item_id, old_tier, to_tier)
            except Exception as e:
                logger.warning(f"Demotion hook failed: {e}")
        
        logger.info(f"Demoted {item_id}: {old_tier} → {to_tier}")
        return True
    
    def _trim_if_needed(self) -> None:
        """Trim priority tier if over capacity."""
        if len(self.priority_items) <= self.config.max_priority_items:
            return
        
        # Sort by score ascending (lowest first)
        sorted_items = sorted(
            self.priority_items.keys(),
            key=lambda x: self.scores.get(x, PriorityScore(x, 0, 0, 0, 0)).total_score,
        )
        
        # Calculate how many to trim
        trim_count = int(len(sorted_items) * self.config.trim_percentage)
        to_trim = sorted_items[:trim_count]
        
        # Demote lowest items
        for item_id in to_trim:
            item = self.priority_items.pop(item_id, None)
            if item:
                self.standard_items[item_id] = item
        
        # Trigger hooks
        for hook in self.on_trimmed:
            try:
                hook(to_trim)
            except Exception as e:
                logger.warning(f"Trim hook failed: {e}")
        
        logger.info(f"Trimmed {len(to_trim)} items from priority tier")
    
    def get_priority_items(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get top priority items sorted by score.
        
        Args:
            limit: Maximum items to return
            
        Returns:
            List of items with their scores
        """
        items_with_scores = []
        
        for item_id, item in self.priority_items.items():
            score = self.scores.get(item_id)
            items_with_scores.append({
                **item,
                "priority_score": score.total_score if score else 0,
            })
        
        # Sort by score descending
        items_with_scores.sort(
            key=lambda x: x.get("priority_score", 0),
            reverse=True,
        )
        
        return items_with_scores[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get promotion service statistics."""
        return {
            "priority_count": len(self.priority_items),
            "standard_count": len(self.standard_items),
            "deprecated_count": len(self.deprecated_items),
            "max_priority": self.config.max_priority_items,
            "capacity_used": len(self.priority_items) / self.config.max_priority_items,
            "total_items": (
                len(self.priority_items) +
                len(self.standard_items) +
                len(self.deprecated_items)
            ),
        }
    
    def rescore_all(self) -> int:
        """
        Rescore all items and adjust tiers.
        
        Useful for periodic maintenance.
        
        Returns:
            Number of items that changed tiers
        """
        changes = 0
        all_items = []
        
        for tier_name, tier_dict in [
            ("priority", self.priority_items),
            ("standard", self.standard_items),
            ("deprecated", self.deprecated_items),
        ]:
            for item_id, item in list(tier_dict.items()):
                all_items.append((item_id, item, tier_name))
        
        for item_id, item, old_tier in all_items:
            result = self.promote(item)
            if result.tier != old_tier:
                changes += 1
                logger.info(f"Tier change: {item_id} {old_tier} → {result.tier}")
        
        return changes


class CrossAgentPromoter:
    """
    Promotes knowledge across agent boundaries.
    
    Handles knowledge sharing between agents based on subscriptions.
    """
    
    def __init__(self, promotion_service: PromotionService):
        self.service = promotion_service
        
        # Agent subscriptions: agent_id -> set of knowledge types
        self.subscriptions: Dict[str, Set[str]] = {}
        
        # Shared knowledge pool
        self.shared_pool: Dict[str, Dict[str, Any]] = {}
    
    def subscribe(self, agent_id: str, knowledge_types: List[str]) -> None:
        """Subscribe an agent to knowledge types."""
        self.subscriptions[agent_id] = set(knowledge_types)
        logger.info(f"Agent {agent_id} subscribed to: {knowledge_types}")
    
    def share(
        self,
        item: Dict[str, Any],
        from_agent: str,
        knowledge_type: str,
    ) -> List[str]:
        """
        Share knowledge with subscribed agents.
        
        Args:
            item: Knowledge item to share
            from_agent: Source agent ID
            knowledge_type: Type of knowledge
            
        Returns:
            List of agent IDs that received the knowledge
        """
        # Evaluate for promotion
        result = self.service.evaluate(item)
        
        if not result.promoted:
            logger.debug(f"Item {result.item_id} not shared: {result.reason}")
            return []
        
        # Find subscribed agents
        recipients = []
        for agent_id, types in self.subscriptions.items():
            if agent_id != from_agent and knowledge_type in types:
                recipients.append(agent_id)
        
        # Add to shared pool
        if recipients:
            self.shared_pool[result.item_id] = {
                **item,
                "shared_from": from_agent,
                "shared_to": recipients,
                "shared_at": datetime.now().isoformat(),
                "knowledge_type": knowledge_type,
            }
        
        logger.info(f"Shared {result.item_id} from {from_agent} to {len(recipients)} agents")
        return recipients
    
    def get_shared_for_agent(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get all shared knowledge available to an agent."""
        return [
            item for item in self.shared_pool.values()
            if agent_id in item.get("shared_to", [])
        ]


# Convenience functions
def create_promotion_service(
    priority_threshold: float = 0.7,
    max_priority: int = 100,
) -> PromotionService:
    """Create configured PromotionService."""
    config = PromotionConfig(
        priority_threshold=priority_threshold,
        max_priority_items=max_priority,
    )
    return PromotionService(config=config)


# Demo
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("PROMOTION SERVICE DEMO")
    print("=" * 60)
    
    service = create_promotion_service(
        priority_threshold=0.6,
        max_priority=5,
    )
    
    # Add some hook to see promotions
    def log_promotion(result: PromotionResult):
        emoji = "⭐" if result.tier == "priority" else "📄"
        print(f"   {emoji} {result.item_id}: {result.tier} (score: {result.score:.3f})")
    
    service.add_hook("promoted", log_promotion)
    
    # Create test items
    items = [
        {
            "id": "best-practice-1",
            "title": "Always validate input",
            "confidence": 0.95,
            "updated_at": datetime.now(),
            "usage_count": 150,
        },
        {
            "id": "insight-2",
            "title": "Caching improves latency",
            "confidence": 0.85,
            "updated_at": datetime.now(),
            "usage_count": 80,
        },
        {
            "id": "pattern-3",
            "title": "Retry with backoff",
            "confidence": 0.7,
            "updated_at": datetime.now(),
            "usage_count": 45,
        },
        {
            "id": "old-practice-4",
            "title": "Legacy approach",
            "confidence": 0.3,
            "updated_at": datetime(2020, 1, 1),
            "usage_count": 5,
        },
    ]
    
    print("\n📊 Promoting items...\n")
    
    results = service.promote_batch(items)
    
    print("\n📈 Stats:")
    stats = service.get_stats()
    print(f"   Priority tier: {stats['priority_count']}/{stats['max_priority']}")
    print(f"   Standard tier: {stats['standard_count']}")
    print(f"   Deprecated: {stats['deprecated_count']}")
    
    print("\n🏆 Top priority items:")
    for item in service.get_priority_items(limit=3):
        print(f"   • {item['title']} (score: {item['priority_score']:.3f})")
    
    print("\n" + "=" * 60)
    print("✅ Promotion Service ready for integration")
