"""
Promotion & Contradiction Logic
================================
Section 6 of the Brain Layer taxonomy.

This package provides knowledge quality management:
- Priority scoring for value ranking
- Contradiction detection and resolution
- Promotion service for tier management
- Cross-agent knowledge sharing

Components
----------
priority_scorer:
    Calculate composite priority scores based on confidence, recency, usage.
    
contradiction_detector:
    Detect and manage conflicting knowledge items.
    
promotion_service:
    Promote high-value items to priority tiers, manage capacity.

Usage
-----
    from brain_layer.promotion_contradiction import (
        PriorityScorer,
        ContradictionDetector,
        PromotionService,
        create_promotion_service,
    )
    
    # Score items
    scorer = PriorityScorer()
    score = scorer.score("item1", confidence=0.9, updated_at=datetime.now(), usage_count=50)
    
    # Detect contradictions
    detector = ContradictionDetector()
    contradictions = detector.detect(item1, item2)
    
    # Promote to tiers
    service = create_promotion_service(priority_threshold=0.7)
    result = service.promote(item)

Sources
-------
- PriorityKnowledgeSharing: mothership/memory_system_improvements.py
- KnowledgeEdge/contradicts: mothership/knowledge_network.py
"""

from .priority_scorer import (
    PriorityScorer,
    PriorityScore,
    ScoringWeights,
    AdaptiveScorer,
    create_scorer,
    quick_score,
)

from .contradiction_detector import (
    ContradictionDetector,
    ContradictionResolver,
    Contradiction,
    ContradictionType,
    ResolutionStrategy,
    KnowledgeItem,
    create_detector,
    quick_check,
)

from .promotion_service import (
    PromotionService,
    PromotionResult,
    PromotionConfig,
    CrossAgentPromoter,
    create_promotion_service,
)

__all__ = [
    # Priority Scoring
    "PriorityScorer",
    "PriorityScore",
    "ScoringWeights",
    "AdaptiveScorer",
    "create_scorer",
    "quick_score",
    
    # Contradiction Detection
    "ContradictionDetector",
    "ContradictionResolver",
    "Contradiction",
    "ContradictionType",
    "ResolutionStrategy",
    "KnowledgeItem",
    "create_detector",
    "quick_check",
    
    # Promotion Service
    "PromotionService",
    "PromotionResult",
    "PromotionConfig",
    "CrossAgentPromoter",
    "create_promotion_service",
]
