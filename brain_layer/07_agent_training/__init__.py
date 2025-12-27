"""
Agent Training & Strategy Module

Autonomous learning layer components for continuous agent improvement.

Components:
- ModelTrainer: Continuous retraining with versioning
- ABTestFramework: Multi-variant testing for strategies
- FeedbackLoop: Performance tracking and degradation detection
- ReinforcementLearning: Q-learning agent with preference adaptation

Source: F:\Mothership-main\Mothership-main\learning\ and agents\
Recycled: December 26, 2024
"""

from .model_trainer import (
    ModelTrainer,
    ModelType,
    TrainingStatus,
    TrainingConfig,
    ModelMetrics,
    TrainingJob,
)

from .ab_testing import (
    ABTestFramework,
    ABTest,
    TestVariant,
    TestResults,
    TestStatus,
    VariantType,
)

from .feedback_loop import (
    FeedbackLoop,
    PerformanceMetrics,
    PerformanceAlert,
    ImprovementTracker,
    MetricType,
    AlertSeverity,
)

from .reinforcement_learning import (
    ReinforcementLearning,
    State,
    Action,
    Reward,
    Experience,
)

__all__ = [
    # Model Training
    "ModelTrainer",
    "ModelType",
    "TrainingStatus",
    "TrainingConfig",
    "ModelMetrics",
    "TrainingJob",
    # A/B Testing
    "ABTestFramework",
    "ABTest",
    "TestVariant",
    "TestResults",
    "TestStatus",
    "VariantType",
    # Feedback Loop
    "FeedbackLoop",
    "PerformanceMetrics",
    "PerformanceAlert",
    "ImprovementTracker",
    "MetricType",
    "AlertSeverity",
    # Reinforcement Learning
    "ReinforcementLearning",
    "State",
    "Action",
    "Reward",
    "Experience",
]
