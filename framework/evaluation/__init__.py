"""
Evaluation Module - A/B Testing and Performance Metrics.

This module provides the evaluation and continuous improvement framework:
- ABTester: A/B testing with statistical significance analysis
- StrategyMetrics: Performance metrics collection and analysis
- RewardCalculator: RL reward signals for agent training
- AlertManager: Threshold-based alerting
- AdjustmentTrigger: Automatic adjustment recommendations

Strategic Intent:
Continuous improvement requires quantifiable metrics and rigorous experimentation.
This module enables data-driven optimization of agent strategies through A/B testing,
performance tracking, and automated recommendations.
"""

from framework.evaluation.ab_tester import (
    ABTester,
    ExperimentConfig,
    Experiment,
    Variant,
    VariantType,
    ExperimentResults,
    ExperimentStatus,
    MetricGoal,
    get_ab_tester,
    create_experiment,
    calculate_z_score,
    z_to_p_value,
    calculate_confidence_interval,
    calculate_sample_size,
)
from framework.evaluation.strategy_metrics import (
    StrategyMetrics,
    StrategyMetricsConfig,
    MetricCollector,
    MetricsAggregator,
    TrendAnalyzer,
    RewardCalculator,
    AlertManager,
    AdjustmentTrigger,
    MetricType,
    TimeWindow,
    TrendDirection,
    AlertSeverity,
    RewardType,
    MetricDataPoint,
    AggregatedMetric,
    TrendAnalysis,
    Alert,
    Reward,
    AdjustmentRecommendation,
    create_strategy_metrics,
    get_strategy_metrics,
    reset_global_metrics,
)

__all__ = [
    # A/B Testing
    "ABTester",
    "ExperimentConfig",
    "Experiment",
    "Variant",
    "VariantType",
    "ExperimentResults",
    "ExperimentStatus",
    "MetricGoal",
    "get_ab_tester",
    "create_experiment",
    "calculate_z_score",
    "z_to_p_value",
    "calculate_confidence_interval",
    "calculate_sample_size",
    # Strategy Metrics
    "StrategyMetrics",
    "StrategyMetricsConfig",
    "MetricCollector",
    "MetricsAggregator",
    "TrendAnalyzer",
    "RewardCalculator",
    "AlertManager",
    "AdjustmentTrigger",
    # Enums
    "MetricType",
    "TimeWindow",
    "TrendDirection",
    "AlertSeverity",
    "RewardType",
    # Data classes
    "MetricDataPoint",
    "AggregatedMetric",
    "TrendAnalysis",
    "Alert",
    "Reward",
    "AdjustmentRecommendation",
    # Factory functions
    "create_strategy_metrics",
    "get_strategy_metrics",
    "reset_global_metrics",
]
