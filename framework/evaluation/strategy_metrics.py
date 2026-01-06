"""
Strategy Metrics - Performance Metrics and Analysis for Agent Evaluation.

This module provides comprehensive metrics collection and analysis for agent performance:
- PerformanceMetrics: Core metrics calculation (success rate, latency, satisfaction)
- RewardCalculator: RL reward signals based on outcomes
- MetricsAggregator: Periodic aggregation and reporting
- TrendAnalyzer: Trend detection and anomaly identification
- AdjustmentTrigger: Automatic adjustment recommendations

Strategic Intent:
Continuous improvement requires quantifiable metrics. This module collects and analyzes
agent performance data to inform A/B testing, model training, and strategy adjustments.
The reward calculator provides signals for reinforcement learning loops.

Patterns adapted from:
- brain_layer/07_agent_training/feedback_loop.py (FeedbackLoop, RewardCalculator)
- brain_layer/07_agent_training/reinforcement_learning.py (RLTrainer metrics)
- brain_layer/02_activity_tracing/umi_client.py (tracing patterns)
"""

import asyncio
import logging
import statistics
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================


class MetricType(Enum):
    """Types of metrics that can be tracked."""

    SUCCESS_RATE = "success_rate"
    LATENCY_P50 = "latency_p50"
    LATENCY_P95 = "latency_p95"
    LATENCY_P99 = "latency_p99"
    LATENCY_MEAN = "latency_mean"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    USER_SATISFACTION = "user_satisfaction"
    COST_PER_REQUEST = "cost_per_request"
    TOKEN_USAGE = "token_usage"
    CACHE_HIT_RATE = "cache_hit_rate"
    RETRY_RATE = "retry_rate"
    CUSTOM = "custom"


class TimeWindow(Enum):
    """Time windows for metric aggregation."""

    MINUTE = 60
    FIVE_MINUTES = 300
    FIFTEEN_MINUTES = 900
    HOUR = 3600
    DAY = 86400
    WEEK = 604800


class TrendDirection(Enum):
    """Direction of a trend."""

    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Severity levels for metric alerts."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RewardType(Enum):
    """Types of RL rewards."""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    ERROR = "error"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class MetricDataPoint:
    """A single metric data point."""

    metric_type: MetricType
    value: float
    timestamp: datetime
    agent_id: str
    tenant_id: str | None = None
    correlation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedMetric:
    """Aggregated metric over a time window."""

    metric_type: MetricType
    agent_id: str
    window: TimeWindow
    start_time: datetime
    end_time: datetime

    # Statistics
    count: int = 0
    sum_value: float = 0.0
    min_value: float = float("inf")
    max_value: float = float("-inf")
    mean_value: float = 0.0
    std_dev: float = 0.0
    percentiles: dict[int, float] = field(default_factory=dict)

    # Metadata
    tenant_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_type": self.metric_type.value,
            "agent_id": self.agent_id,
            "window": self.window.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "count": self.count,
            "sum": self.sum_value,
            "min": self.min_value if self.min_value != float("inf") else None,
            "max": self.max_value if self.max_value != float("-inf") else None,
            "mean": self.mean_value,
            "std_dev": self.std_dev,
            "percentiles": self.percentiles,
            "tenant_id": self.tenant_id,
            "metadata": self.metadata,
        }


@dataclass
class TrendAnalysis:
    """Result of trend analysis."""

    metric_type: MetricType
    agent_id: str
    direction: TrendDirection
    slope: float  # Rate of change
    r_squared: float  # Correlation strength
    forecast_value: float | None = None
    confidence: float = 0.0
    anomalies: list[datetime] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_type": self.metric_type.value,
            "agent_id": self.agent_id,
            "direction": self.direction.value,
            "slope": self.slope,
            "r_squared": self.r_squared,
            "forecast_value": self.forecast_value,
            "confidence": self.confidence,
            "anomaly_count": len(self.anomalies),
        }


@dataclass
class Alert:
    """A metric alert."""

    id: str
    severity: AlertSeverity
    metric_type: MetricType
    agent_id: str
    message: str
    current_value: float
    threshold_value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    resolved: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "severity": self.severity.value,
            "metric_type": self.metric_type.value,
            "agent_id": self.agent_id,
            "message": self.message,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
        }


@dataclass
class Reward:
    """An RL reward signal."""

    reward_type: RewardType
    value: float
    agent_id: str
    action_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "reward_type": self.reward_type.value,
            "value": self.value,
            "agent_id": self.agent_id,
            "action_id": self.action_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class AdjustmentRecommendation:
    """Recommendation for agent adjustment."""

    agent_id: str
    recommendation_type: str
    description: str
    confidence: float
    suggested_changes: dict[str, Any]
    expected_improvement: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "recommendation_type": self.recommendation_type,
            "description": self.description,
            "confidence": self.confidence,
            "suggested_changes": self.suggested_changes,
            "expected_improvement": self.expected_improvement,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class StrategyMetricsConfig:
    """Configuration for StrategyMetrics."""

    # Aggregation
    aggregation_windows: list[TimeWindow] = field(
        default_factory=lambda: [TimeWindow.MINUTE, TimeWindow.HOUR, TimeWindow.DAY]
    )
    retention_hours: int = 168  # 7 days

    # Thresholds
    success_rate_warning: float = 0.9
    success_rate_critical: float = 0.8
    latency_warning_ms: float = 1000.0
    latency_critical_ms: float = 5000.0
    error_rate_warning: float = 0.05
    error_rate_critical: float = 0.1

    # Trend analysis
    trend_min_samples: int = 10
    trend_anomaly_std_threshold: float = 3.0

    # Rewards
    success_reward: float = 1.0
    failure_reward: float = 0.0
    partial_reward: float = 0.5
    timeout_penalty: float = -0.5
    error_penalty: float = -1.0

    # Adjustments
    adjustment_confidence_threshold: float = 0.8
    min_samples_for_adjustment: int = 100


# =============================================================================
# Metric Collector
# =============================================================================


class MetricCollector:
    """
    Collects and stores raw metric data points.

    Provides efficient in-memory storage with automatic cleanup.
    """

    def __init__(self, config: StrategyMetricsConfig):
        """
        Initialize metric collector.

        Args:
            config: Configuration settings
        """
        self.config = config
        self._data_points: dict[str, list[MetricDataPoint]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None

    async def record(self, data_point: MetricDataPoint) -> None:
        """
        Record a metric data point.

        Args:
            data_point: The metric data point to record
        """
        key = f"{data_point.agent_id}:{data_point.metric_type.value}"

        async with self._lock:
            self._data_points[key].append(data_point)

    async def record_batch(self, data_points: list[MetricDataPoint]) -> None:
        """
        Record multiple data points.

        Args:
            data_points: List of data points to record
        """
        async with self._lock:
            for dp in data_points:
                key = f"{dp.agent_id}:{dp.metric_type.value}"
                self._data_points[key].append(dp)

    async def get_data_points(
        self,
        agent_id: str,
        metric_type: MetricType,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[MetricDataPoint]:
        """
        Retrieve data points for a metric.

        Args:
            agent_id: Agent identifier
            metric_type: Type of metric
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            List of matching data points
        """
        key = f"{agent_id}:{metric_type.value}"

        async with self._lock:
            points = self._data_points.get(key, [])

            if start_time or end_time:
                filtered = []
                for p in points:
                    if start_time and p.timestamp < start_time:
                        continue
                    if end_time and p.timestamp > end_time:
                        continue
                    filtered.append(p)
                return filtered

            return list(points)

    async def cleanup_old_data(self) -> int:
        """
        Remove data points older than retention period.

        Returns:
            Number of data points removed
        """
        cutoff = datetime.utcnow() - timedelta(hours=self.config.retention_hours)
        removed = 0

        async with self._lock:
            for key in list(self._data_points.keys()):
                original_count = len(self._data_points[key])
                self._data_points[key] = [
                    p for p in self._data_points[key] if p.timestamp >= cutoff
                ]
                removed += original_count - len(self._data_points[key])

                # Remove empty keys
                if not self._data_points[key]:
                    del self._data_points[key]

        logger.debug(f"Cleaned up {removed} old metric data points")
        return removed

    async def start_cleanup_loop(self, interval_seconds: int = 3600) -> None:
        """Start background cleanup task."""

        async def cleanup_loop():
            while True:
                await asyncio.sleep(interval_seconds)
                await self.cleanup_old_data()

        self._cleanup_task = asyncio.create_task(cleanup_loop())

    async def stop_cleanup_loop(self) -> None:
        """Stop background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass


# =============================================================================
# Metrics Aggregator
# =============================================================================


class MetricsAggregator:
    """
    Aggregates raw metrics into statistical summaries.

    Computes aggregations over configurable time windows.
    """

    def __init__(self, collector: MetricCollector, config: StrategyMetricsConfig):
        """
        Initialize aggregator.

        Args:
            collector: Metric collector for raw data
            config: Configuration settings
        """
        self.collector = collector
        self.config = config
        self._aggregations: dict[str, AggregatedMetric] = {}

    async def aggregate(
        self,
        agent_id: str,
        metric_type: MetricType,
        window: TimeWindow,
        end_time: datetime | None = None,
    ) -> AggregatedMetric:
        """
        Aggregate metrics over a time window.

        Args:
            agent_id: Agent identifier
            metric_type: Type of metric
            window: Time window for aggregation
            end_time: End time (defaults to now)

        Returns:
            Aggregated metric statistics
        """
        if end_time is None:
            end_time = datetime.utcnow()
        start_time = end_time - timedelta(seconds=window.value)

        # Get raw data points
        data_points = await self.collector.get_data_points(
            agent_id=agent_id,
            metric_type=metric_type,
            start_time=start_time,
            end_time=end_time,
        )

        # Compute statistics
        values = [dp.value for dp in data_points]

        aggregation = AggregatedMetric(
            metric_type=metric_type,
            agent_id=agent_id,
            window=window,
            start_time=start_time,
            end_time=end_time,
            count=len(values),
        )

        if values:
            aggregation.sum_value = sum(values)
            aggregation.min_value = min(values)
            aggregation.max_value = max(values)
            aggregation.mean_value = statistics.mean(values)

            if len(values) >= 2:
                aggregation.std_dev = statistics.stdev(values)

            # Compute percentiles
            sorted_values = sorted(values)
            for p in [50, 90, 95, 99]:
                idx = int(len(sorted_values) * p / 100)
                aggregation.percentiles[p] = sorted_values[min(idx, len(sorted_values) - 1)]

        # Cache aggregation
        cache_key = f"{agent_id}:{metric_type.value}:{window.value}"
        self._aggregations[cache_key] = aggregation

        return aggregation

    async def aggregate_all_windows(
        self,
        agent_id: str,
        metric_type: MetricType,
    ) -> dict[TimeWindow, AggregatedMetric]:
        """
        Aggregate metrics across all configured windows.

        Args:
            agent_id: Agent identifier
            metric_type: Type of metric

        Returns:
            Dictionary of window to aggregation
        """
        results = {}
        for window in self.config.aggregation_windows:
            results[window] = await self.aggregate(agent_id, metric_type, window)
        return results

    def get_cached_aggregation(
        self,
        agent_id: str,
        metric_type: MetricType,
        window: TimeWindow,
    ) -> AggregatedMetric | None:
        """Get cached aggregation if available."""
        cache_key = f"{agent_id}:{metric_type.value}:{window.value}"
        return self._aggregations.get(cache_key)


# =============================================================================
# Trend Analyzer
# =============================================================================


class TrendAnalyzer:
    """
    Analyzes metric trends and detects anomalies.

    Uses linear regression for trend detection and z-score for anomalies.
    """

    def __init__(self, collector: MetricCollector, config: StrategyMetricsConfig):
        """
        Initialize trend analyzer.

        Args:
            collector: Metric collector for data
            config: Configuration settings
        """
        self.collector = collector
        self.config = config

    async def analyze_trend(
        self,
        agent_id: str,
        metric_type: MetricType,
        window: TimeWindow = TimeWindow.HOUR,
    ) -> TrendAnalysis:
        """
        Analyze trend for a metric.

        Args:
            agent_id: Agent identifier
            metric_type: Type of metric
            window: Time window to analyze

        Returns:
            Trend analysis result
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(seconds=window.value)

        data_points = await self.collector.get_data_points(
            agent_id=agent_id,
            metric_type=metric_type,
            start_time=start_time,
            end_time=end_time,
        )

        if len(data_points) < self.config.trend_min_samples:
            return TrendAnalysis(
                metric_type=metric_type,
                agent_id=agent_id,
                direction=TrendDirection.UNKNOWN,
                slope=0.0,
                r_squared=0.0,
                confidence=0.0,
            )

        # Convert to x (time) and y (value) arrays
        base_time = data_points[0].timestamp
        x_values = [(dp.timestamp - base_time).total_seconds() for dp in data_points]
        y_values = [dp.value for dp in data_points]

        # Linear regression
        slope, intercept, r_squared = self._linear_regression(x_values, y_values)

        # Determine direction
        if r_squared >= 0.5:  # Strong correlation
            if slope > 0:
                # For latency, increasing is declining; for success rate, increasing is improving
                if metric_type in [
                    MetricType.SUCCESS_RATE,
                    MetricType.USER_SATISFACTION,
                    MetricType.THROUGHPUT,
                    MetricType.CACHE_HIT_RATE,
                ]:
                    direction = TrendDirection.IMPROVING
                else:
                    direction = TrendDirection.DECLINING
            elif slope < 0:
                if metric_type in [
                    MetricType.SUCCESS_RATE,
                    MetricType.USER_SATISFACTION,
                    MetricType.THROUGHPUT,
                    MetricType.CACHE_HIT_RATE,
                ]:
                    direction = TrendDirection.DECLINING
                else:
                    direction = TrendDirection.IMPROVING
            else:
                direction = TrendDirection.STABLE
        else:
            direction = TrendDirection.STABLE

        # Detect anomalies using z-score
        anomalies = self._detect_anomalies(data_points)

        # Forecast next value
        forecast_x = x_values[-1] + (x_values[-1] - x_values[0]) / len(x_values)
        forecast_value = slope * forecast_x + intercept

        return TrendAnalysis(
            metric_type=metric_type,
            agent_id=agent_id,
            direction=direction,
            slope=slope,
            r_squared=r_squared,
            forecast_value=forecast_value,
            confidence=r_squared,
            anomalies=anomalies,
        )

    def _linear_regression(
        self,
        x: list[float],
        y: list[float],
    ) -> tuple[float, float, float]:
        """
        Compute linear regression.

        Returns:
            Tuple of (slope, intercept, r_squared)
        """
        n = len(x)
        if n < 2:
            return 0.0, 0.0, 0.0

        x_mean = sum(x) / n
        y_mean = sum(y) / n

        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0, y_mean, 0.0

        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        # Calculate R-squared
        y_pred = [slope * x[i] + intercept for i in range(n)]
        ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((y[i] - y_mean) ** 2 for i in range(n))

        if ss_tot == 0:
            r_squared = 1.0 if ss_res == 0 else 0.0
        else:
            r_squared = 1 - (ss_res / ss_tot)

        return slope, intercept, max(0.0, r_squared)

    def _detect_anomalies(
        self,
        data_points: list[MetricDataPoint],
    ) -> list[datetime]:
        """
        Detect anomalies using z-score.

        Returns:
            List of timestamps where anomalies occurred
        """
        if len(data_points) < 3:
            return []

        values = [dp.value for dp in data_points]
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) >= 2 else 0

        if std_val == 0:
            return []

        anomalies = []
        for dp in data_points:
            z_score = abs(dp.value - mean_val) / std_val
            if z_score > self.config.trend_anomaly_std_threshold:
                anomalies.append(dp.timestamp)

        return anomalies


# =============================================================================
# Reward Calculator
# =============================================================================


class RewardCalculator:
    """
    Calculates RL rewards based on agent outcomes.

    Provides standardized reward signals for reinforcement learning.
    """

    def __init__(self, config: StrategyMetricsConfig):
        """
        Initialize reward calculator.

        Args:
            config: Configuration settings
        """
        self.config = config
        self._reward_history: list[Reward] = []

    def calculate_reward(
        self,
        agent_id: str,
        action_id: str,
        success: bool,
        score: float | None = None,
        latency_ms: float | None = None,
        error: str | None = None,
        timeout: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Reward:
        """
        Calculate reward for an agent action.

        Args:
            agent_id: Agent identifier
            action_id: Unique action identifier
            success: Whether action succeeded
            score: Optional quality score (0-1)
            latency_ms: Optional latency in milliseconds
            error: Optional error message
            timeout: Whether action timed out
            metadata: Optional additional metadata

        Returns:
            Calculated reward
        """
        if timeout:
            reward_type = RewardType.TIMEOUT
            value = self.config.timeout_penalty
        elif error:
            reward_type = RewardType.ERROR
            value = self.config.error_penalty
        elif success:
            reward_type = RewardType.SUCCESS
            value = self.config.success_reward

            # Adjust based on score if provided
            if score is not None:
                value *= score

            # Adjust based on latency if provided
            if latency_ms is not None:
                # Penalize slow responses (up to 20% reduction)
                latency_factor = min(1.0, self.config.latency_warning_ms / max(latency_ms, 1))
                value *= 0.8 + 0.2 * latency_factor
        else:
            # Failure but no error/timeout
            if score is not None and score > 0:
                reward_type = RewardType.PARTIAL
                value = self.config.partial_reward * score
            else:
                reward_type = RewardType.FAILURE
                value = self.config.failure_reward

        reward = Reward(
            reward_type=reward_type,
            value=value,
            agent_id=agent_id,
            action_id=action_id,
            metadata=metadata or {},
        )

        self._reward_history.append(reward)
        return reward

    def get_cumulative_reward(
        self,
        agent_id: str,
        since: datetime | None = None,
    ) -> float:
        """
        Get cumulative reward for an agent.

        Args:
            agent_id: Agent identifier
            since: Optional start time

        Returns:
            Sum of rewards
        """
        total = 0.0
        for reward in self._reward_history:
            if reward.agent_id != agent_id:
                continue
            if since and reward.timestamp < since:
                continue
            total += reward.value
        return total

    def get_average_reward(
        self,
        agent_id: str,
        since: datetime | None = None,
    ) -> float:
        """
        Get average reward for an agent.

        Args:
            agent_id: Agent identifier
            since: Optional start time

        Returns:
            Average reward value
        """
        rewards = []
        for reward in self._reward_history:
            if reward.agent_id != agent_id:
                continue
            if since and reward.timestamp < since:
                continue
            rewards.append(reward.value)

        if not rewards:
            return 0.0
        return sum(rewards) / len(rewards)

    def get_reward_distribution(
        self,
        agent_id: str,
    ) -> dict[RewardType, int]:
        """
        Get distribution of reward types.

        Returns:
            Count of each reward type
        """
        distribution: dict[RewardType, int] = defaultdict(int)
        for reward in self._reward_history:
            if reward.agent_id == agent_id:
                distribution[reward.reward_type] += 1
        return dict(distribution)


# =============================================================================
# Alert Manager
# =============================================================================


class AlertManager:
    """
    Manages metric alerts and thresholds.

    Triggers alerts when metrics cross configured thresholds.
    """

    def __init__(self, config: StrategyMetricsConfig):
        """
        Initialize alert manager.

        Args:
            config: Configuration settings
        """
        self.config = config
        self._alerts: list[Alert] = []
        self._alert_callbacks: list[Callable[[Alert], None]] = []

    def register_callback(self, callback: Callable[[Alert], None]) -> None:
        """Register alert callback."""
        self._alert_callbacks.append(callback)

    def check_thresholds(
        self,
        agent_id: str,
        metric_type: MetricType,
        value: float,
    ) -> Alert | None:
        """
        Check if a metric value crosses thresholds.

        Args:
            agent_id: Agent identifier
            metric_type: Type of metric
            value: Current value

        Returns:
            Alert if threshold crossed, None otherwise
        """
        alert = None

        if metric_type == MetricType.SUCCESS_RATE:
            if value < self.config.success_rate_critical:
                alert = self._create_alert(
                    AlertSeverity.CRITICAL,
                    metric_type,
                    agent_id,
                    f"Success rate critically low: {value:.2%}",
                    value,
                    self.config.success_rate_critical,
                )
            elif value < self.config.success_rate_warning:
                alert = self._create_alert(
                    AlertSeverity.WARNING,
                    metric_type,
                    agent_id,
                    f"Success rate below warning threshold: {value:.2%}",
                    value,
                    self.config.success_rate_warning,
                )

        elif metric_type in [MetricType.LATENCY_MEAN, MetricType.LATENCY_P95]:
            if value > self.config.latency_critical_ms:
                alert = self._create_alert(
                    AlertSeverity.CRITICAL,
                    metric_type,
                    agent_id,
                    f"Latency critically high: {value:.0f}ms",
                    value,
                    self.config.latency_critical_ms,
                )
            elif value > self.config.latency_warning_ms:
                alert = self._create_alert(
                    AlertSeverity.WARNING,
                    metric_type,
                    agent_id,
                    f"Latency above warning threshold: {value:.0f}ms",
                    value,
                    self.config.latency_warning_ms,
                )

        elif metric_type == MetricType.ERROR_RATE:
            if value > self.config.error_rate_critical:
                alert = self._create_alert(
                    AlertSeverity.CRITICAL,
                    metric_type,
                    agent_id,
                    f"Error rate critically high: {value:.2%}",
                    value,
                    self.config.error_rate_critical,
                )
            elif value > self.config.error_rate_warning:
                alert = self._create_alert(
                    AlertSeverity.WARNING,
                    metric_type,
                    agent_id,
                    f"Error rate above warning threshold: {value:.2%}",
                    value,
                    self.config.error_rate_warning,
                )

        if alert:
            self._alerts.append(alert)
            for callback in self._alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")

        return alert

    def _create_alert(
        self,
        severity: AlertSeverity,
        metric_type: MetricType,
        agent_id: str,
        message: str,
        current_value: float,
        threshold_value: float,
    ) -> Alert:
        """Create an alert."""
        return Alert(
            id=f"alert-{uuid.uuid4().hex[:12]}",
            severity=severity,
            metric_type=metric_type,
            agent_id=agent_id,
            message=message,
            current_value=current_value,
            threshold_value=threshold_value,
        )

    def get_active_alerts(
        self,
        agent_id: str | None = None,
        severity: AlertSeverity | None = None,
    ) -> list[Alert]:
        """Get active (unresolved) alerts."""
        alerts = [a for a in self._alerts if not a.resolved]

        if agent_id:
            alerts = [a for a in alerts if a.agent_id == agent_id]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.resolved = True
                return True
        return False


# =============================================================================
# Adjustment Trigger
# =============================================================================


class AdjustmentTrigger:
    """
    Generates recommendations for agent adjustments.

    Analyzes metrics to suggest configuration changes.
    """

    def __init__(
        self,
        aggregator: MetricsAggregator,
        trend_analyzer: TrendAnalyzer,
        config: StrategyMetricsConfig,
    ):
        """
        Initialize adjustment trigger.

        Args:
            aggregator: Metrics aggregator
            trend_analyzer: Trend analyzer
            config: Configuration settings
        """
        self.aggregator = aggregator
        self.trend_analyzer = trend_analyzer
        self.config = config

    async def analyze_and_recommend(
        self,
        agent_id: str,
    ) -> list[AdjustmentRecommendation]:
        """
        Analyze metrics and generate recommendations.

        Args:
            agent_id: Agent identifier

        Returns:
            List of recommendations
        """
        recommendations = []

        # Check success rate
        success_agg = await self.aggregator.aggregate(
            agent_id, MetricType.SUCCESS_RATE, TimeWindow.HOUR
        )

        if success_agg.count >= self.config.min_samples_for_adjustment:
            success_trend = await self.trend_analyzer.analyze_trend(
                agent_id, MetricType.SUCCESS_RATE, TimeWindow.HOUR
            )

            if success_trend.direction == TrendDirection.DECLINING:
                recommendations.append(
                    AdjustmentRecommendation(
                        agent_id=agent_id,
                        recommendation_type="model_upgrade",
                        description="Success rate declining. Consider upgrading to a more capable model.",
                        confidence=success_trend.confidence,
                        suggested_changes={
                            "action": "upgrade_model",
                            "current_success_rate": success_agg.mean_value,
                            "trend_slope": success_trend.slope,
                        },
                        expected_improvement=0.1,
                    )
                )

        # Check latency
        latency_agg = await self.aggregator.aggregate(
            agent_id, MetricType.LATENCY_MEAN, TimeWindow.HOUR
        )

        if latency_agg.count >= self.config.min_samples_for_adjustment:
            if latency_agg.mean_value > self.config.latency_warning_ms:
                recommendations.append(
                    AdjustmentRecommendation(
                        agent_id=agent_id,
                        recommendation_type="performance_optimization",
                        description="High latency detected. Consider caching or model optimization.",
                        confidence=0.8,
                        suggested_changes={
                            "action": "enable_caching",
                            "current_latency_ms": latency_agg.mean_value,
                            "target_latency_ms": self.config.latency_warning_ms * 0.8,
                        },
                        expected_improvement=0.3,
                    )
                )

        # Check error rate
        error_agg = await self.aggregator.aggregate(
            agent_id, MetricType.ERROR_RATE, TimeWindow.HOUR
        )

        if error_agg.count >= self.config.min_samples_for_adjustment:
            if error_agg.mean_value > self.config.error_rate_warning:
                recommendations.append(
                    AdjustmentRecommendation(
                        agent_id=agent_id,
                        recommendation_type="error_reduction",
                        description="High error rate detected. Review error patterns and add retry logic.",
                        confidence=0.9,
                        suggested_changes={
                            "action": "increase_retries",
                            "current_error_rate": error_agg.mean_value,
                            "suggested_max_retries": 5,
                        },
                        expected_improvement=0.2,
                    )
                )

        return recommendations


# =============================================================================
# Strategy Metrics (Main Interface)
# =============================================================================


class StrategyMetrics:
    """
    Main interface for strategy metrics and analysis.

    Combines all metric collection, analysis, and recommendation components.

    Example:
        >>> config = StrategyMetricsConfig()
        >>> metrics = StrategyMetrics(config)
        >>> await metrics.record_outcome(
        ...     agent_id="summarizer",
        ...     success=True,
        ...     latency_ms=150.0,
        ...     score=0.92,
        ... )
        >>> report = await metrics.get_agent_report("summarizer")
    """

    def __init__(self, config: StrategyMetricsConfig | None = None):
        """
        Initialize strategy metrics.

        Args:
            config: Optional configuration (uses defaults if not provided)
        """
        self.config = config or StrategyMetricsConfig()

        # Initialize components
        self.collector = MetricCollector(self.config)
        self.aggregator = MetricsAggregator(self.collector, self.config)
        self.trend_analyzer = TrendAnalyzer(self.collector, self.config)
        self.reward_calculator = RewardCalculator(self.config)
        self.alert_manager = AlertManager(self.config)
        self.adjustment_trigger = AdjustmentTrigger(
            self.aggregator, self.trend_analyzer, self.config
        )

        # Statistics
        self._total_outcomes = 0
        self._successful_outcomes = 0

    async def record_outcome(
        self,
        agent_id: str,
        success: bool,
        latency_ms: float,
        score: float | None = None,
        error: str | None = None,
        timeout: bool = False,
        tenant_id: str | None = None,
        correlation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Reward:
        """
        Record an agent outcome.

        Args:
            agent_id: Agent identifier
            success: Whether the operation succeeded
            latency_ms: Latency in milliseconds
            score: Optional quality score (0-1)
            error: Optional error message
            timeout: Whether operation timed out
            tenant_id: Optional tenant identifier
            correlation_id: Optional correlation ID
            metadata: Optional additional metadata

        Returns:
            Calculated reward for the outcome
        """
        timestamp = datetime.utcnow()
        action_id = f"action-{uuid.uuid4().hex[:12]}"

        # Update statistics
        self._total_outcomes += 1
        if success:
            self._successful_outcomes += 1

        # Record individual metrics
        data_points = [
            MetricDataPoint(
                metric_type=MetricType.SUCCESS_RATE,
                value=1.0 if success else 0.0,
                timestamp=timestamp,
                agent_id=agent_id,
                tenant_id=tenant_id,
                correlation_id=correlation_id,
            ),
            MetricDataPoint(
                metric_type=MetricType.LATENCY_MEAN,
                value=latency_ms,
                timestamp=timestamp,
                agent_id=agent_id,
                tenant_id=tenant_id,
                correlation_id=correlation_id,
            ),
        ]

        if error:
            data_points.append(
                MetricDataPoint(
                    metric_type=MetricType.ERROR_RATE,
                    value=1.0,
                    timestamp=timestamp,
                    agent_id=agent_id,
                    tenant_id=tenant_id,
                    correlation_id=correlation_id,
                )
            )

        if score is not None:
            data_points.append(
                MetricDataPoint(
                    metric_type=MetricType.USER_SATISFACTION,
                    value=score,
                    timestamp=timestamp,
                    agent_id=agent_id,
                    tenant_id=tenant_id,
                    correlation_id=correlation_id,
                )
            )

        await self.collector.record_batch(data_points)

        # Calculate reward
        reward = self.reward_calculator.calculate_reward(
            agent_id=agent_id,
            action_id=action_id,
            success=success,
            score=score,
            latency_ms=latency_ms,
            error=error,
            timeout=timeout,
            metadata=metadata,
        )

        # Check thresholds (using rolling average)
        agg = await self.aggregator.aggregate(
            agent_id, MetricType.SUCCESS_RATE, TimeWindow.FIVE_MINUTES
        )
        if agg.count >= 10:
            self.alert_manager.check_thresholds(agent_id, MetricType.SUCCESS_RATE, agg.mean_value)

        return reward

    async def get_agent_report(
        self,
        agent_id: str,
        window: TimeWindow = TimeWindow.HOUR,
    ) -> dict[str, Any]:
        """
        Get comprehensive report for an agent.

        Args:
            agent_id: Agent identifier
            window: Time window for analysis

        Returns:
            Report dictionary with metrics, trends, and recommendations
        """
        # Aggregate all metric types
        aggregations = {}
        for metric_type in [
            MetricType.SUCCESS_RATE,
            MetricType.LATENCY_MEAN,
            MetricType.ERROR_RATE,
            MetricType.USER_SATISFACTION,
        ]:
            try:
                agg = await self.aggregator.aggregate(agent_id, metric_type, window)
                if agg.count > 0:
                    aggregations[metric_type.value] = agg.to_dict()
            except Exception:
                pass

        # Analyze trends
        trends = {}
        for metric_type in [MetricType.SUCCESS_RATE, MetricType.LATENCY_MEAN]:
            try:
                trend = await self.trend_analyzer.analyze_trend(agent_id, metric_type, window)
                if trend.direction != TrendDirection.UNKNOWN:
                    trends[metric_type.value] = trend.to_dict()
            except Exception:
                pass

        # Get rewards
        reward_dist = self.reward_calculator.get_reward_distribution(agent_id)
        avg_reward = self.reward_calculator.get_average_reward(agent_id)

        # Get active alerts
        alerts = [a.to_dict() for a in self.alert_manager.get_active_alerts(agent_id)]

        # Get recommendations
        try:
            recommendations = await self.adjustment_trigger.analyze_and_recommend(agent_id)
            recommendations = [r.to_dict() for r in recommendations]
        except Exception:
            recommendations = []

        return {
            "agent_id": agent_id,
            "window": window.value,
            "generated_at": datetime.utcnow().isoformat(),
            "aggregations": aggregations,
            "trends": trends,
            "rewards": {
                "average": avg_reward,
                "distribution": {k.value: v for k, v in reward_dist.items()},
            },
            "alerts": alerts,
            "recommendations": recommendations,
        }

    async def get_system_summary(self) -> dict[str, Any]:
        """Get summary of overall system metrics."""
        success_rate = (
            self._successful_outcomes / self._total_outcomes if self._total_outcomes > 0 else 0.0
        )

        return {
            "total_outcomes": self._total_outcomes,
            "successful_outcomes": self._successful_outcomes,
            "overall_success_rate": success_rate,
            "active_alerts": len(self.alert_manager.get_active_alerts()),
            "critical_alerts": len(
                self.alert_manager.get_active_alerts(severity=AlertSeverity.CRITICAL)
            ),
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get current statistics."""
        return {
            "total_outcomes_recorded": self._total_outcomes,
            "successful_outcomes": self._successful_outcomes,
            "overall_success_rate": (
                self._successful_outcomes / self._total_outcomes
                if self._total_outcomes > 0
                else 0.0
            ),
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_strategy_metrics(
    success_rate_warning: float = 0.9,
    success_rate_critical: float = 0.8,
    latency_warning_ms: float = 1000.0,
    latency_critical_ms: float = 5000.0,
    **kwargs,
) -> StrategyMetrics:
    """
    Factory function to create StrategyMetrics with custom thresholds.

    Args:
        success_rate_warning: Warning threshold for success rate
        success_rate_critical: Critical threshold for success rate
        latency_warning_ms: Warning threshold for latency
        latency_critical_ms: Critical threshold for latency
        **kwargs: Additional config parameters

    Returns:
        Configured StrategyMetrics instance
    """
    config = StrategyMetricsConfig(
        success_rate_warning=success_rate_warning,
        success_rate_critical=success_rate_critical,
        latency_warning_ms=latency_warning_ms,
        latency_critical_ms=latency_critical_ms,
        **kwargs,
    )
    return StrategyMetrics(config)


# =============================================================================
# Singleton for Global Access
# =============================================================================

_global_metrics: StrategyMetrics | None = None


def get_strategy_metrics(
    config: StrategyMetricsConfig | None = None,
) -> StrategyMetrics:
    """
    Get or create global StrategyMetrics instance.

    Args:
        config: Optional configuration for first creation

    Returns:
        Global StrategyMetrics instance
    """
    global _global_metrics

    if _global_metrics is None:
        _global_metrics = StrategyMetrics(config)

    return _global_metrics


def reset_global_metrics() -> None:
    """Reset global metrics instance (for testing)."""
    global _global_metrics
    _global_metrics = None
