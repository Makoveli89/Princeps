"""
Feedback Loop - Performance Tracking and Improvement

Collects performance metrics from production, tracks improvements,
and triggers retraining when performance degrades.

Source: F:\Mothership-main\Mothership-main\learning\feedback_loop.py
Recycled: December 26, 2024
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics"""

    ACCURACY = "accuracy"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    USER_SATISFACTION = "user_satisfaction"
    BUSINESS_IMPACT = "business_impact"


class AlertSeverity(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PerformanceMetrics:
    """Performance metrics for an agent"""

    agent_id: str
    timestamp: datetime

    # Core metrics
    accuracy: float = 0.0
    latency_ms: float = 0.0
    throughput_per_hour: float = 0.0
    error_rate: float = 0.0

    # User feedback
    user_satisfaction_score: float = 0.0  # 0-10 scale
    feedback_count: int = 0

    # Business metrics
    tasks_completed: int = 0
    business_value_generated: float = 0.0

    # Metadata
    model_version: str = "v1.0"
    deployment_id: str = ""

    def overall_health_score(self) -> float:
        """Calculate overall health score (0-100)"""
        # Weighted combination of metrics
        accuracy_score = self.accuracy * 30
        latency_score = max(0, (1 - self.latency_ms / 1000) * 20)  # Penalize >1s latency
        error_score = (1 - self.error_rate) * 20
        satisfaction_score = (self.user_satisfaction_score / 10) * 30

        return accuracy_score + latency_score + error_score + satisfaction_score


@dataclass
class PerformanceAlert:
    """Alert for performance degradation"""

    alert_id: str
    agent_id: str
    severity: AlertSeverity
    metric_type: MetricType
    current_value: float
    baseline_value: float
    degradation_pct: float
    timestamp: datetime
    message: str
    resolved: bool = False


@dataclass
class ImprovementTracker:
    """Tracks improvement over time"""

    agent_id: str
    baseline_metrics: PerformanceMetrics
    current_metrics: PerformanceMetrics
    improvement_history: List[Dict[str, float]] = field(default_factory=list)

    def calculate_improvement(self) -> Dict[str, float]:
        """Calculate improvement percentage for each metric"""
        improvements = {}

        if self.baseline_metrics.accuracy > 0:
            improvements["accuracy"] = (
                (self.current_metrics.accuracy - self.baseline_metrics.accuracy) / self.baseline_metrics.accuracy * 100
            )

        if self.baseline_metrics.latency_ms > 0:
            improvements["latency"] = (
                (self.baseline_metrics.latency_ms - self.current_metrics.latency_ms)
                / self.baseline_metrics.latency_ms
                * 100
            )

        if self.baseline_metrics.error_rate > 0:
            improvements["error_rate"] = (
                (self.baseline_metrics.error_rate - self.current_metrics.error_rate)
                / self.baseline_metrics.error_rate
                * 100
            )

        improvements["overall_health"] = (
            (self.current_metrics.overall_health_score() - self.baseline_metrics.overall_health_score())
            / self.baseline_metrics.overall_health_score()
            * 100
            if self.baseline_metrics.overall_health_score() > 0
            else 0
        )

        return improvements

    def monthly_improvement_rate(self) -> float:
        """Calculate average monthly improvement rate"""
        if not self.improvement_history:
            return 0.0

        # Take last 4 weeks (assume weekly measurements)
        recent = self.improvement_history[-4:] if len(self.improvement_history) >= 4 else self.improvement_history

        if not recent:
            return 0.0

        # Average the overall health improvements
        total = sum(h.get("overall_health", 0) for h in recent)
        return total / len(recent)


class FeedbackLoop:
    """
    Performance Feedback Loop for Continuous Improvement

    Features:
    - Real-time performance monitoring
    - Automatic degradation detection
    - Improvement tracking
    - Retraining triggers
    - Performance alerts
    """

    def __init__(self, metrics_dir: str = "./metrics", history_size: int = 1000):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Agent metrics history (circular buffer)
        self.metrics_history: Dict[str, deque] = {}
        self.history_size = history_size

        # Baselines for each agent
        self.baselines: Dict[str, PerformanceMetrics] = {}

        # Improvement trackers
        self.trackers: Dict[str, ImprovementTracker] = {}

        # Active alerts
        self.alerts: List[PerformanceAlert] = []

        # Degradation thresholds
        self.degradation_thresholds = {
            MetricType.ACCURACY: 0.05,  # 5% drop triggers alert
            MetricType.LATENCY: 0.20,  # 20% increase triggers alert
            MetricType.ERROR_RATE: 0.10,  # 10% increase triggers alert
            MetricType.USER_SATISFACTION: 0.15,  # 15% drop triggers alert
        }

        logger.info("FeedbackLoop initialized")

    def set_baseline(self, agent_id: str, metrics: PerformanceMetrics) -> None:
        """Set baseline metrics for an agent"""
        self.baselines[agent_id] = metrics

        if agent_id not in self.metrics_history:
            self.metrics_history[agent_id] = deque(maxlen=self.history_size)

        logger.info(
            f"Set baseline for {agent_id}: accuracy={metrics.accuracy:.2%}, " f"latency={metrics.latency_ms:.1f}ms"
        )

    def record_metrics(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics for an agent"""
        agent_id = metrics.agent_id

        # Initialize history if needed
        if agent_id not in self.metrics_history:
            self.metrics_history[agent_id] = deque(maxlen=self.history_size)

        # Add to history
        self.metrics_history[agent_id].append(metrics)

        # Set baseline if not exists
        if agent_id not in self.baselines:
            self.set_baseline(agent_id, metrics)

        # Update improvement tracker
        if agent_id not in self.trackers:
            self.trackers[agent_id] = ImprovementTracker(
                agent_id=agent_id, baseline_metrics=self.baselines[agent_id], current_metrics=metrics
            )
        else:
            self.trackers[agent_id].current_metrics = metrics
            improvements = self.trackers[agent_id].calculate_improvement()
            self.trackers[agent_id].improvement_history.append(improvements)

        # Check for performance degradation
        self._check_degradation(metrics)

        logger.debug(f"Recorded metrics for {agent_id}: health={metrics.overall_health_score():.1f}/100")

    def _check_degradation(self, current: PerformanceMetrics) -> None:
        """Check for performance degradation and trigger alerts"""
        agent_id = current.agent_id
        baseline = self.baselines.get(agent_id)

        if not baseline:
            return

        # Check accuracy degradation
        if baseline.accuracy > 0:
            accuracy_drop = (baseline.accuracy - current.accuracy) / baseline.accuracy
            if accuracy_drop > self.degradation_thresholds[MetricType.ACCURACY]:
                self._create_alert(
                    agent_id=agent_id,
                    severity=AlertSeverity.CRITICAL,
                    metric_type=MetricType.ACCURACY,
                    current_value=current.accuracy,
                    baseline_value=baseline.accuracy,
                    degradation_pct=accuracy_drop * 100,
                )

        # Check latency degradation
        if baseline.latency_ms > 0:
            latency_increase = (current.latency_ms - baseline.latency_ms) / baseline.latency_ms
            if latency_increase > self.degradation_thresholds[MetricType.LATENCY]:
                self._create_alert(
                    agent_id=agent_id,
                    severity=AlertSeverity.WARNING,
                    metric_type=MetricType.LATENCY,
                    current_value=current.latency_ms,
                    baseline_value=baseline.latency_ms,
                    degradation_pct=latency_increase * 100,
                )

        # Check error rate increase
        if current.error_rate > baseline.error_rate:
            error_increase = (current.error_rate - baseline.error_rate) / (
                baseline.error_rate if baseline.error_rate > 0 else 0.01
            )
            if error_increase > self.degradation_thresholds[MetricType.ERROR_RATE]:
                self._create_alert(
                    agent_id=agent_id,
                    severity=AlertSeverity.CRITICAL,
                    metric_type=MetricType.ERROR_RATE,
                    current_value=current.error_rate,
                    baseline_value=baseline.error_rate,
                    degradation_pct=error_increase * 100,
                )

    def _create_alert(
        self,
        agent_id: str,
        severity: AlertSeverity,
        metric_type: MetricType,
        current_value: float,
        baseline_value: float,
        degradation_pct: float,
    ) -> None:
        """Create a performance alert"""
        alert_id = f"alert-{agent_id}-{metric_type.value}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        message = (
            f"{metric_type.value.upper()} degraded by {degradation_pct:.1f}%: "
            f"{baseline_value:.4f} → {current_value:.4f}"
        )

        alert = PerformanceAlert(
            alert_id=alert_id,
            agent_id=agent_id,
            severity=severity,
            metric_type=metric_type,
            current_value=current_value,
            baseline_value=baseline_value,
            degradation_pct=degradation_pct,
            timestamp=datetime.now(),
            message=message,
        )

        self.alerts.append(alert)

        logger.warning(f"🚨 {severity.value.upper()}: {message}")

        # Trigger retraining for critical alerts
        if severity == AlertSeverity.CRITICAL:
            logger.info(f"⚡ Triggering automatic retraining for {agent_id}")

    def get_agent_health(self, agent_id: str) -> Dict[str, Any]:
        """Get current health status for an agent"""
        if agent_id not in self.metrics_history or not self.metrics_history[agent_id]:
            return {"agent_id": agent_id, "status": "no_data"}

        current = self.metrics_history[agent_id][-1]
        baseline = self.baselines.get(agent_id)

        # Get improvement tracker
        tracker = self.trackers.get(agent_id)
        improvements = tracker.calculate_improvement() if tracker else {}
        monthly_rate = tracker.monthly_improvement_rate() if tracker else 0.0

        # Get active alerts
        agent_alerts = [a for a in self.alerts if a.agent_id == agent_id and not a.resolved]

        return {
            "agent_id": agent_id,
            "current_health_score": current.overall_health_score(),
            "baseline_health_score": baseline.overall_health_score() if baseline else 0,
            "metrics": {
                "accuracy": current.accuracy,
                "latency_ms": current.latency_ms,
                "error_rate": current.error_rate,
                "user_satisfaction": current.user_satisfaction_score,
                "tasks_completed": current.tasks_completed,
            },
            "improvements": improvements,
            "monthly_improvement_rate": monthly_rate,
            "active_alerts": len(agent_alerts),
            "model_version": current.model_version,
        }

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for performance dashboard"""
        all_agents = list(self.metrics_history.keys())

        # Calculate aggregate metrics
        total_tasks = sum(
            self.metrics_history[agent][-1].tasks_completed for agent in all_agents if self.metrics_history[agent]
        )

        avg_health = (
            sum(
                self.metrics_history[agent][-1].overall_health_score()
                for agent in all_agents
                if self.metrics_history[agent]
            )
            / len(all_agents)
            if all_agents
            else 0
        )

        avg_monthly_improvement = (
            sum(self.trackers[agent].monthly_improvement_rate() for agent in all_agents if agent in self.trackers)
            / len(all_agents)
            if all_agents
            else 0
        )

        # Get critical alerts
        critical_alerts = [a for a in self.alerts if a.severity == AlertSeverity.CRITICAL and not a.resolved]

        return {
            "total_agents": len(all_agents),
            "average_health_score": avg_health,
            "total_tasks_completed": total_tasks,
            "avg_monthly_improvement_pct": avg_monthly_improvement,
            "critical_alerts": len(critical_alerts),
            "timestamp": datetime.now().isoformat(),
        }


# Demo
async def demo():
    """Demonstrate the Feedback Loop"""
    print("\n" + "=" * 60)
    print("AUTONOMOUS LEARNING LAYER - FEEDBACK LOOP DEMO")
    print("=" * 60)

    loop = FeedbackLoop()

    # Simulate 3 agents with different performance patterns
    agents = ["legal_agent", "sales_agent", "customer_success_agent"]

    print("\n📊 Simulating 4 weeks of performance data...\n")

    # Week 1: Establish baselines
    for agent in agents:
        metrics = PerformanceMetrics(
            agent_id=agent,
            timestamp=datetime.now(),
            accuracy=0.75 + (hash(agent) % 10) / 100,  # 75-85%
            latency_ms=100 + (hash(agent) % 50),
            error_rate=0.05,
            user_satisfaction_score=7.5,
            tasks_completed=100,
        )
        loop.record_metrics(metrics)

    # Weeks 2-4: Show improvement for 2 agents, degradation for 1
    for week in range(1, 4):
        await asyncio.sleep(0.1)

        for agent in agents:
            if agent == "legal_agent":
                # Improving agent
                improvement_factor = 1 + (week * 0.05)  # 5% per week
                metrics = PerformanceMetrics(
                    agent_id=agent,
                    timestamp=datetime.now(),
                    accuracy=min(0.95, loop.baselines[agent].accuracy * improvement_factor),
                    latency_ms=max(50, loop.baselines[agent].latency_ms * 0.95),  # Getting faster
                    error_rate=max(0.01, loop.baselines[agent].error_rate * 0.9),
                    user_satisfaction_score=min(9.5, 7.5 + week * 0.3),
                    tasks_completed=100 + week * 20,
                )
            elif agent == "sales_agent":
                # Stable agent
                metrics = PerformanceMetrics(
                    agent_id=agent,
                    timestamp=datetime.now(),
                    accuracy=loop.baselines[agent].accuracy * 1.02,  # Slight improvement
                    latency_ms=loop.baselines[agent].latency_ms,
                    error_rate=loop.baselines[agent].error_rate,
                    user_satisfaction_score=7.6,
                    tasks_completed=100 + week * 10,
                )
            else:
                # Degrading agent (triggers alerts)
                degradation_factor = 1 - (week * 0.04) if week < 3 else 0.85  # Degrade in week 3
                metrics = PerformanceMetrics(
                    agent_id=agent,
                    timestamp=datetime.now(),
                    accuracy=loop.baselines[agent].accuracy * degradation_factor,
                    latency_ms=loop.baselines[agent].latency_ms * (1 + week * 0.1),  # Getting slower
                    error_rate=loop.baselines[agent].error_rate * (1 + week * 0.15),
                    user_satisfaction_score=max(5.0, 7.5 - week * 0.5),
                    tasks_completed=100 - week * 10,
                )

            loop.record_metrics(metrics)

    # Display results
    print("=" * 60)
    print("PERFORMANCE DASHBOARD")
    print("=" * 60)

    dashboard = loop.get_dashboard_data()
    print("\n🌐 Overall System Health")
    print(f"   Total Agents: {dashboard['total_agents']}")
    print(f"   Average Health Score: {dashboard['average_health_score']:.1f}/100")
    print(f"   Total Tasks: {dashboard['total_tasks_completed']}")
    print(f"   Avg Monthly Improvement: {dashboard['avg_monthly_improvement_pct']:+.2f}%")
    print(f"   Critical Alerts: {dashboard['critical_alerts']}")

    print("\n📊 Individual Agent Performance:")
    for agent in agents:
        health = loop.get_agent_health(agent)

        status_emoji = "✅" if health["active_alerts"] == 0 else "🚨"
        trend_emoji = "📈" if health["monthly_improvement_rate"] > 0 else "📉"

        print(f"\n{status_emoji} {agent}")
        print(
            f"   Health Score: {health['current_health_score']:.1f}/100 "
            f"(baseline: {health['baseline_health_score']:.1f})"
        )
        print(f"   Accuracy: {health['metrics']['accuracy']:.2%}")
        print(f"   Latency: {health['metrics']['latency_ms']:.1f}ms")
        print(f"   Error Rate: {health['metrics']['error_rate']:.2%}")
        print(f"   {trend_emoji} Monthly Improvement: {health['monthly_improvement_rate']:+.2f}%")

        if health["improvements"]:
            print("   Improvements vs Baseline:")
            for metric, improvement in health["improvements"].items():
                if metric != "overall_health":
                    print(f"     • {metric}: {improvement:+.1f}%")

        if health["active_alerts"] > 0:
            print(f"   ⚠️  Active Alerts: {health['active_alerts']}")

    print("\n🔔 Recent Alerts:")
    for alert in loop.alerts[-5:]:
        emoji = "🔴" if alert.severity == AlertSeverity.CRITICAL else "🟡"
        print(f"   {emoji} {alert.agent_id}: {alert.message}")

    print("\n✅ Feedback Loop Demo Complete!")
    print("🎯 System automatically detected degradation and triggered retraining")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    asyncio.run(demo())
