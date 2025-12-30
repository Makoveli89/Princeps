"""
A/B Testing Framework - Experiment Management for Agent Optimization

This module provides comprehensive A/B testing capabilities:
- Multi-variant experiments with traffic splitting
- Consistent user/session assignment to variants
- Real-time metrics collection
- Statistical significance testing
- Automatic winner determination
- Integration with Brain Layer for persistence

The A/B tester enables controlled experiments to test different agent
behaviors, prompts, strategies, and model versions in production.

Strategic Intent:
We might run two variants of PlannerAgent (with different council decision
rules) on a random split of tasks to see which produces better outcomes.
This module automates running experiments and logging their outcomes.

Adapted from patterns in:
- brain_layer/07_agent_training/ab_testing.py (ABTestFramework)
- brain_layer/07_agent_training/feedback_loop.py (metrics collection)
"""

import asyncio
import hashlib
import json
import logging
import math
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Configuration
# =============================================================================

class ExperimentStatus(Enum):
    """Status of an experiment."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ANALYZING = "analyzing"


class VariantType(Enum):
    """Type of experiment variant."""
    CONTROL = "control"      # Baseline version
    TREATMENT = "treatment"  # New version being tested


class MetricGoal(Enum):
    """Goal direction for metrics."""
    MAXIMIZE = "maximize"  # Higher is better (accuracy, success rate)
    MINIMIZE = "minimize"  # Lower is better (latency, error rate)


@dataclass
class ExperimentConfig:
    """Configuration for the A/B testing framework."""

    # Statistical settings
    min_sample_size: int = 100
    confidence_level: float = 0.95
    min_detectable_effect: float = 0.05  # 5% minimum effect to detect

    # Traffic settings
    default_traffic_split: float = 0.5  # 50-50 split by default

    # Persistence
    persist_results: bool = True
    results_dir: str = "./experiments"

    # Safety
    max_concurrent_experiments: int = 10
    auto_stop_on_significance: bool = False
    guard_rails_enabled: bool = True  # Don't let variants bypass safety


# =============================================================================
# Variant Definition
# =============================================================================

@dataclass
class Variant:
    """A variant in an A/B experiment."""

    id: str
    name: str
    variant_type: VariantType
    traffic_percentage: float  # 0.0 to 1.0
    config: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    # Metrics (updated during experiment)
    impressions: int = 0
    conversions: int = 0  # Success count
    total_latency_ms: float = 0.0
    total_score: float = 0.0
    error_count: int = 0

    # Additional metric accumulators
    metric_sums: Dict[str, float] = field(default_factory=dict)
    metric_counts: Dict[str, int] = field(default_factory=dict)

    def success_rate(self) -> float:
        """Calculate success/conversion rate."""
        return self.conversions / self.impressions if self.impressions > 0 else 0.0

    def average_latency(self) -> float:
        """Calculate average latency."""
        return self.total_latency_ms / self.impressions if self.impressions > 0 else 0.0

    def average_score(self) -> float:
        """Calculate average score."""
        return self.total_score / self.impressions if self.impressions > 0 else 0.0

    def error_rate(self) -> float:
        """Calculate error rate."""
        return self.error_count / self.impressions if self.impressions > 0 else 0.0

    def get_metric_average(self, metric_name: str) -> float:
        """Get average for a custom metric."""
        count = self.metric_counts.get(metric_name, 0)
        if count == 0:
            return 0.0
        return self.metric_sums.get(metric_name, 0.0) / count

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "variant_type": self.variant_type.value,
            "traffic_percentage": self.traffic_percentage,
            "config": self.config,
            "impressions": self.impressions,
            "conversions": self.conversions,
            "success_rate": self.success_rate(),
            "average_latency": self.average_latency(),
            "average_score": self.average_score(),
            "error_rate": self.error_rate(),
        }


# =============================================================================
# Experiment Definition
# =============================================================================

@dataclass
class Experiment:
    """An A/B test experiment."""

    id: str
    name: str
    description: str
    agent_id: str  # Which agent is being tested
    primary_metric: str  # Main metric to optimize
    metric_goal: MetricGoal = MetricGoal.MAXIMIZE

    variants: List[Variant] = field(default_factory=list)
    status: ExperimentStatus = ExperimentStatus.DRAFT

    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    # Targets
    target_sample_size: int = 1000
    max_duration_hours: Optional[float] = None

    # Results
    winner_variant_id: Optional[str] = None
    is_significant: bool = False
    p_value: float = 1.0
    improvement_percentage: float = 0.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_control(self) -> Optional[Variant]:
        """Get the control variant."""
        for v in self.variants:
            if v.variant_type == VariantType.CONTROL:
                return v
        return self.variants[0] if self.variants else None

    def get_treatment(self) -> Optional[Variant]:
        """Get the treatment variant."""
        for v in self.variants:
            if v.variant_type == VariantType.TREATMENT:
                return v
        return self.variants[1] if len(self.variants) > 1 else None

    def total_impressions(self) -> int:
        """Get total impressions across all variants."""
        return sum(v.impressions for v in self.variants)

    def progress(self) -> float:
        """Get progress towards target sample size."""
        return min(1.0, self.total_impressions() / self.target_sample_size)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "agent_id": self.agent_id,
            "primary_metric": self.primary_metric,
            "metric_goal": self.metric_goal.value,
            "status": self.status.value,
            "variants": [v.to_dict() for v in self.variants],
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "target_sample_size": self.target_sample_size,
            "progress": self.progress(),
            "winner_variant_id": self.winner_variant_id,
            "is_significant": self.is_significant,
            "p_value": self.p_value,
            "improvement_percentage": self.improvement_percentage,
        }


# =============================================================================
# Experiment Results
# =============================================================================

@dataclass
class ExperimentResults:
    """Results from an A/B experiment."""

    experiment_id: str
    experiment_name: str
    agent_id: str
    primary_metric: str

    # Timing
    start_time: datetime
    end_time: datetime
    duration_hours: float

    # Variant results
    control: Variant
    treatment: Variant

    # Statistical analysis
    p_value: float = 1.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    is_significant: bool = False
    confidence_level: float = 0.95

    # Winner
    winner_variant_id: Optional[str] = None
    improvement_percentage: float = 0.0
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "agent_id": self.agent_id,
            "primary_metric": self.primary_metric,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_hours": self.duration_hours,
            "control": self.control.to_dict(),
            "treatment": self.treatment.to_dict(),
            "p_value": self.p_value,
            "confidence_interval": self.confidence_interval,
            "is_significant": self.is_significant,
            "winner_variant_id": self.winner_variant_id,
            "improvement_percentage": self.improvement_percentage,
            "recommendation": self.recommendation,
        }


# =============================================================================
# Statistical Functions
# =============================================================================

def calculate_z_score(p1: float, p2: float, n1: int, n2: int) -> float:
    """Calculate z-score for two proportions."""
    if n1 == 0 or n2 == 0:
        return 0.0

    # Pooled proportion
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)

    if p_pool == 0 or p_pool == 1:
        return 0.0

    # Standard error
    se = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

    if se == 0:
        return 0.0

    return (p1 - p2) / se


def z_to_p_value(z: float) -> float:
    """Convert z-score to two-tailed p-value using approximation."""
    # Approximation of normal CDF
    x = abs(z)
    t = 1.0 / (1.0 + 0.2316419 * x)
    d = 0.3989423 * math.exp(-x * x / 2)
    p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))

    return 2 * p  # Two-tailed


def calculate_confidence_interval(
    p: float,
    n: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Calculate confidence interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)

    # Z-value for confidence level
    z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%

    # Standard error
    se = math.sqrt(p * (1 - p) / n)

    margin = z * se
    return (max(0, p - margin), min(1, p + margin))


def calculate_sample_size(
    baseline_rate: float,
    min_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Calculate required sample size per variant."""
    if baseline_rate <= 0 or baseline_rate >= 1:
        return 100  # Default

    p1 = baseline_rate
    p2 = baseline_rate * (1 + min_detectable_effect)

    # Z-values
    z_alpha = 1.96  # Two-tailed 0.05
    z_beta = 0.84   # Power 0.8

    p_avg = (p1 + p2) / 2
    n = (2 * p_avg * (1 - p_avg) * (z_alpha + z_beta) ** 2) / ((p2 - p1) ** 2)

    return max(100, int(n))


# =============================================================================
# A/B Tester Implementation
# =============================================================================

class ABTester:
    """
    A/B Testing Framework for Agent Optimization.

    Enables controlled experiments to test different agent configurations,
    prompts, and strategies with statistical rigor.

    Usage:
        tester = ABTester()

        # Create experiment
        exp_id = tester.create_experiment(
            name="Planner Prompt Test",
            agent_id="planner",
            primary_metric="success_rate",
            control_config={"prompt_version": "v1"},
            treatment_config={"prompt_version": "v2"},
        )

        # Start experiment
        tester.start_experiment(exp_id)

        # For each request, get variant and record results
        variant = tester.assign_variant(exp_id, user_id="user_123")
        # ... execute with variant.config ...
        tester.record_result(exp_id, user_id, success=True, latency_ms=150)

        # Analyze results
        results = tester.analyze_experiment(exp_id)
    """

    def __init__(
        self,
        config: Optional[ExperimentConfig] = None,
    ):
        self.config = config or ExperimentConfig()

        # Experiments registry
        self._experiments: Dict[str, Experiment] = {}

        # Variant assignments (consistent hashing)
        self._assignments: Dict[str, Dict[str, str]] = {}  # {exp_id: {user_id: variant_id}}

        # Results directory
        self._results_dir = Path(self.config.results_dir)
        if self.config.persist_results:
            self._results_dir.mkdir(parents=True, exist_ok=True)

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Statistics
        self._stats = {
            "total_experiments": 0,
            "completed_experiments": 0,
            "total_impressions": 0,
        }

        logger.info("ABTester initialized")

    # =========================================================================
    # Experiment Management
    # =========================================================================

    def create_experiment(
        self,
        name: str,
        agent_id: str,
        primary_metric: str,
        control_config: Dict[str, Any],
        treatment_config: Dict[str, Any],
        traffic_split: float = 0.5,
        description: str = "",
        target_sample_size: int = 1000,
        metric_goal: MetricGoal = MetricGoal.MAXIMIZE,
        **metadata,
    ) -> str:
        """
        Create a new A/B experiment.

        Args:
            name: Experiment name
            agent_id: ID of agent being tested
            primary_metric: Main metric to optimize
            control_config: Configuration for control variant
            treatment_config: Configuration for treatment variant
            traffic_split: Fraction of traffic to treatment (0.0-1.0)
            description: Experiment description
            target_sample_size: Target total sample size
            metric_goal: Whether to maximize or minimize the metric
            **metadata: Additional metadata

        Returns:
            Experiment ID
        """
        exp_id = f"exp-{agent_id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Create variants
        control = Variant(
            id=f"{exp_id}-control",
            name="Control",
            variant_type=VariantType.CONTROL,
            traffic_percentage=1.0 - traffic_split,
            config=control_config,
            description="Baseline version",
        )

        treatment = Variant(
            id=f"{exp_id}-treatment",
            name="Treatment",
            variant_type=VariantType.TREATMENT,
            traffic_percentage=traffic_split,
            config=treatment_config,
            description="New version being tested",
        )

        experiment = Experiment(
            id=exp_id,
            name=name,
            description=description,
            agent_id=agent_id,
            primary_metric=primary_metric,
            metric_goal=metric_goal,
            variants=[control, treatment],
            target_sample_size=target_sample_size,
            metadata=metadata,
        )

        self._experiments[exp_id] = experiment
        self._assignments[exp_id] = {}
        self._stats["total_experiments"] += 1

        logger.info(f"Created experiment {exp_id}: {name}")
        logger.info(f"  Agent: {agent_id}, Metric: {primary_metric}")
        logger.info(f"  Traffic split: {(1-traffic_split)*100:.0f}% control, {traffic_split*100:.0f}% treatment")

        return exp_id

    def start_experiment(self, experiment_id: str) -> None:
        """Start an experiment."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            raise ValueError(f"Experiment {experiment_id} not found")

        if exp.status != ExperimentStatus.DRAFT:
            raise ValueError(f"Experiment {experiment_id} is not in DRAFT status")

        exp.status = ExperimentStatus.RUNNING
        exp.started_at = datetime.utcnow()

        logger.info(f"Started experiment {experiment_id}: {exp.name}")

    def pause_experiment(self, experiment_id: str) -> None:
        """Pause a running experiment."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            raise ValueError(f"Experiment {experiment_id} not found")

        if exp.status == ExperimentStatus.RUNNING:
            exp.status = ExperimentStatus.PAUSED
            logger.info(f"Paused experiment {experiment_id}")

    def resume_experiment(self, experiment_id: str) -> None:
        """Resume a paused experiment."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            raise ValueError(f"Experiment {experiment_id} not found")

        if exp.status == ExperimentStatus.PAUSED:
            exp.status = ExperimentStatus.RUNNING
            logger.info(f"Resumed experiment {experiment_id}")

    def stop_experiment(self, experiment_id: str) -> ExperimentResults:
        """Stop an experiment and get final results."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            raise ValueError(f"Experiment {experiment_id} not found")

        exp.status = ExperimentStatus.COMPLETED
        exp.ended_at = datetime.utcnow()

        results = self._analyze_experiment(exp)

        # Update experiment with results
        exp.winner_variant_id = results.winner_variant_id
        exp.is_significant = results.is_significant
        exp.p_value = results.p_value
        exp.improvement_percentage = results.improvement_percentage

        # Persist results
        if self.config.persist_results:
            self._save_results(results)

        self._stats["completed_experiments"] += 1

        logger.info(f"Stopped experiment {experiment_id}")
        logger.info(f"  Winner: {results.winner_variant_id}")
        logger.info(f"  Improvement: {results.improvement_percentage:+.2f}%")
        logger.info(f"  Significant: {results.is_significant}")

        return results

    # =========================================================================
    # Variant Assignment
    # =========================================================================

    def assign_variant(
        self,
        experiment_id: str,
        user_id: str,
    ) -> Variant:
        """
        Assign a user to a variant.

        Uses consistent hashing to ensure the same user always gets
        the same variant within an experiment.

        Args:
            experiment_id: Experiment ID
            user_id: User or session identifier

        Returns:
            Assigned Variant with configuration
        """
        exp = self._experiments.get(experiment_id)

        if not exp or exp.status != ExperimentStatus.RUNNING:
            # Return control variant if experiment not running
            if exp and exp.variants:
                return exp.get_control()
            raise ValueError(f"Experiment {experiment_id} not running")

        # Check for existing assignment
        if user_id in self._assignments.get(experiment_id, {}):
            variant_id = self._assignments[experiment_id][user_id]
            for v in exp.variants:
                if v.id == variant_id:
                    return v

        # Assign based on consistent hash
        hash_input = f"{experiment_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16) % 100 / 100.0

        cumulative = 0.0
        assigned_variant = exp.variants[0]  # Default to first

        for variant in exp.variants:
            cumulative += variant.traffic_percentage
            if hash_value < cumulative:
                assigned_variant = variant
                break

        # Store assignment
        if experiment_id not in self._assignments:
            self._assignments[experiment_id] = {}
        self._assignments[experiment_id][user_id] = assigned_variant.id

        return assigned_variant

    def get_variant_config(
        self,
        experiment_id: str,
        user_id: str,
    ) -> Dict[str, Any]:
        """Get the configuration for a user's assigned variant."""
        variant = self.assign_variant(experiment_id, user_id)
        return variant.config

    # =========================================================================
    # Result Recording
    # =========================================================================

    def record_impression(
        self,
        experiment_id: str,
        user_id: str,
        latency_ms: float = 0.0,
    ) -> None:
        """
        Record an impression (exposure to variant).

        Args:
            experiment_id: Experiment ID
            user_id: User identifier
            latency_ms: Latency in milliseconds
        """
        exp = self._experiments.get(experiment_id)
        if not exp or exp.status != ExperimentStatus.RUNNING:
            return

        variant = self.assign_variant(experiment_id, user_id)
        variant.impressions += 1
        variant.total_latency_ms += latency_ms

        self._stats["total_impressions"] += 1

    def record_result(
        self,
        experiment_id: str,
        user_id: str,
        success: bool = False,
        score: float = 0.0,
        latency_ms: float = 0.0,
        error: bool = False,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Record the result of an interaction.

        Args:
            experiment_id: Experiment ID
            user_id: User identifier
            success: Whether the interaction was successful
            score: Quality score (0.0-1.0)
            latency_ms: Latency in milliseconds
            error: Whether an error occurred
            metrics: Additional custom metrics
        """
        exp = self._experiments.get(experiment_id)
        if not exp or exp.status != ExperimentStatus.RUNNING:
            return

        variant = self.assign_variant(experiment_id, user_id)

        # Update core metrics
        variant.impressions += 1
        if success:
            variant.conversions += 1
        variant.total_latency_ms += latency_ms
        variant.total_score += score
        if error:
            variant.error_count += 1

        # Update custom metrics
        if metrics:
            for name, value in metrics.items():
                if name not in variant.metric_sums:
                    variant.metric_sums[name] = 0.0
                    variant.metric_counts[name] = 0
                variant.metric_sums[name] += value
                variant.metric_counts[name] += 1

        self._stats["total_impressions"] += 1

        # Check if we should auto-stop
        if self.config.auto_stop_on_significance:
            if exp.total_impressions() >= self.config.min_sample_size:
                results = self._analyze_experiment(exp)
                if results.is_significant:
                    logger.info(f"Auto-stopping experiment {experiment_id} - significance reached")
                    self.stop_experiment(experiment_id)

    def record_conversion(
        self,
        experiment_id: str,
        user_id: str,
    ) -> None:
        """Record a conversion (success) for a user."""
        exp = self._experiments.get(experiment_id)
        if not exp or exp.status != ExperimentStatus.RUNNING:
            return

        # Get variant assignment
        if experiment_id not in self._assignments:
            return
        if user_id not in self._assignments[experiment_id]:
            return

        variant_id = self._assignments[experiment_id][user_id]
        for variant in exp.variants:
            if variant.id == variant_id:
                variant.conversions += 1
                break

    # =========================================================================
    # Analysis
    # =========================================================================

    def analyze_experiment(self, experiment_id: str) -> ExperimentResults:
        """Analyze an experiment and get results."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            raise ValueError(f"Experiment {experiment_id} not found")

        return self._analyze_experiment(exp)

    def _analyze_experiment(self, exp: Experiment) -> ExperimentResults:
        """Internal analysis method."""
        control = exp.get_control()
        treatment = exp.get_treatment()

        if not control or not treatment:
            raise ValueError("Experiment must have control and treatment variants")

        # Calculate duration
        start_time = exp.started_at or exp.created_at
        end_time = exp.ended_at or datetime.utcnow()
        duration_hours = (end_time - start_time).total_seconds() / 3600

        # Get primary metric values
        if exp.primary_metric == "success_rate":
            control_value = control.success_rate()
            treatment_value = treatment.success_rate()
        elif exp.primary_metric == "latency":
            control_value = control.average_latency()
            treatment_value = treatment.average_latency()
        elif exp.primary_metric == "score":
            control_value = control.average_score()
            treatment_value = treatment.average_score()
        elif exp.primary_metric == "error_rate":
            control_value = control.error_rate()
            treatment_value = treatment.error_rate()
        else:
            control_value = control.get_metric_average(exp.primary_metric)
            treatment_value = treatment.get_metric_average(exp.primary_metric)

        # Calculate statistical significance
        z_score = calculate_z_score(
            control.success_rate(),
            treatment.success_rate(),
            control.impressions,
            treatment.impressions,
        )
        p_value = z_to_p_value(z_score)

        is_significant = (
            p_value < (1 - self.config.confidence_level)
            and control.impressions >= self.config.min_sample_size
            and treatment.impressions >= self.config.min_sample_size
        )

        # Calculate confidence interval
        ci = calculate_confidence_interval(
            treatment.success_rate() - control.success_rate(),
            control.impressions + treatment.impressions,
            self.config.confidence_level,
        )

        # Determine winner
        if exp.metric_goal == MetricGoal.MAXIMIZE:
            winner_id = treatment.id if treatment_value > control_value else control.id
            improvement = ((treatment_value - control_value) / control_value * 100) if control_value > 0 else 0
        else:
            winner_id = treatment.id if treatment_value < control_value else control.id
            improvement = ((control_value - treatment_value) / control_value * 100) if control_value > 0 else 0

        # Generate recommendation
        if not is_significant:
            recommendation = "Not enough evidence to determine a winner. Continue the experiment."
        elif winner_id == treatment.id:
            recommendation = f"Treatment is the winner with {improvement:+.1f}% improvement. Consider deploying."
        else:
            recommendation = f"Control performs better. Treatment shows {improvement:+.1f}% degradation."

        return ExperimentResults(
            experiment_id=exp.id,
            experiment_name=exp.name,
            agent_id=exp.agent_id,
            primary_metric=exp.primary_metric,
            start_time=start_time,
            end_time=end_time,
            duration_hours=duration_hours,
            control=control,
            treatment=treatment,
            p_value=p_value,
            confidence_interval=ci,
            is_significant=is_significant,
            confidence_level=self.config.confidence_level,
            winner_variant_id=winner_id if is_significant else None,
            improvement_percentage=improvement,
            recommendation=recommendation,
        )

    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get current status of an experiment."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return {"error": "Experiment not found"}

        return exp.to_dict()

    def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None,
        agent_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List experiments with optional filtering."""
        results = []

        for exp in self._experiments.values():
            if status and exp.status != status:
                continue
            if agent_id and exp.agent_id != agent_id:
                continue
            results.append(exp.to_dict())

        return results

    # =========================================================================
    # Persistence
    # =========================================================================

    def _save_results(self, results: ExperimentResults) -> None:
        """Save experiment results to disk."""
        results_file = self._results_dir / f"{results.experiment_id}_results.json"

        with open(results_file, "w") as f:
            json.dump(results.to_dict(), f, indent=2, default=str)

        logger.info(f"Saved results to {results_file}")

    def load_experiment(self, experiment_id: str) -> Optional[ExperimentResults]:
        """Load saved experiment results."""
        results_file = self._results_dir / f"{experiment_id}_results.json"

        if not results_file.exists():
            return None

        with open(results_file) as f:
            data = json.load(f)

        # Reconstruct results (simplified)
        return data

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get A/B tester statistics."""
        running = sum(1 for e in self._experiments.values() if e.status == ExperimentStatus.RUNNING)

        return {
            **self._stats,
            "running_experiments": running,
            "active_experiments": list(
                e.id for e in self._experiments.values()
                if e.status == ExperimentStatus.RUNNING
            ),
        }


# =============================================================================
# Factory Function
# =============================================================================

# Global A/B tester instance
_ab_tester: Optional[ABTester] = None


def get_ab_tester(config: Optional[ExperimentConfig] = None) -> ABTester:
    """Get or create the global A/B tester."""
    global _ab_tester

    if _ab_tester is None:
        _ab_tester = ABTester(config)

    return _ab_tester


def create_experiment(
    name: str,
    agent_id: str,
    primary_metric: str,
    control_config: Dict[str, Any],
    treatment_config: Dict[str, Any],
    **kwargs,
) -> str:
    """Convenience function to create an experiment."""
    tester = get_ab_tester()
    return tester.create_experiment(
        name=name,
        agent_id=agent_id,
        primary_metric=primary_metric,
        control_config=control_config,
        treatment_config=treatment_config,
        **kwargs,
    )
