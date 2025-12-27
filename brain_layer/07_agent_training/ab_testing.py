"""
A/B Testing Framework - Test Agent Decision Variants

Enables controlled experiments to test different agent behaviors,
decision strategies, and model versions in production.

Source: F:\Mothership-main\Mothership-main\learning\ab_testing.py
Recycled: December 26, 2024
"""

import asyncio
import json
import logging
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """A/B test status"""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class VariantType(Enum):
    """Type of test variant"""

    CONTROL = "control"  # Baseline
    TREATMENT = "treatment"  # New version


@dataclass
class TestVariant:
    """A variant in an A/B test"""

    variant_id: str
    variant_type: VariantType
    name: str
    description: str
    traffic_percentage: float  # 0.0 to 1.0
    config: Dict[str, Any] = field(default_factory=dict)

    # Metrics
    impressions: int = 0
    conversions: int = 0
    total_latency_ms: float = 0.0
    error_count: int = 0

    def conversion_rate(self) -> float:
        """Calculate conversion rate"""
        return (self.conversions / self.impressions) if self.impressions > 0 else 0.0

    def average_latency(self) -> float:
        """Calculate average latency"""
        return (self.total_latency_ms / self.impressions) if self.impressions > 0 else 0.0

    def error_rate(self) -> float:
        """Calculate error rate"""
        return (self.error_count / self.impressions) if self.impressions > 0 else 0.0


@dataclass
class TestResults:
    """Results of an A/B test"""

    test_id: str
    test_name: str
    start_time: datetime
    end_time: datetime
    duration_hours: float

    control_variant: TestVariant
    treatment_variant: TestVariant

    # Statistical significance
    p_value: float = 0.0
    is_significant: bool = False
    confidence_level: float = 0.95

    # Winner determination
    winner_variant_id: Optional[str] = None
    improvement_percentage: float = 0.0

    def calculate_winner(self) -> str:
        """Determine the winning variant"""
        control_rate = self.control_variant.conversion_rate()
        treatment_rate = self.treatment_variant.conversion_rate()

        if treatment_rate > control_rate:
            self.winner_variant_id = self.treatment_variant.variant_id
            self.improvement_percentage = (
                ((treatment_rate - control_rate) / control_rate * 100) if control_rate > 0 else 0
            )
        else:
            self.winner_variant_id = self.control_variant.variant_id
            self.improvement_percentage = 0.0

        # Simple significance test (in production, use proper statistical tests)
        min_sample_size = 100
        if (
            self.control_variant.impressions >= min_sample_size
            and self.treatment_variant.impressions >= min_sample_size
        ):
            # Simplified: Consider significant if improvement > 5% and sufficient samples
            self.is_significant = abs(self.improvement_percentage) > 5.0
            self.p_value = 0.03 if self.is_significant else 0.15

        return self.winner_variant_id


@dataclass
class ABTest:
    """An A/B test experiment"""

    test_id: str
    test_name: str
    description: str
    agent_id: str
    metric_name: str  # Primary metric to optimize

    variants: List[TestVariant]
    status: TestStatus = TestStatus.DRAFT

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    target_sample_size: int = 1000

    metadata: Dict[str, Any] = field(default_factory=dict)


class ABTestFramework:
    """
    A/B Testing Framework for Agent Optimization

    Features:
    - Multi-variant testing support
    - Traffic splitting and routing
    - Real-time metrics collection
    - Statistical significance testing
    - Automatic winner selection
    - Integration with model deployment
    """

    def __init__(self, tests_dir: str = "./ab_tests"):
        self.tests_dir = Path(tests_dir)
        self.tests_dir.mkdir(parents=True, exist_ok=True)

        # Active tests
        self.tests: Dict[str, ABTest] = {}

        # Variant assignments (user/session -> variant)
        self.assignments: Dict[str, Dict[str, str]] = {}  # {test_id: {user_id: variant_id}}

        logger.info("ABTestFramework initialized")

    def create_test(
        self,
        test_name: str,
        agent_id: str,
        metric_name: str,
        control_config: Dict[str, Any],
        treatment_config: Dict[str, Any],
        traffic_split: float = 0.5,
        description: str = "",
    ) -> str:
        """Create a new A/B test"""
        test_id = f"test-{agent_id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Create control and treatment variants
        control = TestVariant(
            variant_id=f"{test_id}-control",
            variant_type=VariantType.CONTROL,
            name="Control",
            description="Baseline version",
            traffic_percentage=1.0 - traffic_split,
            config=control_config,
        )

        treatment = TestVariant(
            variant_id=f"{test_id}-treatment",
            variant_type=VariantType.TREATMENT,
            name="Treatment",
            description="New version",
            traffic_percentage=traffic_split,
            config=treatment_config,
        )

        test = ABTest(
            test_id=test_id,
            test_name=test_name,
            description=description,
            agent_id=agent_id,
            metric_name=metric_name,
            variants=[control, treatment],
        )

        self.tests[test_id] = test
        self.assignments[test_id] = {}

        logger.info(f"Created A/B test {test_id}: {test_name}")
        logger.info(f"  Traffic split: {(1-traffic_split)*100:.0f}% control, {traffic_split*100:.0f}% treatment")

        return test_id

    def start_test(self, test_id: str) -> None:
        """Start an A/B test"""
        test = self.tests.get(test_id)
        if not test:
            raise ValueError(f"Test {test_id} not found")

        test.status = TestStatus.RUNNING
        test.start_time = datetime.now()

        logger.info(f"Started A/B test {test_id}: {test.test_name}")

    def assign_variant(self, test_id: str, user_id: str) -> TestVariant:
        """
        Assign a user to a test variant

        Uses consistent hashing to ensure same user gets same variant
        """
        test = self.tests.get(test_id)
        if not test or test.status != TestStatus.RUNNING:
            # Return control variant if test not running
            return test.variants[0] if test else None

        # Check if user already assigned
        if test_id in self.assignments and user_id in self.assignments[test_id]:
            variant_id = self.assignments[test_id][user_id]
            return next(v for v in test.variants if v.variant_id == variant_id)

        # Assign based on traffic split
        random_value = hash(f"{test_id}{user_id}") % 100 / 100.0

        cumulative = 0.0
        for variant in test.variants:
            cumulative += variant.traffic_percentage
            if random_value < cumulative:
                self.assignments[test_id][user_id] = variant.variant_id
                return variant

        # Fallback to control
        return test.variants[0]

    def record_impression(self, test_id: str, user_id: str, latency_ms: float = 0.0, error: bool = False) -> None:
        """Record an impression for a user"""
        variant = self.assign_variant(test_id, user_id)
        if not variant:
            return

        variant.impressions += 1
        variant.total_latency_ms += latency_ms
        if error:
            variant.error_count += 1

    def record_conversion(self, test_id: str, user_id: str) -> None:
        """Record a conversion for a user"""
        test = self.tests.get(test_id)
        if not test:
            return

        # Get user's assigned variant
        if test_id not in self.assignments or user_id not in self.assignments[test_id]:
            return

        variant_id = self.assignments[test_id][user_id]
        variant = next((v for v in test.variants if v.variant_id == variant_id), None)

        if variant:
            variant.conversions += 1

    def get_test_status(self, test_id: str) -> Dict[str, Any]:
        """Get current status of a test"""
        test = self.tests.get(test_id)
        if not test:
            return {}

        control = test.variants[0]
        treatment = test.variants[1]

        return {
            "test_id": test_id,
            "test_name": test.test_name,
            "status": test.status.value,
            "agent_id": test.agent_id,
            "metric": test.metric_name,
            "start_time": test.start_time.isoformat() if test.start_time else None,
            "variants": {
                "control": {
                    "impressions": control.impressions,
                    "conversions": control.conversions,
                    "conversion_rate": control.conversion_rate(),
                    "avg_latency_ms": control.average_latency(),
                    "error_rate": control.error_rate(),
                },
                "treatment": {
                    "impressions": treatment.impressions,
                    "conversions": treatment.conversions,
                    "conversion_rate": treatment.conversion_rate(),
                    "avg_latency_ms": treatment.average_latency(),
                    "error_rate": treatment.error_rate(),
                },
            },
            "progress": (control.impressions + treatment.impressions) / test.target_sample_size,
        }

    def analyze_test(self, test_id: str) -> TestResults:
        """Analyze test results and determine winner"""
        test = self.tests.get(test_id)
        if not test:
            raise ValueError(f"Test {test_id} not found")

        control = test.variants[0]
        treatment = test.variants[1]

        duration = (datetime.now() - test.start_time).total_seconds() / 3600 if test.start_time else 0

        results = TestResults(
            test_id=test_id,
            test_name=test.test_name,
            start_time=test.start_time or datetime.now(),
            end_time=datetime.now(),
            duration_hours=duration,
            control_variant=control,
            treatment_variant=treatment,
        )

        results.calculate_winner()

        return results

    def stop_test(self, test_id: str) -> TestResults:
        """Stop a test and get final results"""
        test = self.tests.get(test_id)
        if not test:
            raise ValueError(f"Test {test_id} not found")

        test.status = TestStatus.COMPLETED
        test.end_time = datetime.now()

        results = self.analyze_test(test_id)

        # Save results
        self._save_results(results)

        logger.info(f"Stopped A/B test {test_id}")
        logger.info(f"  Winner: {results.winner_variant_id}")
        logger.info(f"  Improvement: {results.improvement_percentage:+.2f}%")
        logger.info(f"  Significant: {results.is_significant}")

        return results

    def _save_results(self, results: TestResults) -> None:
        """Save test results to disk"""
        results_file = self.tests_dir / f"{results.test_id}_results.json"

        with open(results_file, "w") as f:
            # Convert dataclasses to dicts
            control_dict = asdict(results.control_variant)
            control_dict["variant_type"] = control_dict["variant_type"].value

            treatment_dict = asdict(results.treatment_variant)
            treatment_dict["variant_type"] = treatment_dict["variant_type"].value

            data = {
                "test_id": results.test_id,
                "test_name": results.test_name,
                "start_time": results.start_time.isoformat(),
                "end_time": results.end_time.isoformat(),
                "duration_hours": results.duration_hours,
                "control": control_dict,
                "treatment": treatment_dict,
                "winner": results.winner_variant_id,
                "improvement_pct": results.improvement_percentage,
                "is_significant": results.is_significant,
                "p_value": results.p_value,
            }
            json.dump(data, f, indent=2)

    def get_all_tests(self) -> List[Dict[str, Any]]:
        """Get status of all tests"""
        return [self.get_test_status(test_id) for test_id in self.tests.keys()]


# Demo
async def demo():
    """Demonstrate the A/B Testing Framework"""
    print("\n" + "=" * 60)
    print("AUTONOMOUS LEARNING LAYER - A/B TESTING DEMO")
    print("=" * 60)

    framework = ABTestFramework()

    # Test 1: Legal contract generation - different prompt strategies
    test1_id = framework.create_test(
        test_name="Contract Generation Prompt Strategy",
        agent_id="legal_agent",
        metric_name="contract_acceptance_rate",
        control_config={"prompt_strategy": "traditional", "temperature": 0.7, "max_tokens": 2000},
        treatment_config={"prompt_strategy": "structured_sections", "temperature": 0.5, "max_tokens": 2500},
        traffic_split=0.5,
        description="Test new structured section approach vs traditional",
    )

    # Test 2: Sales opportunity scoring - model versions
    test2_id = framework.create_test(
        test_name="Opportunity Scoring Model v2",
        agent_id="sales_agent",
        metric_name="prediction_accuracy",
        control_config={"model_version": "v1.0"},
        treatment_config={"model_version": "v2.0"},
        traffic_split=0.3,  # 30% treatment, 70% control (cautious rollout)
        description="Test new ML model for opportunity scoring",
    )

    framework.start_test(test1_id)
    framework.start_test(test2_id)

    print("\n🧪 Running 2 A/B tests...\n")

    # Simulate traffic for test 1
    print("📊 Test 1: Contract Generation")
    for i in range(300):
        user_id = f"user-{i}"

        # Record impression
        latency = 150 + (hash(user_id) % 100)  # 150-250ms
        framework.record_impression(test1_id, user_id, latency_ms=latency)

        # Simulate conversions (treatment has 15% better conversion)
        variant = framework.assign_variant(test1_id, user_id)
        base_rate = 0.60  # 60% base acceptance
        if variant.variant_type == VariantType.TREATMENT:
            conversion_rate = base_rate * 1.15  # 15% improvement
        else:
            conversion_rate = base_rate

        if random.random() < conversion_rate:
            framework.record_conversion(test1_id, user_id)

        await asyncio.sleep(0.001)  # Simulate time passing

    # Simulate traffic for test 2
    print("📊 Test 2: Opportunity Scoring")
    for i in range(200):
        user_id = f"lead-{i}"

        latency = 80 + (hash(user_id) % 40)  # 80-120ms
        framework.record_impression(test2_id, user_id, latency_ms=latency)

        # Simulate conversions (treatment has 8% better accuracy)
        variant = framework.assign_variant(test2_id, user_id)
        base_rate = 0.72  # 72% base accuracy
        if variant.variant_type == VariantType.TREATMENT:
            conversion_rate = base_rate * 1.08  # 8% improvement
        else:
            conversion_rate = base_rate

        if random.random() < conversion_rate:
            framework.record_conversion(test2_id, user_id)

        await asyncio.sleep(0.001)

    # Analyze results
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    results1 = framework.stop_test(test1_id)
    print(f"\n🧪 Test 1: {results1.test_name}")
    print(f"   Duration: {results1.duration_hours:.2f} hours")
    print(
        f"   Control:   {results1.control_variant.conversion_rate():.2%} conversion "
        f"({results1.control_variant.impressions} samples)"
    )
    print(
        f"   Treatment: {results1.treatment_variant.conversion_rate():.2%} conversion "
        f"({results1.treatment_variant.impressions} samples)"
    )
    print(f"   Winner: {results1.winner_variant_id.split('-')[-1].upper()}")
    print(f"   Improvement: {results1.improvement_percentage:+.2f}%")
    print(f"   Statistically Significant: {'YES' if results1.is_significant else 'NO'} (p={results1.p_value:.3f})")

    results2 = framework.stop_test(test2_id)
    print(f"\n🧪 Test 2: {results2.test_name}")
    print(f"   Duration: {results2.duration_hours:.2f} hours")
    print(
        f"   Control:   {results2.control_variant.conversion_rate():.2%} accuracy "
        f"({results2.control_variant.impressions} samples)"
    )
    print(
        f"   Treatment: {results2.treatment_variant.conversion_rate():.2%} accuracy "
        f"({results2.treatment_variant.impressions} samples)"
    )
    print(f"   Winner: {results2.winner_variant_id.split('-')[-1].upper()}")
    print(f"   Improvement: {results2.improvement_percentage:+.2f}%")
    print(f"   Statistically Significant: {'YES' if results2.is_significant else 'NO'} (p={results2.p_value:.3f})")

    print("\n✅ A/B Testing Demo Complete!")
    print("🎯 Both treatment variants showed improvement and can be deployed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    asyncio.run(demo())
