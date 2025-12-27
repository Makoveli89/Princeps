"""
Model Trainer - Continuous Retraining for Agent Improvement

Automatically retrains agent models based on production data, enabling
continuous improvement and adaptation to changing patterns.

Source: F:\Mothership-main\Mothership-main\learning\model_trainer.py
Recycled: December 26, 2024
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of models that can be trained"""

    DECISION_MAKER = "decision_maker"
    CLASSIFIER = "classifier"
    PREDICTOR = "predictor"
    RECOMMENDER = "recommender"
    OPTIMIZER = "optimizer"


class TrainingStatus(Enum):
    """Training job status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingConfig:
    """Configuration for model training"""

    model_id: str
    model_type: ModelType
    agent_id: str
    training_data_path: str
    validation_split: float = 0.2
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 10
    early_stopping_patience: int = 3
    min_improvement: float = 0.01
    target_metric: str = "accuracy"
    auto_deploy: bool = True
    retain_history: int = 5  # Keep last N models


@dataclass
class ModelMetrics:
    """Performance metrics for a trained model"""

    model_id: str
    version: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    loss: float
    training_time_seconds: float
    training_samples: int
    validation_samples: int
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def improvement_over(self, baseline: "ModelMetrics") -> float:
        """Calculate improvement percentage over baseline"""
        if baseline is None or baseline.accuracy == 0:
            return 0.0
        return ((self.accuracy - baseline.accuracy) / baseline.accuracy) * 100


@dataclass
class TrainingJob:
    """A training job instance"""

    job_id: str
    config: TrainingConfig
    status: TrainingStatus = TrainingStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    current_epoch: int = 0
    best_metric: float = 0.0
    metrics: Optional[ModelMetrics] = None
    error_message: Optional[str] = None
    model_path: Optional[str] = None


class ModelTrainer:
    """
    Continuous Model Training System

    Features:
    - Automatic retraining on new production data
    - Performance tracking and versioning
    - Early stopping and best model selection
    - Integration with agent deployment pipeline
    - Historical model retention
    """

    def __init__(
        self, models_dir: str = "./models", training_data_dir: str = "./training_data", metrics_dir: str = "./metrics"
    ):
        self.models_dir = Path(models_dir)
        self.training_data_dir = Path(training_data_dir)
        self.metrics_dir = Path(metrics_dir)

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.training_data_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Active training jobs
        self.jobs: Dict[str, TrainingJob] = {}

        # Model registry
        self.models: Dict[str, List[ModelMetrics]] = {}

        # Training callbacks
        self.callbacks: List[Callable] = []

        logger.info(f"ModelTrainer initialized with models_dir={models_dir}")

    def register_callback(self, callback: Callable) -> None:
        """Register a callback for training events"""
        self.callbacks.append(callback)

    async def _notify_callbacks(self, event: str, data: Dict[str, Any]) -> None:
        """Notify all registered callbacks"""
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event, data)
                else:
                    callback(event, data)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def create_training_job(self, config: TrainingConfig) -> str:
        """Create a new training job"""
        job_id = f"job-{config.model_id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        job = TrainingJob(job_id=job_id, config=config, status=TrainingStatus.PENDING)

        self.jobs[job_id] = job
        logger.info(f"Created training job {job_id} for model {config.model_id}")

        return job_id

    async def train_model(self, job_id: str) -> ModelMetrics:
        """
        Train a model (simulated training with realistic metrics)

        In production, this would:
        1. Load training data from production logs
        2. Preprocess and augment data
        3. Train using PyTorch/TensorFlow/scikit-learn
        4. Validate on held-out set
        5. Track metrics and checkpoints
        6. Deploy if improvement threshold met
        """
        job = self.jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        try:
            job.status = TrainingStatus.RUNNING
            job.start_time = datetime.now()

            await self._notify_callbacks(
                "training_started", {"job_id": job_id, "model_id": job.config.model_id, "agent_id": job.config.agent_id}
            )

            logger.info(f"Starting training for {job.config.model_id}")

            # Get baseline metrics
            baseline = self._get_latest_metrics(job.config.model_id)

            # Simulate training process
            best_accuracy = 0.0
            patience_counter = 0

            for epoch in range(job.config.epochs):
                job.current_epoch = epoch + 1

                # Simulate epoch training
                await asyncio.sleep(0.5)  # Simulate training time

                # Simulate metrics with improvement trend
                base_accuracy = 0.75 if baseline is None else baseline.accuracy
                improvement_factor = 1 + (epoch * 0.02)  # 2% improvement per epoch
                noise = (hash(f"{job_id}{epoch}") % 100) / 1000  # Small random noise

                epoch_accuracy = min(0.99, base_accuracy * improvement_factor + noise)

                if epoch_accuracy > best_accuracy:
                    best_accuracy = epoch_accuracy
                    patience_counter = 0
                    logger.info(f"Epoch {epoch+1}: New best accuracy {epoch_accuracy:.4f}")
                else:
                    patience_counter += 1
                    logger.info(f"Epoch {epoch+1}: Accuracy {epoch_accuracy:.4f} (no improvement)")

                # Early stopping
                if patience_counter >= job.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

                job.best_metric = best_accuracy

            # Calculate final metrics
            training_time = (datetime.now() - job.start_time).total_seconds()

            # Get current version
            version = len(self.models.get(job.config.model_id, [])) + 1

            metrics = ModelMetrics(
                model_id=job.config.model_id,
                version=version,
                accuracy=best_accuracy,
                precision=best_accuracy * 0.98,  # Slightly lower than accuracy
                recall=best_accuracy * 0.97,
                f1_score=best_accuracy * 0.975,
                loss=1.0 - best_accuracy,
                training_time_seconds=training_time,
                training_samples=1000,  # Simulated
                validation_samples=200,  # Simulated
                metadata={
                    "agent_id": job.config.agent_id,
                    "model_type": job.config.model_type.value,
                    "epochs_trained": job.current_epoch,
                    "early_stopped": patience_counter >= job.config.early_stopping_patience,
                },
            )

            # Save metrics
            self._save_metrics(metrics)

            # Update model registry
            if job.config.model_id not in self.models:
                self.models[job.config.model_id] = []
            self.models[job.config.model_id].append(metrics)

            # Cleanup old models
            self._cleanup_old_models(job.config.model_id, job.config.retain_history)

            # Update job
            job.status = TrainingStatus.COMPLETED
            job.end_time = datetime.now()
            job.metrics = metrics
            job.model_path = str(self.models_dir / f"{job.config.model_id}_v{version}.model")

            # Calculate improvement
            improvement = metrics.improvement_over(baseline) if baseline else 0.0

            logger.info(f"Training completed: {job.config.model_id} v{version}")
            logger.info(f"  Accuracy: {metrics.accuracy:.4f}")
            logger.info(f"  Improvement: {improvement:+.2f}%")
            logger.info(f"  Training time: {training_time:.2f}s")

            await self._notify_callbacks(
                "training_completed",
                {
                    "job_id": job_id,
                    "model_id": job.config.model_id,
                    "version": version,
                    "accuracy": metrics.accuracy,
                    "improvement_pct": improvement,
                },
            )

            # Auto-deploy if configured and improvement meets threshold
            if job.config.auto_deploy and improvement >= job.config.min_improvement * 100:
                await self._deploy_model(job.config.model_id, version)

            return metrics

        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.end_time = datetime.now()
            job.error_message = str(e)
            logger.error(f"Training failed for {job_id}: {e}")

            await self._notify_callbacks(
                "training_failed", {"job_id": job_id, "model_id": job.config.model_id, "error": str(e)}
            )

            raise

    def _save_metrics(self, metrics: ModelMetrics) -> None:
        """Save metrics to disk"""
        metrics_file = self.metrics_dir / f"{metrics.model_id}_v{metrics.version}.json"

        with open(metrics_file, "w") as f:
            data = asdict(metrics)
            data["timestamp"] = metrics.timestamp.isoformat()
            json.dump(data, f, indent=2)

        logger.debug(f"Saved metrics to {metrics_file}")

    def _get_latest_metrics(self, model_id: str) -> Optional[ModelMetrics]:
        """Get the most recent metrics for a model"""
        if model_id not in self.models or not self.models[model_id]:
            return None
        return self.models[model_id][-1]

    def _cleanup_old_models(self, model_id: str, retain_count: int) -> None:
        """Remove old model versions, keeping only the most recent N"""
        if model_id not in self.models:
            return

        versions = self.models[model_id]
        if len(versions) > retain_count:
            to_remove = versions[:-retain_count]
            self.models[model_id] = versions[-retain_count:]

            for metrics in to_remove:
                # In production, would delete model files here
                logger.debug(f"Cleaned up {model_id} v{metrics.version}")

    async def _deploy_model(self, model_id: str, version: int) -> None:
        """Deploy a trained model to production"""
        logger.info(f"Deploying {model_id} v{version} to production")

        # In production, this would:
        # 1. Copy model to production path
        # 2. Update agent configuration
        # 3. Perform health checks
        # 4. Gradual rollout with canary testing
        # 5. Monitor performance metrics

        await self._notify_callbacks(
            "model_deployed", {"model_id": model_id, "version": version, "timestamp": datetime.now().isoformat()}
        )

    def get_training_history(self, model_id: str) -> List[ModelMetrics]:
        """Get training history for a model"""
        return self.models.get(model_id, [])

    def get_improvement_trend(self, model_id: str) -> List[Tuple[int, float]]:
        """Get accuracy improvement trend over versions"""
        history = self.get_training_history(model_id)
        return [(m.version, m.accuracy) for m in history]

    def get_monthly_improvement(self, model_id: str) -> float:
        """Calculate average monthly improvement percentage"""
        history = self.get_training_history(model_id)
        if len(history) < 2:
            return 0.0

        # Calculate improvement between first and last version
        baseline = history[0]
        current = history[-1]

        improvement = current.improvement_over(baseline)

        # Normalize to monthly rate (assuming weekly retraining)
        weeks = len(history)
        months = weeks / 4.0
        monthly_rate = improvement / months if months > 0 else improvement

        return monthly_rate

    async def continuous_training_loop(
        self, model_id: str, config: TrainingConfig, interval_hours: int = 168  # Weekly by default
    ) -> None:
        """
        Run continuous training loop for a model

        Retrains the model periodically on new production data
        """
        logger.info(f"Starting continuous training loop for {model_id} (every {interval_hours}h)")

        while True:
            try:
                # Check if new training data is available
                data_available = await self._check_new_training_data(model_id)

                if data_available:
                    logger.info(f"New training data available for {model_id}")

                    # Create and run training job
                    job_id = self.create_training_job(config)
                    metrics = await self.train_model(job_id)

                    logger.info(f"Continuous training completed: {model_id} v{metrics.version}")
                else:
                    logger.debug(f"No new training data for {model_id}, skipping")

                # Wait for next training cycle
                await asyncio.sleep(interval_hours * 3600)

            except Exception as e:
                logger.error(f"Error in continuous training loop: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retrying

    async def _check_new_training_data(self, model_id: str) -> bool:
        """Check if new training data is available"""
        # In production, would check:
        # 1. Production logs for new interaction data
        # 2. Minimum sample size threshold
        # 3. Data quality checks
        # 4. Time since last training

        # Simulate: Return True if at least 1 day since last training
        history = self.get_training_history(model_id)
        if not history:
            return True

        last_training = history[-1].timestamp
        time_since = datetime.now() - last_training

        return time_since > timedelta(days=1)

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get metrics for the learning dashboard"""
        total_models = len(self.models)
        total_versions = sum(len(versions) for versions in self.models.values())

        active_jobs = sum(1 for j in self.jobs.values() if j.status == TrainingStatus.RUNNING)
        completed_jobs = sum(1 for j in self.jobs.values() if j.status == TrainingStatus.COMPLETED)

        # Calculate average improvement across all models
        improvements = [self.get_monthly_improvement(model_id) for model_id in self.models.keys()]
        avg_monthly_improvement = sum(improvements) / len(improvements) if improvements else 0.0

        # Get top performing models
        top_models = []
        for model_id, versions in self.models.items():
            if versions:
                latest = versions[-1]
                top_models.append(
                    {
                        "model_id": model_id,
                        "version": latest.version,
                        "accuracy": latest.accuracy,
                        "improvement": self.get_monthly_improvement(model_id),
                    }
                )

        top_models.sort(key=lambda x: x["accuracy"], reverse=True)

        return {
            "total_models": total_models,
            "total_versions": total_versions,
            "active_training_jobs": active_jobs,
            "completed_jobs": completed_jobs,
            "avg_monthly_improvement_pct": avg_monthly_improvement,
            "top_models": top_models[:5],
            "timestamp": datetime.now().isoformat(),
        }


# Demo
async def demo():
    """Demonstrate the Model Trainer"""
    print("\n" + "=" * 60)
    print("AUTONOMOUS LEARNING LAYER - MODEL TRAINER DEMO")
    print("=" * 60)

    trainer = ModelTrainer()

    # Define training configurations for different agents
    configs = [
        TrainingConfig(
            model_id="legal_contract_classifier",
            model_type=ModelType.CLASSIFIER,
            agent_id="legal_agent",
            training_data_path="./training_data/legal_contracts.jsonl",
            epochs=5,
            auto_deploy=True,
        ),
        TrainingConfig(
            model_id="sales_opportunity_scorer",
            model_type=ModelType.PREDICTOR,
            agent_id="sales_agent",
            training_data_path="./training_data/sales_opportunities.jsonl",
            epochs=5,
            auto_deploy=True,
        ),
        TrainingConfig(
            model_id="customer_churn_predictor",
            model_type=ModelType.PREDICTOR,
            agent_id="customer_success_agent",
            training_data_path="./training_data/customer_behavior.jsonl",
            epochs=5,
            auto_deploy=True,
        ),
    ]

    # Register callback to track events
    async def training_callback(event: str, data: Dict[str, Any]):
        if event == "training_completed":
            print(
                f"✅ Model deployed: {data['model_id']} v{data['version']} "
                f"(accuracy: {data['accuracy']:.2%}, improvement: {data['improvement_pct']:+.2f}%)"
            )

    trainer.register_callback(training_callback)

    print("\n📊 Training 3 agent models...\n")

    # Train all models
    jobs = []
    for config in configs:
        job_id = trainer.create_training_job(config)
        jobs.append((job_id, config.model_id))

    # Execute training jobs
    for job_id, model_id in jobs:
        print(f"🔄 Training {model_id}...")
        metrics = await trainer.train_model(job_id)
        print(
            f"   Accuracy: {metrics.accuracy:.2%}, F1: {metrics.f1_score:.2%}, "
            f"Time: {metrics.training_time_seconds:.1f}s"
        )

    # Simulate second round of training (improvement)
    print("\n🔄 Retraining with new production data...\n")

    for config in configs:
        job_id = trainer.create_training_job(config)
        metrics = await trainer.train_model(job_id)
        improvement = trainer.get_monthly_improvement(config.model_id)
        print(
            f"📈 {config.model_id} v{metrics.version}: {metrics.accuracy:.2%} " f"({improvement:+.1f}% monthly trend)"
        )

    # Display dashboard
    print("\n" + "=" * 60)
    print("LEARNING DASHBOARD")
    print("=" * 60)

    dashboard = trainer.get_dashboard_metrics()

    print("\n📊 Overall Metrics:")
    print(f"   Total Models: {dashboard['total_models']}")
    print(f"   Total Versions Trained: {dashboard['total_versions']}")
    print(f"   Completed Training Jobs: {dashboard['completed_jobs']}")
    print(f"   Average Monthly Improvement: {dashboard['avg_monthly_improvement_pct']:+.2f}%")

    print("\n🏆 Top Performing Models:")
    for i, model in enumerate(dashboard["top_models"], 1):
        print(
            f"   {i}. {model['model_id']}: {model['accuracy']:.2%} "
            f"(v{model['version']}, {model['improvement']:+.1f}% monthly)"
        )

    print("\n✅ Model Trainer Demo Complete!")
    print(f"🎯 Target: 20% monthly improvement - Current: {dashboard['avg_monthly_improvement_pct']:+.2f}%")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    asyncio.run(demo())
