"""
Reinforcement Learning Agent

Enables agent behavior optimization through rewards, adaptive learning from
developer feedback, and self-improving capabilities.

Source: F:\Mothership-main\Mothership-main\agents\reinforcement_learning.py
Recycled: December 26, 2024

Based on PRD: Reinforcement Learning ($1,200 budget)
Target: 40% improvement in agent performance, personalized to team styles
"""

import logging
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)


@dataclass
class State:
    """Agent state representation"""

    context: str
    features: Dict[str, Any]
    timestamp: datetime


@dataclass
class Action:
    """Agent action"""

    name: str
    parameters: Dict[str, Any]
    confidence: float


@dataclass
class Reward:
    """Reward signal"""

    value: float
    source: str  # user_feedback, metric, auto
    metadata: Dict[str, Any]


@dataclass
class Experience:
    """Experience tuple for learning"""

    state: State
    action: Action
    reward: Reward
    next_state: State


class ReinforcementLearning:
    """
    Reinforcement Learning Agent

    Features:
    - Q-learning for policy optimization
    - Reward shaping from user feedback
    - Adaptive behavior based on preferences
    - Self-improvement through experience
    - Multi-armed bandit for exploration
    """

    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9, epsilon: float = 0.1):
        """
        Initialize RL agent

        Args:
            learning_rate: How quickly to update Q-values
            discount_factor: Weight of future rewards
            epsilon: Exploration rate (epsilon-greedy)
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        # Q-table: state-action values
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # Experience replay buffer
        self.experience_buffer = deque(maxlen=10000)

        # Action statistics
        self.action_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "total_reward": 0.0, "avg_reward": 0.0, "success_count": 0, "success_rate": 0.0}
        )

        # Learning statistics
        self.learning_stats = {
            "episodes": 0,
            "total_reward": 0.0,
            "avg_reward": 0.0,
            "exploration_rate": epsilon,
            "q_table_size": 0,
            "experiences_collected": 0,
        }

        # Available actions (can be extended)
        self.action_space = ["suggest_code", "refactor", "optimize", "document", "test", "review", "explain"]

        logger.info("✅ ReinforcementLearning initialized")
        logger.info(f"   Learning rate: {learning_rate}")
        logger.info(f"   Discount factor: {discount_factor}")
        logger.info(f"   Epsilon: {epsilon}")
        logger.info(f"   Action space: {len(self.action_space)} actions")

    def select_action(self, state: State, explore: bool = True) -> Action:
        """
        Select action using epsilon-greedy policy

        Args:
            state: Current state
            explore: Whether to explore (vs exploit)

        Returns:
            Selected action
        """
        state_key = self._state_to_key(state)

        # Epsilon-greedy exploration
        if explore and random.random() < self.epsilon:
            # Explore: random action
            action_name = random.choice(self.action_space)
            confidence = 0.5
            logger.debug(f"🎲 Exploring: {action_name}")
        else:
            # Exploit: best known action
            action_name, confidence = self._get_best_action(state_key)
            logger.debug(f"🎯 Exploiting: {action_name} (Q={confidence:.2f})")

        # Update action stats
        self.action_stats[action_name]["count"] += 1

        return Action(name=action_name, parameters={}, confidence=confidence)

    def learn(self, state: State, action: Action, reward: Reward, next_state: State):
        """
        Update Q-values based on experience

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        # Store experience
        experience = Experience(state, action, reward, next_state)
        self.experience_buffer.append(experience)
        self.learning_stats["experiences_collected"] += 1

        # Update Q-value
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)

        # Q-learning update rule
        old_q = self.q_table[state_key][action.name]
        next_max_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0.0

        new_q = old_q + self.learning_rate * (reward.value + self.discount_factor * next_max_q - old_q)

        self.q_table[state_key][action.name] = new_q

        # Update action statistics
        stats = self.action_stats[action.name]
        stats["total_reward"] += reward.value
        stats["avg_reward"] = stats["total_reward"] / max(1, stats["count"])

        if reward.value > 0:
            stats["success_count"] += 1
        stats["success_rate"] = stats["success_count"] / max(1, stats["count"])

        # Update learning stats
        self.learning_stats["episodes"] += 1
        self.learning_stats["total_reward"] += reward.value
        self.learning_stats["avg_reward"] = self.learning_stats["total_reward"] / max(
            1, self.learning_stats["episodes"]
        )
        self.learning_stats["q_table_size"] = sum(len(actions) for actions in self.q_table.values())

        logger.info(f"📚 Learned: {action.name} | Reward: {reward.value:.2f} | Q: {old_q:.2f} → {new_q:.2f}")

    def get_feedback(self, action: Action, feedback_type: str, value: float = None) -> Reward:
        """
        Process user feedback into reward signal

        Args:
            action: Action that was taken
            feedback_type: Type of feedback (accept, reject, modify, ignore)
            value: Optional explicit reward value

        Returns:
            Reward signal
        """
        # Map feedback to reward
        if value is not None:
            reward_value = value
        else:
            feedback_rewards = {"accept": 1.0, "modify": 0.5, "reject": -0.5, "ignore": -0.1}
            reward_value = feedback_rewards.get(feedback_type, 0.0)

        reward = Reward(
            value=reward_value, source="user_feedback", metadata={"feedback_type": feedback_type, "action": action.name}
        )

        logger.info(f"👤 Feedback: {feedback_type} → Reward: {reward_value:.2f}")

        return reward

    def adapt_to_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """
        Adapt behavior based on user preferences

        Args:
            user_id: User identifier
            preferences: User preferences (style, verbosity, etc.)
        """
        # Adjust exploration rate based on user experience
        if preferences.get("expert_mode", False):
            self.epsilon = max(0.05, self.epsilon * 0.9)  # Less exploration for experts
        else:
            self.epsilon = min(0.3, self.epsilon * 1.1)  # More exploration for novices

        # Boost Q-values for preferred actions
        preferred_actions = preferences.get("preferred_actions", [])
        for action in preferred_actions:
            if action in self.action_space:
                # Small boost to Q-values
                for state_key in self.q_table:
                    if action in self.q_table[state_key]:
                        self.q_table[state_key][action] += 0.1

        self.learning_stats["exploration_rate"] = self.epsilon

        logger.info(f"⚙️  Adapted to {user_id}: ε={self.epsilon:.2f}")

    def optimize_policy(self, num_iterations: int = 100, batch_size: int = 32):
        """
        Optimize policy using experience replay

        Args:
            num_iterations: Number of training iterations
            batch_size: Size of experience batch
        """
        if len(self.experience_buffer) < batch_size:
            logger.warning(f"Not enough experiences ({len(self.experience_buffer)} < {batch_size})")
            return

        for i in range(num_iterations):
            # Sample random batch
            batch = random.sample(self.experience_buffer, batch_size)

            for exp in batch:
                # Re-learn from experience
                state_key = self._state_to_key(exp.state)
                next_state_key = self._state_to_key(exp.next_state)

                old_q = self.q_table[state_key][exp.action.name]
                next_max_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0.0

                new_q = old_q + self.learning_rate * (exp.reward.value + self.discount_factor * next_max_q - old_q)

                self.q_table[state_key][exp.action.name] = new_q

        logger.info(f"🔧 Optimized policy: {num_iterations} iterations on {batch_size}-sample batches")

    def _state_to_key(self, state: State) -> str:
        """Convert state to hashable key"""
        # Simple hashing based on context
        return f"{state.context}_{hash(frozenset(state.features.items()))}"

    def _get_best_action(self, state_key: str) -> Tuple[str, float]:
        """Get best action for state"""
        if state_key not in self.q_table or not self.q_table[state_key]:
            # No experience with this state, random action
            return random.choice(self.action_space), 0.5

        # Get action with highest Q-value
        best_action = max(self.q_table[state_key].items(), key=lambda x: x[1])
        return best_action[0], best_action[1]

    def get_policy(self) -> Dict[str, str]:
        """Get current policy (best action for each state)"""
        policy = {}

        for state_key in self.q_table:
            best_action, _ = self._get_best_action(state_key)
            policy[state_key] = best_action

        return policy

    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        return {
            **self.learning_stats,
            "action_stats": dict(self.action_stats),
            "buffer_size": len(self.experience_buffer),
        }


# Demo usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    rl_agent = ReinforcementLearning(learning_rate=0.1, discount_factor=0.9, epsilon=0.2)

    print("\n" + "=" * 80)
    print("REINFORCEMENT LEARNING - DEMO")
    print("=" * 80 + "\n")

    # Simulate learning episodes
    print("Example 1: Learning from Experience")
    print("-" * 80)

    contexts = ["code_completion", "refactoring", "documentation", "testing"]

    for episode in range(20):
        # Random state
        context = random.choice(contexts)
        state = State(context=context, features={"complexity": random.randint(1, 10)}, timestamp=datetime.now())

        # Select action
        action = rl_agent.select_action(state)

        # Simulate reward based on context
        if context == "code_completion" and action.name == "suggest_code":
            reward_value = 1.0
        elif context == "refactoring" and action.name == "refactor":
            reward_value = 1.0
        elif context == "documentation" and action.name == "document":
            reward_value = 1.0
        else:
            reward_value = random.choice([0.0, -0.2, 0.3])

        reward = Reward(value=reward_value, source="auto", metadata={"episode": episode})

        # Next state
        next_state = State(
            context=random.choice(contexts), features={"complexity": random.randint(1, 10)}, timestamp=datetime.now()
        )

        # Learn
        rl_agent.learn(state, action, reward, next_state)

    print("\nCompleted 20 learning episodes")
    print(f"Exploration rate: {rl_agent.epsilon:.2f}")
    print(f"Q-table size: {rl_agent.learning_stats['q_table_size']} entries")
    print(f"Average reward: {rl_agent.learning_stats['avg_reward']:.2f}")

    print("\n" + "=" * 80 + "\n")

    # Test learned policy
    print("Example 2: Testing Learned Policy")
    print("-" * 80)

    test_contexts = ["code_completion", "refactoring", "documentation"]

    for context in test_contexts:
        state = State(context=context, features={"complexity": 5}, timestamp=datetime.now())

        action = rl_agent.select_action(state, explore=False)  # Pure exploitation
        print(f"{context} → {action.name} (confidence: {action.confidence:.2f})")

    print("\n" + "=" * 80 + "\n")

    # User feedback simulation
    print("Example 3: Learning from User Feedback")
    print("-" * 80)

    state = State(context="code_completion", features={}, timestamp=datetime.now())

    action = rl_agent.select_action(state)
    print(f"Agent suggests: {action.name}")

    # Simulate user accepting suggestion
    feedback = rl_agent.get_feedback(action, "accept")
    next_state = State(context="code_completion", features={}, timestamp=datetime.now())
    rl_agent.learn(state, action, feedback, next_state)

    print(f"User feedback: ACCEPT → Reward: {feedback.value}")

    # Try again with rejection
    action2 = rl_agent.select_action(state)
    feedback2 = rl_agent.get_feedback(action2, "reject")
    rl_agent.learn(state, action2, feedback2, next_state)

    print(f"User feedback: REJECT → Reward: {feedback2.value}")

    print("\n" + "=" * 80 + "\n")

    # Adaptation example
    print("Example 4: Adapting to User Preferences")
    print("-" * 80)

    print(f"Initial exploration rate: {rl_agent.epsilon:.2f}")

    # Expert user preferences
    expert_prefs = {"expert_mode": True, "preferred_actions": ["optimize", "refactor"]}

    rl_agent.adapt_to_preferences("expert_user", expert_prefs)
    print(f"After expert adaptation: {rl_agent.epsilon:.2f}")

    # Novice user preferences
    novice_prefs = {"expert_mode": False, "preferred_actions": ["explain", "document"]}

    rl_agent.adapt_to_preferences("novice_user", novice_prefs)
    print(f"After novice adaptation: {rl_agent.epsilon:.2f}")

    print("\n" + "=" * 80 + "\n")

    # Print statistics
    stats = rl_agent.get_stats()
    print("STATISTICS")
    print("=" * 80)

    print(f"Episodes: {stats['episodes']}")
    print(f"Total Reward: {stats['total_reward']:.2f}")
    print(f"Average Reward: {stats['avg_reward']:.2f}")
    print(f"Exploration Rate: {stats['exploration_rate']:.2f}")
    print(f"Q-table Size: {stats['q_table_size']}")
    print(f"Buffer Size: {stats['buffer_size']}")

    print("\nAction Statistics:")
    for action, action_stat in stats["action_stats"].items():
        if action_stat["count"] > 0:
            print(f"  {action}:")
            print(f"    Count: {action_stat['count']}")
            print(f"    Avg Reward: {action_stat['avg_reward']:.2f}")
            print(f"    Success Rate: {action_stat['success_rate']:.1%}")
