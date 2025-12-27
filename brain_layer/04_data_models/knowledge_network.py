"""
Knowledge Network - Cross-Agent Intelligence Sharing

Enables agents to automatically share learnings, insights, and solutions,
creating compound intelligence across the entire ecosystem.
"""

import asyncio
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


class KnowledgeType(Enum):
    """Types of knowledge that can be shared"""

    SOLUTION = "solution"  # Problem solution
    PATTERN = "pattern"  # Recurring pattern
    INSIGHT = "insight"  # Strategic insight
    BEST_PRACTICE = "best_practice"  # Proven approach
    FAILURE = "failure"  # What didn't work (negative knowledge)
    OPTIMIZATION = "optimization"  # Performance improvement


class ShareScope(Enum):
    """Scope of knowledge sharing"""

    PRIVATE = "private"  # Agent-specific
    TEAM = "team"  # Team of related agents
    PUBLIC = "public"  # All agents


@dataclass
class KnowledgeNode:
    """A piece of knowledge in the network"""

    node_id: str
    agent_id: str  # Source agent
    knowledge_type: KnowledgeType
    title: str
    content: str
    tags: List[str]

    # Metadata
    created_at: datetime
    updated_at: datetime
    version: int = 1

    # Usage tracking
    access_count: int = 0
    success_rate: float = 0.0  # How often it helped
    relevance_score: float = 1.0

    # Sharing
    scope: ShareScope = ShareScope.PUBLIC
    shared_with: List[str] = field(default_factory=list)  # Agent IDs

    # Context
    problem_domain: str = ""
    prerequisites: List[str] = field(default_factory=list)
    related_nodes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        data["knowledge_type"] = self.knowledge_type.value
        data["scope"] = self.scope.value
        return data


@dataclass
class KnowledgeEdge:
    """Connection between knowledge nodes"""

    edge_id: str
    from_node_id: str
    to_node_id: str
    relationship: str  # "builds_on", "contradicts", "enhances", etc.
    strength: float  # 0.0 to 1.0
    created_at: datetime


@dataclass
class KnowledgeFlow:
    """Tracks knowledge transfer between agents"""

    flow_id: str
    from_agent_id: str
    to_agent_id: str
    knowledge_node_id: str
    timestamp: datetime
    outcome: Optional[str] = None  # "success", "failure", "pending"
    impact_score: float = 0.0  # Measured improvement


class KnowledgeNetwork:
    """
    Cross-Agent Knowledge Sharing Network

    Features:
    - Automatic knowledge extraction from agent outputs
    - Semantic search for relevant knowledge
    - Knowledge propagation across agents
    - Impact tracking and ranking
    - Network effects (compound intelligence)
    """

    def __init__(self, network_dir: str = "./knowledge_network", auto_share: bool = True):
        self.network_dir = Path(network_dir)
        self.network_dir.mkdir(parents=True, exist_ok=True)

        # Knowledge graph
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: List[KnowledgeEdge] = []

        # Knowledge flows (transfer history)
        self.flows: List[KnowledgeFlow] = []

        # Agent connections
        self.agent_graph: Dict[str, Set[str]] = {}  # {agent_id: set of connected agents}

        # Auto-sharing enabled
        self.auto_share = auto_share

        # Subscriptions (agent -> knowledge types they want)
        self.subscriptions: Dict[str, List[KnowledgeType]] = {}

        logger.info(f"KnowledgeNetwork initialized with auto_share={auto_share}")

    def connect_agents(self, agent1_id: str, agent2_id: str) -> None:
        """Create a bidirectional connection between agents"""
        if agent1_id not in self.agent_graph:
            self.agent_graph[agent1_id] = set()
        if agent2_id not in self.agent_graph:
            self.agent_graph[agent2_id] = set()

        self.agent_graph[agent1_id].add(agent2_id)
        self.agent_graph[agent2_id].add(agent1_id)

        logger.info(f"Connected {agent1_id} ↔ {agent2_id}")

    def subscribe(self, agent_id: str, knowledge_types: List[KnowledgeType]) -> None:
        """Subscribe an agent to specific knowledge types"""
        self.subscriptions[agent_id] = knowledge_types
        logger.info(f"{agent_id} subscribed to {[kt.value for kt in knowledge_types]}")

    def add_knowledge(
        self,
        agent_id: str,
        knowledge_type: KnowledgeType,
        title: str,
        content: str,
        tags: List[str] = None,
        problem_domain: str = "",
        scope: ShareScope = ShareScope.PUBLIC,
    ) -> str:
        """Add new knowledge to the network"""
        node_id = f"knowledge-{agent_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        node = KnowledgeNode(
            node_id=node_id,
            agent_id=agent_id,
            knowledge_type=knowledge_type,
            title=title,
            content=content,
            tags=tags or [],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            problem_domain=problem_domain,
            scope=scope,
        )

        self.nodes[node_id] = node

        logger.info(f"📚 {agent_id} added knowledge: {title} ({knowledge_type.value})")

        # Auto-share if enabled
        if self.auto_share and scope == ShareScope.PUBLIC:
            asyncio.create_task(self._propagate_knowledge(node_id))

        # Save to disk
        self._save_node(node)

        return node_id

    async def _propagate_knowledge(self, node_id: str) -> None:
        """Propagate knowledge to relevant agents"""
        node = self.nodes.get(node_id)
        if not node:
            return

        # Find agents who would benefit from this knowledge
        recipients = self._find_recipients(node)

        logger.info(f"📡 Propagating {node.title} to {len(recipients)} agents")

        for agent_id in recipients:
            # Record knowledge flow
            flow = KnowledgeFlow(
                flow_id=f"flow-{node_id}-{agent_id}",
                from_agent_id=node.agent_id,
                to_agent_id=agent_id,
                knowledge_node_id=node_id,
                timestamp=datetime.now(),
                outcome="pending",
            )
            self.flows.append(flow)

            # In production, would:
            # 1. Send notification to agent
            # 2. Update agent's knowledge base
            # 3. Track if agent uses this knowledge
            # 4. Measure impact on agent's performance

            node.access_count += 1
            node.shared_with.append(agent_id)

        logger.debug(f"Knowledge {node_id} now accessed {node.access_count} times")

    def _find_recipients(self, node: KnowledgeNode) -> List[str]:
        """Find agents who should receive this knowledge"""
        recipients = []

        # Get connected agents
        connected = self.agent_graph.get(node.agent_id, set())

        for agent_id in connected:
            # Check if agent is subscribed to this type
            if agent_id in self.subscriptions:
                if node.knowledge_type in self.subscriptions[agent_id]:
                    recipients.append(agent_id)
            else:
                # If no specific subscription, share public knowledge
                if node.scope == ShareScope.PUBLIC:
                    recipients.append(agent_id)

        return recipients

    def search_knowledge(
        self, query: str, agent_id: str, knowledge_type: Optional[KnowledgeType] = None, limit: int = 10
    ) -> List[KnowledgeNode]:
        """Search for relevant knowledge"""
        results = []

        for node in self.nodes.values():
            # Check access permissions
            if node.scope == ShareScope.PRIVATE and node.agent_id != agent_id:
                continue

            # Filter by type if specified
            if knowledge_type and node.knowledge_type != knowledge_type:
                continue

            # Simple relevance scoring (in production, use embedding similarity)
            score = 0.0
            query_lower = query.lower()

            if query_lower in node.title.lower():
                score += 2.0
            if query_lower in node.content.lower():
                score += 1.0
            if any(query_lower in tag.lower() for tag in node.tags):
                score += 1.5
            if query_lower in node.problem_domain.lower():
                score += 1.0

            # Boost by success rate and access count
            score *= 1 + node.success_rate
            score *= 1 + min(node.access_count / 100, 1.0)  # Cap boost at 100 accesses

            if score > 0:
                node.relevance_score = score
                results.append(node)

        # Sort by relevance
        results.sort(key=lambda n: n.relevance_score, reverse=True)

        return results[:limit]

    def record_knowledge_usage(self, node_id: str, agent_id: str, success: bool, impact_score: float = 0.0) -> None:
        """Record when an agent uses knowledge"""
        node = self.nodes.get(node_id)
        if not node:
            return

        # Update success rate (exponential moving average)
        alpha = 0.1  # Learning rate
        new_success = 1.0 if success else 0.0
        node.success_rate = alpha * new_success + (1 - alpha) * node.success_rate

        # Find and update corresponding flow
        for flow in self.flows:
            if flow.knowledge_node_id == node_id and flow.to_agent_id == agent_id:
                flow.outcome = "success" if success else "failure"
                flow.impact_score = impact_score
                break

        result = "✅" if success else "❌"
        logger.info(f"{result} {agent_id} used knowledge: {node.title} " f"(success_rate now {node.success_rate:.2%})")

    def link_knowledge(self, from_node_id: str, to_node_id: str, relationship: str, strength: float = 1.0) -> None:
        """Create a relationship between knowledge nodes"""
        edge_id = f"edge-{from_node_id}-{to_node_id}"

        edge = KnowledgeEdge(
            edge_id=edge_id,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            relationship=relationship,
            strength=strength,
            created_at=datetime.now(),
        )

        self.edges.append(edge)

        # Update node relationships
        if from_node_id in self.nodes:
            self.nodes[from_node_id].related_nodes.append(to_node_id)

        logger.debug(f"Linked {from_node_id} --{relationship}--> {to_node_id}")

    def get_agent_knowledge_graph(self, agent_id: str) -> Dict[str, Any]:
        """Get knowledge graph for an agent"""
        # Get all knowledge accessible to this agent
        accessible = [
            node
            for node in self.nodes.values()
            if (node.agent_id == agent_id or node.scope == ShareScope.PUBLIC or agent_id in node.shared_with)
        ]

        # Get knowledge created by this agent
        created = [n for n in accessible if n.agent_id == agent_id]

        # Get knowledge consumed by this agent
        consumed_flows = [f for f in self.flows if f.to_agent_id == agent_id and f.outcome == "success"]
        consumed_node_ids = set(f.knowledge_node_id for f in consumed_flows)
        consumed = [n for n in accessible if n.node_id in consumed_node_ids]

        # Calculate impact
        total_impact = sum(f.impact_score for f in consumed_flows)

        return {
            "agent_id": agent_id,
            "knowledge_created": len(created),
            "knowledge_consumed": len(consumed),
            "knowledge_accessible": len(accessible),
            "total_impact_score": total_impact,
            "connections": len(self.agent_graph.get(agent_id, set())),
            "subscriptions": [kt.value for kt in self.subscriptions.get(agent_id, [])],
        }

    def get_network_stats(self) -> Dict[str, Any]:
        """Get overall network statistics"""
        total_nodes = len(self.nodes)
        total_flows = len(self.flows)
        successful_flows = len([f for f in self.flows if f.outcome == "success"])

        # Calculate network density
        total_agents = len(self.agent_graph)
        possible_connections = total_agents * (total_agents - 1) / 2 if total_agents > 1 else 0
        actual_connections = sum(len(connections) for connections in self.agent_graph.values()) / 2
        network_density = (actual_connections / possible_connections) if possible_connections > 0 else 0

        # Top knowledge by impact
        knowledge_impacts = {}
        for flow in self.flows:
            if flow.outcome == "success":
                knowledge_impacts[flow.knowledge_node_id] = (
                    knowledge_impacts.get(flow.knowledge_node_id, 0) + flow.impact_score
                )

        top_knowledge = sorted(knowledge_impacts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_agents": total_agents,
            "total_knowledge_nodes": total_nodes,
            "total_knowledge_flows": total_flows,
            "successful_flows": successful_flows,
            "success_rate": (successful_flows / total_flows) if total_flows > 0 else 0,
            "network_density": network_density,
            "total_connections": int(actual_connections),
            "top_knowledge": [{"node_id": nid, "total_impact": impact} for nid, impact in top_knowledge],
        }

    def _save_node(self, node: KnowledgeNode) -> None:
        """Save knowledge node to disk"""
        node_file = self.network_dir / f"{node.node_id}.json"

        with open(node_file, "w") as f:
            json.dump(node.to_dict(), f, indent=2)


# Demo
async def demo():
    """Demonstrate the Knowledge Network"""
    print("\n" + "=" * 60)
    print("KNOWLEDGE NETWORK - CROSS-AGENT INTELLIGENCE DEMO")
    print("=" * 60)

    network = KnowledgeNetwork()

    # Setup agent network
    _agents = ["legal_agent", "sales_agent", "customer_success_agent", "cfo_agent", "cro_agent", "marketing_agent"]

    print("\n🔗 Building agent network connections...\n")

    # Connect related agents
    network.connect_agents("legal_agent", "sales_agent")  # Legal reviews contracts
    network.connect_agents("sales_agent", "customer_success_agent")  # Handoff
    network.connect_agents("sales_agent", "cfo_agent")  # Revenue tracking
    network.connect_agents("cro_agent", "sales_agent")  # Revenue leadership
    network.connect_agents("cro_agent", "marketing_agent")  # Revenue operations
    network.connect_agents("cro_agent", "customer_success_agent")  # Retention
    network.connect_agents("marketing_agent", "sales_agent")  # Lead generation

    # Subscribe agents to knowledge types
    network.subscribe("legal_agent", [KnowledgeType.SOLUTION, KnowledgeType.BEST_PRACTICE])
    network.subscribe("sales_agent", [KnowledgeType.PATTERN, KnowledgeType.OPTIMIZATION])
    network.subscribe("customer_success_agent", [KnowledgeType.SOLUTION, KnowledgeType.PATTERN])
    network.subscribe("cfo_agent", [KnowledgeType.INSIGHT, KnowledgeType.OPTIMIZATION])

    print("✅ Connected 6 agents with 7 connections")
    print("✅ Set up knowledge subscriptions\n")

    # Add knowledge from different agents
    print("📚 Agents sharing knowledge...\n")

    # Legal agent shares contract negotiation insight
    node1 = network.add_knowledge(
        agent_id="legal_agent",
        knowledge_type=KnowledgeType.BEST_PRACTICE,
        title="Enterprise Contract Negotiation Strategy",
        content="For contracts >$50K, include performance milestones and early termination clauses to reduce risk",
        tags=["contracts", "enterprise", "risk-management"],
        problem_domain="legal",
    )

    # Sales agent shares deal pattern
    node2 = network.add_knowledge(
        agent_id="sales_agent",
        knowledge_type=KnowledgeType.PATTERN,
        title="Enterprise Deal Acceleration Pattern",
        content="Multi-stakeholder deals close 40% faster when legal is involved early in discovery phase",
        tags=["sales", "enterprise", "velocity"],
        problem_domain="sales",
    )

    # Customer Success shares churn prevention
    node3 = network.add_knowledge(
        agent_id="customer_success_agent",
        knowledge_type=KnowledgeType.SOLUTION,
        title="Proactive Churn Prevention via QBRs",
        content="Quarterly business reviews with C-suite reduce churn by 60% for enterprise accounts",
        tags=["retention", "enterprise", "qbr"],
        problem_domain="customer_success",
    )

    # CFO shares financial optimization
    node4 = network.add_knowledge(
        agent_id="cfo_agent",
        knowledge_type=KnowledgeType.OPTIMIZATION,
        title="Contract Payment Terms Optimization",
        content="Annual upfront payment with 10% discount improves cash flow and reduces churn",
        tags=["finance", "cash-flow", "retention"],
        problem_domain="finance",
    )

    # Wait for propagation
    await asyncio.sleep(0.5)

    print("✅ 4 knowledge nodes created and propagated\n")

    # Agents search for and use knowledge
    print("🔍 Agents discovering relevant knowledge...\n")

    # Sales agent searches for enterprise deal knowledge
    results = network.search_knowledge(query="enterprise deal", agent_id="sales_agent", limit=3)

    print(f"💡 Sales Agent found {len(results)} relevant knowledge items:")
    for i, node in enumerate(results, 1):
        print(f"   {i}. {node.title} (from {node.agent_id})")
        print(f"      Relevance: {node.relevance_score:.2f}")

    # Record usage
    if results:
        network.record_knowledge_usage(
            node_id=results[0].node_id,
            agent_id="sales_agent",
            success=True,
            impact_score=15.0,  # 15% improvement in close rate
        )

    # Legal agent applies sales pattern
    legal_results = network.search_knowledge(query="contract negotiation", agent_id="legal_agent")
    if legal_results:
        network.record_knowledge_usage(
            node_id=legal_results[0].node_id,
            agent_id="legal_agent",
            success=True,
            impact_score=12.0,  # 12% faster contract reviews
        )

    # Link related knowledge
    network.link_knowledge(node1, node2, "enhances", strength=0.8)
    network.link_knowledge(node2, node3, "builds_on", strength=0.9)
    network.link_knowledge(node3, node4, "complements", strength=0.7)

    print("\n🔗 Created knowledge relationships\n")

    # Display network statistics
    print("=" * 60)
    print("NETWORK STATISTICS")
    print("=" * 60)

    stats = network.get_network_stats()

    print("\n🌐 Network Overview:")
    print(f"   Total Agents: {stats['total_agents']}")
    print(f"   Total Knowledge Nodes: {stats['total_knowledge_nodes']}")
    print(f"   Total Connections: {stats['total_connections']}")
    print(f"   Network Density: {stats['network_density']:.1%}")

    print("\n📊 Knowledge Sharing:")
    print(f"   Total Knowledge Flows: {stats['total_knowledge_flows']}")
    print(f"   Successful Applications: {stats['successful_flows']}")
    print(f"   Success Rate: {stats['success_rate']:.1%}")

    # Show individual agent graphs
    print("\n👥 Agent Knowledge Graphs:")
    for agent in ["sales_agent", "legal_agent", "customer_success_agent"]:
        graph = network.get_agent_knowledge_graph(agent)
        print(f"\n   📌 {agent}:")
        print(f"      Created: {graph['knowledge_created']} nodes")
        print(f"      Consumed: {graph['knowledge_consumed']} nodes")
        print(f"      Accessible: {graph['knowledge_accessible']} nodes")
        print(f"      Total Impact: {graph['total_impact_score']:.1f}%")
        print(f"      Connections: {graph['connections']} agents")

    print("\n✅ Knowledge Network Demo Complete!")
    print("🎯 Compound Intelligence: Knowledge shared automatically across all agents")
    print("📈 Network Effects: Each agent's learning benefits the entire system")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    asyncio.run(demo())
