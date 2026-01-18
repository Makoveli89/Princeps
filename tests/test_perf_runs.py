import sys
from unittest.mock import MagicMock

# Mock Agent modules before importing server
modules_to_mock = [
    "framework.agents.concept_agent",
    "framework.agents.entity_agent",
    "framework.agents.example_agent",
    "framework.agents.executor_agent",
    "framework.agents.planner_agent",
    "framework.agents.retriever_agent",
    "framework.agents.topic_agent",
    "framework.ingestion.service"
]

for module in modules_to_mock:
    sys.modules[module] = MagicMock()

import time  # noqa: E402
import uuid  # noqa: E402
from brain.core.models import AgentRun  # noqa: E402

# Now import server
from server import get_runs  # noqa: E402, I001

def test_perf_get_runs(session, tenant):
    # Setup: Insert 500 runs with large data
    # 100KB per text field * 2 = 200KB per row
    # 500 rows * 200KB = 100MB
    large_text = "x" * 100000
    large_json = {"data": "y" * 100000}

    runs = []
    for i in range(500):
        runs.append(AgentRun(
            id=uuid.uuid4(),
            tenant_id=tenant.id,
            agent_id=f"agent-{i}",
            task=large_text,
            solution=large_json,
            success=True,
            duration_ms=100
        ))
    session.add_all(runs)
    session.commit()

    # Measure
    start_time = time.perf_counter()

    # Call the logic directly (mocking db dependency)
    # limit=500 to fetch all
    results = get_runs(workspaceId=str(tenant.id), limit=500, db=session)

    end_time = time.perf_counter()
    duration = end_time - start_time

    print(f"\nTime taken: {duration:.4f} seconds")

    assert len(results) == 500
    # Basic check to ensure data is correct
    assert len(results[0].input_preview) > 0
