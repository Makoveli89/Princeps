import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

# Note: This test mocks core modules globally. Run it in isolation or it may affect other tests.
# pytest tests/test_security_limits.py

# Add root to path
sys.path.append(os.getcwd())

# Mock dependencies to avoid DB/LLM connection requirements
# We use a try-except block to only mock if not already imported (to avoid overwriting real modules if run in suite)
# However, for this test to work standalone, we force mocks.
# To be safe in CI, we skip this test by default unless enabled manually or if modules are missing.


@pytest.mark.skip(
    reason="Mocks global modules, run in isolation: pytest tests/test_security_limits.py"
)
def test_security_limits_placeholders():
    pass


# We define the actual logic but only execute if we can setup mocks safely
# or if we are running this file specifically.


def setup_mocks():
    modules_to_mock = [
        "brain.core.db",
        "brain.core.models",
        "framework.agents.example_agent",
        "framework.ingestion.service",
        "framework.llms.multi_llm_client",
        "framework.retrieval.vector_search",
        "framework.skills.registry",
        "framework.skills.resolver",
    ]
    for mod in modules_to_mock:
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()


# Only run setup if we are main or if we decide to force it.
# For now, we will just implement the test logic using `server` if it can be imported.

try:
    setup_mocks()
    from server import agent_manager, app, ingestion_service

    client = TestClient(app)
except ImportError:
    client = None


@pytest.mark.skipif(client is None, reason="Could not initialize client with mocks")
def test_large_input_validation():
    """
    Test that inputs exceeding the character limit are rejected (422).
    """
    # 100,001 characters
    large_input = "a" * 100001

    payload = {"agentId": "test-agent", "input": large_input, "workspaceId": "test-workspace"}

    # Mock the manager to avoid actual execution
    if hasattr(agent_manager, "run_agent"):
        agent_manager.run_agent = AsyncMock(return_value={"status": "mocked"})

    response = client.post("/api/run", json=payload)

    assert response.status_code == 422, "Should reject inputs > 100k chars"


@pytest.mark.skipif(client is None, reason="Could not initialize client with mocks")
def test_large_file_upload():
    """
    Test that files larger than 10MB are rejected (413).
    """
    # 11MB dummy content
    large_content = b"0" * (11 * 1024 * 1024)

    # Mock ingestion service
    if hasattr(ingestion_service, "ingest_file"):
        ingestion_service.ingest_file = AsyncMock(return_value={"status": "mocked"})

    files = {"file": ("large_file.txt", large_content, "text/plain")}
    data = {"workspace_id": "test-workspace"}

    response = client.post("/api/ingest", files=files, data=data)

    assert response.status_code == 413, "Should reject files > 10MB"
