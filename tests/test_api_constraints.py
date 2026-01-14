"""Tests for API schema constraints."""
import pytest
from pydantic import ValidationError
from server import CreateWorkspaceRequest, RunRequest, SkillRunRequest

class TestApiSchemaConstraints:
    """Tests for API Pydantic model constraints."""

    def test_create_workspace_request_constraints(self):
        """Test constraints for CreateWorkspaceRequest."""

        # Valid request
        req = CreateWorkspaceRequest(name="Valid Name", description="Valid Description")
        assert req.name == "Valid Name"

        # Name too long
        with pytest.raises(ValidationError):
            CreateWorkspaceRequest(name="a" * 51, description="desc")

        # Name too short
        with pytest.raises(ValidationError):
            CreateWorkspaceRequest(name="", description="desc")

        # Description too long
        with pytest.raises(ValidationError):
            CreateWorkspaceRequest(name="name", description="a" * 201)

    def test_run_request_constraints(self):
        """Test constraints for RunRequest."""

        # Valid request
        req = RunRequest(agentId="agent1", input="some input", workspaceId="ws1")
        assert req.agentId == "agent1"

        # Input too long
        with pytest.raises(ValidationError):
            RunRequest(
                agentId="agent1",
                input="a" * 100001,
                workspaceId="ws1"
            )

        # Agent ID too long
        with pytest.raises(ValidationError):
            RunRequest(
                agentId="a" * 101,
                input="input",
                workspaceId="ws1"
            )

    def test_skill_run_request_constraints(self):
        """Test constraints for SkillRunRequest."""

        # Valid request
        req = SkillRunRequest(query="valid query", workspaceId="ws1")
        assert req.query == "valid query"

        # Query too long
        with pytest.raises(ValidationError):
            SkillRunRequest(
                query="a" * 10001,
                workspaceId="ws1"
            )
