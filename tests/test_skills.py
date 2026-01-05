"""
Tests for Skill System
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from framework.skills.base_skill import BaseSkill
from framework.skills.registry import SkillRegistry
from framework.skills.resolver import SkillResolver

# Mock Skill
class MockSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "mock-skill"

    @property
    def description(self) -> str:
        return "A mock skill for testing."

    @property
    def parameters(self) -> dict:
        return {"param": "value"}

    async def execute(self, **kwargs) -> dict:
        return {"result": "success", "kwargs": kwargs}

@pytest.fixture
def registry():
    reg = SkillRegistry()
    reg.register(MockSkill)
    return reg

def test_registry_registration(registry):
    assert registry.get_skill("mock-skill") is not None
    assert len(registry.list_skills()) == 1
    assert registry.list_skills()[0]["name"] == "mock-skill"

def test_registry_get_unknown(registry):
    assert registry.get_skill("unknown") is None

@pytest.mark.asyncio
async def test_resolver_match():
    # Mock LLM Client
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value={
        "text": '{"skill": "mock-skill", "parameters": {"test": "value"}}'
    })

    resolver = SkillResolver(llm_client=mock_llm)
    # Inject our local registry into the resolver (usually it uses global)
    resolver.registry.register(MockSkill)

    skill_name, params = await resolver.resolve("run the mock skill")

    assert skill_name == "mock-skill"
    assert params == {"test": "value"}

@pytest.mark.asyncio
async def test_resolver_no_match():
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value={
        "text": '{"skill": "none"}'
    })

    resolver = SkillResolver(llm_client=mock_llm)
    resolver.registry.register(MockSkill)

    skill_name, params = await resolver.resolve("do something unrelated")

    assert skill_name is None
    assert params == {}

@pytest.mark.asyncio
async def test_resolver_json_parsing_error():
    mock_llm = MagicMock()
    mock_llm.generate = AsyncMock(return_value={
        "text": 'Invalid JSON'
    })

    resolver = SkillResolver(llm_client=mock_llm)
    resolver.registry.register(MockSkill)

    skill_name, params = await resolver.resolve("break it")

    assert skill_name is None
    assert params == {}
