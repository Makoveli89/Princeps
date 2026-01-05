"""
Base Skill Definition

This module defines the abstract base class for all Skills in Princeps.
A Skill is a specialized capability that can be activated via natural language.
It acts as a wrapper around one or more Agents, Workflows, or Tools.

Key components:
- Name and Description: For the Resolver to identify the skill.
- Parameters: Schema for the arguments the skill accepts.
- Execution Logic: The actual implementation of the skill.

Design inspired by 'claude-flow' Skills System.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

class SkillParameter(BaseModel):
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None

class BaseSkill(ABC):
    """
    Abstract base class for a Skill.
    """

    def __init__(self, context: Optional[Dict[str, Any]] = None):
        self.context = context or {}

    @property
    @abstractmethod
    def name(self) -> str:
        """The unique name of the skill (e.g., 'code-review')."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """A natural language description of what the skill does."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """
        JSON Schema defining the parameters this skill accepts.
        Used by the Resolver to extract arguments from user input.
        """
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the skill with the provided arguments.

        Returns:
            Dict containing the execution result.
        """
        pass
