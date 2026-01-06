"""
Skill Registry

Manages the registration and retrieval of available Skills.
"""

from typing import Any

from framework.skills.base_skill import BaseSkill


class SkillRegistry:
    def __init__(self):
        self._skills: dict[str, type[BaseSkill]] = {}

    def register(self, skill_cls: type[BaseSkill]):
        """Register a new skill class."""
        # Instantiate once to get name/properties or just store the class
        # Storing class allows creating fresh instances per request
        temp_instance = skill_cls()
        if temp_instance.name in self._skills:
            # Overwrite or warn? Let's overwrite for now.
            pass
        self._skills[temp_instance.name] = skill_cls

    def get_skill(self, name: str) -> type[BaseSkill] | None:
        """Get a skill class by name."""
        return self._skills.get(name)

    def list_skills(self) -> list[dict[str, Any]]:
        """List all registered skills with their metadata."""
        skills_metadata = []
        for name, cls in self._skills.items():
            instance = cls()
            skills_metadata.append(
                {
                    "name": instance.name,
                    "description": instance.description,
                    "parameters": instance.parameters,
                }
            )
        return skills_metadata


# Global Registry Instance
registry = SkillRegistry()


def register_skill(skill_cls: type[BaseSkill]):
    """Decorator to register a skill."""
    registry.register(skill_cls)
    return skill_cls


def get_registry() -> SkillRegistry:
    return registry
