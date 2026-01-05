"""
Skill Resolver

This module handles the "Natural Language -> Skill" resolution.
It uses an LLM to analyze the user's intent and select the appropriate skill,
extracting the necessary parameters.
"""

import json
import logging
from typing import Dict, Any, Optional, Tuple, List

from framework.llms.multi_llm_client import MultiLLMClient, LLMProvider
from framework.skills.registry import get_registry

logger = logging.getLogger(__name__)

class SkillResolver:
    """
    Resolves natural language queries to specific Skills and their parameters.
    """

    def __init__(self, llm_client: Optional[MultiLLMClient] = None):
        self.llm_client = llm_client or MultiLLMClient()
        self.registry = get_registry()

    async def resolve(self, query: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Analyze the query and return the matched skill name and parameters.

        Returns:
            Tuple(skill_name, parameters_dict)
            Returns (None, {}) if no skill matches or if intent is unclear.
        """
        available_skills = self.registry.list_skills()

        if not available_skills:
            logger.warning("No skills registered.")
            return None, {}

        # Construct the prompt for the LLM
        system_prompt = self._build_system_prompt(available_skills)

        # We use a lower temperature for deterministic routing
        try:
            response = await self.llm_client.generate(
                prompt=query,
                system_prompt=system_prompt,
                provider=LLMProvider.ANTHROPIC, # Prefer Anthropic for complex reasoning
                temperature=0.0
            )

            text_response = response["text"]
            # Extract JSON from response (assuming the model follows instructions)
            # We look for a JSON block

            parsed = self._parse_response(text_response)
            if not parsed:
                return None, {}

            skill_name = parsed.get("skill")
            parameters = parsed.get("parameters", {})

            if skill_name == "none" or not skill_name:
                return None, {}

            return skill_name, parameters

        except Exception as e:
            logger.error(f"Skill resolution failed: {e}")
            return None, {}

    def _build_system_prompt(self, skills: List[Dict[str, Any]]) -> str:
        """
        Build the system prompt listing available skills.
        """
        skills_json = json.dumps(skills, indent=2)

        return f"""You are the Skill Resolver for the Princeps AI Platform.
Your goal is to map a user's natural language request to one of the available Skills.

AVAILABLE SKILLS:
{skills_json}

INSTRUCTIONS:
1. Analyze the user's request.
2. Determine if it matches one of the available skills.
3. If it matches, extract the parameters required by that skill from the request.
4. If a required parameter is missing, try to infer it or leave it null (the skill will handle it).
5. If the request does not match any skill, return "skill": "none".

OUTPUT FORMAT:
You must output ONLY a valid JSON object. Do not include markdown formatting or explanations.
{{
  "skill": "<skill_name>",
  "parameters": {{
    "<param_name>": "<param_value>",
    ...
  }}
}}
"""

    def _parse_response(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Parse the LLM response to extraction JSON.
        Handles potential Markdown wrapping (```json ... ```).
        """
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from resolver response: {text}")
            return None
