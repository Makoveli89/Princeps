"""
Research Skill

A skill that performs deep research on a topic.
"""

from typing import Any, Dict
from framework.skills.base_skill import BaseSkill
from framework.skills.registry import register_skill
from framework.llms.multi_llm_client import MultiLLMClient, LLMProvider

@register_skill
class ResearchSkill(BaseSkill):

    @property
    def name(self) -> str:
        return "deep-research"

    @property
    def description(self) -> str:
        return "Conducts deep research on a specific topic, breaking it down into sub-questions."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The research topic or question."
                },
                "depth": {
                    "type": "string",
                    "description": "Depth of research (brief, detailed, comprehensive).",
                    "enum": ["brief", "detailed", "comprehensive"],
                    "default": "detailed"
                }
            },
            "required": ["topic"]
        }

    async def execute(self, **kwargs) -> Dict[str, Any]:
        topic = kwargs.get("topic")
        depth = kwargs.get("depth", "detailed")

        if not topic:
            return {"error": "No topic provided."}

        llm_client = MultiLLMClient()

        prompt = f"""Conduct a {depth} research analysis on: "{topic}".

Please structure your response as follows:
1. Executive Summary
2. Key Concepts & Definitions
3. Detailed Analysis
4. Controversies or Debates (if any)
5. Future Outlook
6. Conclusion
"""

        if depth == "comprehensive":
            prompt += "\nInclude extensive details, historical context, and multiple perspectives."

        response = await llm_client.generate(
            prompt=prompt,
            provider=LLMProvider.ANTHROPIC,
            temperature=0.7 # Higher creativity for research
        )

        return {
            "status": "completed",
            "report": response["text"],
            "topic": topic
        }
