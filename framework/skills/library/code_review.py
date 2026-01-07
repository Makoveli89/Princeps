"""
Code Review Skill

A skill that performs a comprehensive code review on a provided snippet or file.
Simulates the 'github-code-review' skill from claude-flow.
"""

from typing import Any

from framework.agents.base_agent import LLMProvider
from framework.llms.multi_llm_client import MultiLLMClient
from framework.skills.base_skill import BaseSkill
from framework.skills.registry import register_skill

# We will create a specialized agent on the fly or reuse one.
# For this example, let's use the MultiLLMClient directly or a generic agent.


@register_skill
class CodeReviewSkill(BaseSkill):

    @property
    def name(self) -> str:
        return "code-review"

    @property
    def description(self) -> str:
        return "Reviews code for security issues, bugs, and best practices."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The code snippet or file content to review.",
                },
                "focus": {
                    "type": "string",
                    "description": "Specific area to focus on (e.g., 'security', 'performance', 'style').",
                    "enum": ["security", "performance", "style", "general"],
                    "default": "general",
                },
            },
            "required": ["code"],
        }

    async def execute(self, **kwargs) -> dict[str, Any]:
        code = kwargs.get("code")
        focus = kwargs.get("focus", "general")

        if not code:
            return {"error": "No code provided for review."}

        # Initialize LLM Client (in a real app, this might be injected)
        llm_client = MultiLLMClient()

        # Construct Prompt
        prompt = f"""Please perform a {focus} code review on the following code:

```
{code}
```

Identify potential bugs, security vulnerabilities, and improvements.
"""

        # specific instructions based on focus
        if focus == "security":
            prompt += "\nPay special attention to OWASP Top 10 vulnerabilities, injection risks, and data handling."
        elif focus == "performance":
            prompt += "\nFocus on time complexity, memory usage, and optimization opportunities."

        # Call LLM
        response = await llm_client.generate(
            prompt=prompt,
            provider=LLMProvider.ANTHROPIC,  # Claude is good for code
            model="claude-3-5-sonnet-20241022",  # Hardcoded for now, or use config
            temperature=0.2,
        )

        return {"status": "completed", "review": response["text"], "focus": focus}
