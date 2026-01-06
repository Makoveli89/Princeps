"""
Example Agent Implementation

Demonstrates how to create a specialized agent by extending BaseAgent.
This example implements a simple summarization agent.
"""

from typing import Any

from framework.agents.base_agent import (
    AgentResponse,
    AgentTask,
    BaseAgent,
    TaskStatus,
)


class SummarizationAgent(BaseAgent):
    """
    Example agent for text summarization tasks.

    Demonstrates:
    - Implementing abstract methods from BaseAgent
    - Custom system prompts
    - Response processing and validation
    - Fallback handling
    """

    def _initialize_capabilities(self) -> list[str]:
        """Define summarization-specific capabilities"""
        return [
            "text-summarization",
            "key-points-extraction",
            "abstract-generation",
            "bullet-point-summary",
            "executive-summary",
        ]

    def _get_system_prompt(self, task: AgentTask) -> str:
        """Generate system prompt for summarization tasks"""
        summary_type = task.parameters.get("summary_type", "concise")

        prompts = {
            "concise": """You are an expert summarization assistant. Your task is to create
clear, concise summaries that capture the essential information from the provided text.
Focus on the main ideas and key points. Keep the summary brief but comprehensive.""",
            "detailed": """You are an expert summarization assistant. Create detailed summaries
that preserve important nuances and supporting details from the source text.
Include relevant examples and explanations where appropriate.""",
            "bullet": """You are an expert summarization assistant. Create summaries in
bullet-point format, listing the key points and important information.
Use clear, actionable language. Group related points together.""",
            "executive": """You are an expert executive summary writer. Create summaries suitable
for busy executives who need to quickly understand the key points and implications.
Focus on conclusions, recommendations, and actionable insights.""",
        }

        return prompts.get(summary_type, prompts["concise"])

    def _process_response(self, raw_response: str, task: AgentTask) -> dict[str, Any]:
        """Process and validate the summarization response"""
        # Basic validation
        if not raw_response or len(raw_response.strip()) < 10:
            return {
                "text": "",
                "error": "Response too short to be a valid summary",
            }

        # Check for error indicators
        error_indicators = ["i cannot", "i'm unable", "error", "failed"]
        if any(indicator in raw_response.lower()[:100] for indicator in error_indicators):
            return {
                "text": raw_response,
                "warning": "Response may indicate an error or inability to complete the task",
            }

        # Extract structured data if applicable
        structured_output = None
        if task.parameters.get("summary_type") == "bullet":
            # Parse bullet points
            lines = [line.strip() for line in raw_response.split("\n") if line.strip()]
            bullet_points = [
                line.lstrip("•-*").strip()
                for line in lines
                if line.startswith(("•", "-", "*", "·"))
            ]
            if bullet_points:
                structured_output = {"bullet_points": bullet_points}

        return {
            "text": raw_response,
            "structured_output": structured_output,
        }

    def _fallback_handler(self, task: AgentTask, error: str) -> AgentResponse:
        """Provide fallback when all LLMs fail"""
        # Simple extractive fallback - take first few sentences
        text = task.prompt or task.context or ""
        sentences = text.split(".")[:3]
        fallback_summary = ". ".join(sentences).strip()

        if fallback_summary:
            fallback_summary += "..."
        else:
            fallback_summary = "Unable to generate summary."

        return AgentResponse(
            task_id=task.task_id,
            success=False,
            status=TaskStatus.FAILED,
            response_text=fallback_summary,
            error=error,
            error_type="FallbackUsed",
        )


class CodeReviewAgent(BaseAgent):
    """
    Example agent for code review tasks.

    Demonstrates specialized agent for code analysis.
    """

    def _initialize_capabilities(self) -> list[str]:
        """Define code review capabilities"""
        return [
            "code-review",
            "bug-detection",
            "security-analysis",
            "performance-suggestions",
            "style-checking",
            "refactoring-suggestions",
        ]

    def _get_system_prompt(self, task: AgentTask) -> str:
        """Generate system prompt for code review"""
        language = task.parameters.get("language", "python")
        focus = task.parameters.get("focus", "general")

        base_prompt = f"""You are an expert {language} code reviewer.
Analyze the provided code and provide constructive feedback."""

        focus_additions = {
            "security": "\nFocus especially on security vulnerabilities, input validation, "
            "and potential injection attacks.",
            "performance": "\nFocus on performance optimizations, algorithmic efficiency, "
            "and resource management.",
            "style": "\nFocus on code style, readability, naming conventions, "
            "and adherence to best practices.",
            "bugs": "\nFocus on identifying potential bugs, edge cases, and logical errors.",
        }

        return base_prompt + focus_additions.get(focus, "")

    def _process_response(self, raw_response: str, task: AgentTask) -> dict[str, Any]:
        """Process code review response"""
        # Try to extract structured feedback
        structured = {
            "issues": [],
            "suggestions": [],
            "positive_aspects": [],
        }

        lines = raw_response.split("\n")
        current_section = None

        for line in lines:
            line_lower = line.lower()
            if "issue" in line_lower or "problem" in line_lower or "bug" in line_lower:
                current_section = "issues"
            elif "suggest" in line_lower or "recommend" in line_lower:
                current_section = "suggestions"
            elif "good" in line_lower or "well" in line_lower or "positive" in line_lower:
                current_section = "positive_aspects"
            elif current_section and line.strip().startswith(("-", "*", "•")):
                structured[current_section].append(line.strip().lstrip("-*• "))

        return {
            "text": raw_response,
            "structured_output": structured if any(structured.values()) else None,
        }

    def _fallback_handler(self, task: AgentTask, error: str) -> AgentResponse:
        """Fallback for code review"""
        return AgentResponse(
            task_id=task.task_id,
            success=False,
            status=TaskStatus.FAILED,
            response_text="Code review could not be completed. Please try again later.",
            error=error,
            error_type="FallbackUsed",
        )


# Example usage and testing
if __name__ == "__main__":
    import asyncio

    async def test_agents():
        """Test the example agents"""
        print("Testing Example Agents\n" + "=" * 50)

        # Create a summarization agent
        summarizer = SummarizationAgent(
            agent_name="test_summarizer",
            agent_type="summarization",
        )

        print("\nSummarization Agent Capabilities:")
        caps = summarizer.get_capabilities()
        print(f"  - Initialized: {caps['initialized']}")
        print(f"  - Capabilities: {caps['capabilities']}")

        # Create a code review agent
        reviewer = CodeReviewAgent(
            agent_name="test_reviewer",
            agent_type="code_review",
        )

        print("\nCode Review Agent Capabilities:")
        caps = reviewer.get_capabilities()
        print(f"  - Initialized: {caps['initialized']}")
        print(f"  - Capabilities: {caps['capabilities']}")

        # Create sample tasks
        summary_task = summarizer.create_task(
            prompt="The quick brown fox jumps over the lazy dog. This is a sample text.",
            task_type="summarization",
            parameters={"summary_type": "concise"},
        )
        print(f"\nCreated summarization task: {summary_task.task_id}")

        review_task = reviewer.create_task(
            prompt="def foo(x): return x + 1",
            task_type="code_review",
            parameters={"language": "python", "focus": "style"},
        )
        print(f"Created code review task: {review_task.task_id}")

        print("\nNote: Actual LLM calls require a configured MultiLLMClient")

    asyncio.run(test_agents())
