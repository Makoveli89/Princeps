"""
LLM Council - Council-of-Experts Mechanisms for Multi-LLM Decision Making

This module encapsulates the council-of-experts pattern for the PlannerAgent,
providing utility functions for:
- Querying multiple LLMs in parallel
- Evaluating and comparing responses
- Ensemble voting and consensus building
- LLM-as-judge pattern for plan evaluation
- Confidence scoring and plan ranking

Strategic Intent:
The LLM Council leverages diverse AI opinions to mitigate individual model
biases or blind spots. By combining perspectives from multiple LLMs, the
system produces more reliable and well-rounded plans.

Adapted from patterns in:
- reinforcement_learning.py: Action scoring and selection mechanisms
- ab_testing.py: Comparing variants and statistical analysis
- multi_llm_client.py: Council pattern for response comparison
"""

import asyncio
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class VotingStrategy(Enum):
    """Strategy for selecting the best plan from council responses"""

    MAJORITY_VOTE = "majority_vote"  # Vote on common elements
    CONFIDENCE_WEIGHTED = "confidence_weighted"  # Weight by model confidence
    LLM_JUDGE = "llm_judge"  # Use another LLM to judge
    SIMILARITY_CONSENSUS = "similarity_consensus"  # Find most central response
    STEP_INTERSECTION = "step_intersection"  # Combine common steps
    RANKED_CHOICE = "ranked_choice"  # Rank and aggregate preferences


class PlanQuality(Enum):
    """Quality assessment levels for plans"""

    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    REJECTED = "rejected"


@dataclass
class CouncilMember:
    """Represents an LLM in the council"""

    provider: str
    model: str
    weight: float = 1.0  # Weight in voting (can be adjusted based on performance)
    specialty: str | None = None  # e.g., "code", "reasoning", "creativity"
    reliability_score: float = 1.0  # Track historical performance

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "weight": self.weight,
            "specialty": self.specialty,
            "reliability_score": self.reliability_score,
        }


@dataclass
class PlanProposal:
    """A plan proposal from a council member"""

    member: CouncilMember
    raw_response: str
    steps: list[str] = field(default_factory=list)
    structured_plan: dict[str, Any] | None = None
    confidence_score: float = 0.0
    reasoning: str | None = None
    estimated_complexity: str | None = None
    risks: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "member": self.member.to_dict(),
            "raw_response": self.raw_response[:500] + "..."
            if len(self.raw_response) > 500
            else self.raw_response,
            "steps": self.steps,
            "structured_plan": self.structured_plan,
            "confidence_score": self.confidence_score,
            "reasoning": self.reasoning,
            "estimated_complexity": self.estimated_complexity,
            "risks": self.risks,
            "dependencies": self.dependencies,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CouncilDecision:
    """Final decision from the council deliberation"""

    winning_proposal: PlanProposal
    all_proposals: list[PlanProposal]
    voting_strategy: VotingStrategy
    agreement_score: float  # 0-1 how much the council agreed
    consensus_steps: list[str]  # Steps that multiple members agreed on
    dissenting_views: list[str]  # Notable disagreements
    quality_assessment: PlanQuality
    confidence_score: float
    deliberation_summary: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "winning_proposal": self.winning_proposal.to_dict(),
            "voting_strategy": self.voting_strategy.value,
            "agreement_score": self.agreement_score,
            "consensus_steps": self.consensus_steps,
            "dissenting_views": self.dissenting_views,
            "quality_assessment": self.quality_assessment.value,
            "confidence_score": self.confidence_score,
            "deliberation_summary": self.deliberation_summary,
            "num_proposals": len(self.all_proposals),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class JudgeVerdict:
    """Verdict from the LLM judge"""

    winner_index: int
    winner_provider: str
    reasoning: str
    scores: dict[str, float]  # provider -> score
    improvement_suggestions: list[str]
    quality_assessment: PlanQuality
    confidence: float


class PlanParser:
    """Parses plan responses from LLMs into structured format"""

    # Patterns for detecting plan steps
    STEP_PATTERNS = [
        r"^\s*\d+[\.\)]\s*(.+)$",  # "1. Step" or "1) Step"
        r"^\s*[-*•]\s*(.+)$",  # "- Step" or "* Step"
        r"^\s*Step\s+\d+[:\s]+(.+)$",  # "Step 1: description"
        r"^##\s+(.+)$",  # "## Step"
    ]

    @classmethod
    def extract_steps(cls, text: str) -> list[str]:
        """Extract plan steps from response text"""
        steps = []
        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            for pattern in cls.STEP_PATTERNS:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    step = match.group(1).strip()
                    if step and len(step) > 3:  # Avoid very short matches
                        steps.append(step)
                    break

        return steps

    @classmethod
    def extract_json_plan(cls, text: str) -> dict[str, Any] | None:
        """Try to extract JSON-formatted plan from response"""
        # Look for JSON code blocks
        json_patterns = [
            r"```json\s*\n([\s\S]*?)\n```",
            r"```\s*\n([\s\S]*?)\n```",
            r'\{[\s\S]*"steps"[\s\S]*\}',
        ]

        for pattern in json_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    json_str = match.group(1) if "```" in pattern else match.group(0)
                    return json.loads(json_str)
                except (json.JSONDecodeError, IndexError):
                    continue

        return None

    @classmethod
    def estimate_confidence(cls, text: str, steps: list[str]) -> float:
        """Estimate confidence score based on response quality"""
        score = 0.5  # Base score

        # Boost for structured steps
        if len(steps) >= 3:
            score += 0.15
        elif len(steps) >= 1:
            score += 0.05

        # Check for confidence indicators
        confidence_positive = ["definitely", "clearly", "recommend", "best approach", "optimal"]
        confidence_negative = ["maybe", "perhaps", "not sure", "uncertain", "alternatively"]

        text_lower = text.lower()
        for indicator in confidence_positive:
            if indicator in text_lower:
                score += 0.05
        for indicator in confidence_negative:
            if indicator in text_lower:
                score -= 0.05

        # Check for reasoning/justification
        reasoning_indicators = ["because", "since", "therefore", "this ensures", "the reason"]
        for indicator in reasoning_indicators:
            if indicator in text_lower:
                score += 0.03

        # Check for risk awareness
        risk_indicators = ["risk", "caveat", "consideration", "be aware", "potential issue"]
        for indicator in risk_indicators:
            if indicator in text_lower:
                score += 0.02

        return max(0.0, min(1.0, score))

    @classmethod
    def extract_risks(cls, text: str) -> list[str]:
        """Extract mentioned risks from the plan"""
        risks = []
        risk_patterns = [
            r"(?:risk|caveat|warning|consideration|potential issue)[:\s]+([^.]+\.)",
            r"(?:be aware|note that|keep in mind)[:\s]+([^.]+\.)",
        ]

        for pattern in risk_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            risks.extend(matches)

        return risks[:5]  # Limit to top 5 risks


class LLMCouncil:
    """
    Council of LLM Experts for collaborative plan evaluation.

    This class manages multiple LLM "council members" that can be queried
    in parallel to generate and evaluate plans. It implements various
    voting strategies to select the best plan.

    Features:
    - Query multiple LLMs in parallel
    - Multiple voting/consensus strategies
    - LLM-as-judge pattern for plan evaluation
    - Plan parsing and structuring
    - Confidence scoring
    - Deliberation logging

    Usage:
        council = LLMCouncil(llm_client=multi_llm_client)

        # Add council members
        council.add_member("anthropic", "claude-3-5-sonnet-20241022", weight=1.2)
        council.add_member("openai", "gpt-4-turbo-preview", weight=1.0)

        # Query all members
        proposals = await council.query_all(planning_prompt)

        # Evaluate and select best plan
        decision = await council.evaluate_responses(
            proposals,
            strategy=VotingStrategy.LLM_JUDGE
        )
    """

    def __init__(
        self,
        llm_client=None,
        default_strategy: VotingStrategy = VotingStrategy.SIMILARITY_CONSENSUS,
        min_agreement_threshold: float = 0.6,
        enable_judge: bool = True,
    ):
        """
        Initialize the LLM Council.

        Args:
            llm_client: MultiLLMClient instance for API calls
            default_strategy: Default voting strategy
            min_agreement_threshold: Minimum agreement for consensus
            enable_judge: Whether to use LLM judge for tie-breaking
        """
        self.llm_client = llm_client
        self.default_strategy = default_strategy
        self.min_agreement_threshold = min_agreement_threshold
        self.enable_judge = enable_judge

        # Council members
        self.members: list[CouncilMember] = []

        # Performance tracking per member
        self.member_stats: dict[str, dict[str, Any]] = {}

        # Parser for plan extraction
        self.parser = PlanParser()

        # Deliberation history
        self.deliberation_history: list[CouncilDecision] = []

        logger.info("LLMCouncil initialized")

    def add_member(
        self,
        provider: str,
        model: str,
        weight: float = 1.0,
        specialty: str | None = None,
    ) -> CouncilMember:
        """
        Add a member to the council.

        Args:
            provider: LLM provider name
            model: Model identifier
            weight: Voting weight (higher = more influence)
            specialty: Area of expertise

        Returns:
            The created CouncilMember
        """
        member = CouncilMember(
            provider=provider,
            model=model,
            weight=weight,
            specialty=specialty,
        )
        self.members.append(member)

        # Initialize stats
        member_key = f"{provider}:{model}"
        self.member_stats[member_key] = {
            "queries": 0,
            "wins": 0,
            "total_confidence": 0.0,
            "avg_latency_ms": 0.0,
        }

        logger.info(f"Added council member: {provider}/{model} (weight={weight})")
        return member

    def remove_member(self, provider: str, model: str) -> bool:
        """Remove a member from the council"""
        for i, member in enumerate(self.members):
            if member.provider == provider and member.model == model:
                self.members.pop(i)
                logger.info(f"Removed council member: {provider}/{model}")
                return True
        return False

    def set_member_weight(self, provider: str, model: str, weight: float):
        """Update a member's voting weight"""
        for member in self.members:
            if member.provider == provider and member.model == model:
                member.weight = weight
                logger.info(f"Updated weight for {provider}/{model}: {weight}")
                return

    async def query_all(
        self,
        prompt: str,
        system_prompt: str | None = None,
        members: list[CouncilMember] | None = None,
        timeout_seconds: float = 60.0,
        **kwargs,
    ) -> list[PlanProposal]:
        """
        Query all council members in parallel.

        Args:
            prompt: The planning prompt to send to all members
            system_prompt: Optional system prompt for context
            members: Specific members to query (defaults to all)
            timeout_seconds: Timeout for each query
            **kwargs: Additional parameters for LLM calls

        Returns:
            List of PlanProposals from responding members
        """
        if self.llm_client is None:
            raise RuntimeError("No LLM client configured")

        target_members = members or self.members
        if not target_members:
            raise ValueError("No council members configured")

        logger.info(f"Querying {len(target_members)} council members")

        # Create tasks for parallel execution
        async def query_member(member: CouncilMember) -> PlanProposal | None:
            import time

            start_time = time.time()

            try:
                from framework.llms.multi_llm_client import LLMProvider

                # Map provider string to enum
                provider_map = {
                    "anthropic": LLMProvider.ANTHROPIC,
                    "openai": LLMProvider.OPENAI,
                    "google": LLMProvider.GOOGLE,
                    "local": LLMProvider.LOCAL,
                }

                provider_enum = provider_map.get(member.provider.lower())
                if not provider_enum:
                    logger.warning(f"Unknown provider: {member.provider}")
                    return None

                # Make the API call
                response = await asyncio.wait_for(
                    self.llm_client.generate(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        provider=provider_enum,
                        model=member.model,
                        enable_fallback=False,  # Don't fallback within council query
                        **kwargs,
                    ),
                    timeout=timeout_seconds,
                )

                latency_ms = (time.time() - start_time) * 1000

                if not response.get("text"):
                    logger.warning(f"Empty response from {member.provider}/{member.model}")
                    return None

                raw_text = response["text"]

                # Parse the response
                steps = self.parser.extract_steps(raw_text)
                structured_plan = self.parser.extract_json_plan(raw_text)
                confidence = self.parser.estimate_confidence(raw_text, steps)
                risks = self.parser.extract_risks(raw_text)

                # Update member stats
                member_key = f"{member.provider}:{member.model}"
                if member_key in self.member_stats:
                    stats = self.member_stats[member_key]
                    stats["queries"] += 1
                    stats["total_confidence"] += confidence
                    n = stats["queries"]
                    stats["avg_latency_ms"] = (stats["avg_latency_ms"] * (n - 1) + latency_ms) / n

                return PlanProposal(
                    member=member,
                    raw_response=raw_text,
                    steps=steps,
                    structured_plan=structured_plan,
                    confidence_score=confidence,
                    risks=risks,
                    latency_ms=latency_ms,
                )

            except asyncio.TimeoutError:
                logger.warning(f"Timeout querying {member.provider}/{member.model}")
                return None
            except Exception as e:
                logger.error(f"Error querying {member.provider}/{member.model}: {e}")
                return None

        # Execute all queries in parallel
        tasks = [query_member(member) for member in target_members]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful responses
        proposals = []
        for result in results:
            if isinstance(result, PlanProposal):
                proposals.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Query exception: {result}")

        logger.info(f"Received {len(proposals)} proposals from council")
        return proposals

    async def evaluate_responses(
        self,
        proposals: list[PlanProposal],
        strategy: VotingStrategy | None = None,
        task_context: str | None = None,
    ) -> CouncilDecision:
        """
        Evaluate multiple proposals and select the best plan.

        Args:
            proposals: List of plan proposals to evaluate
            strategy: Voting strategy to use (defaults to default_strategy)
            task_context: Optional context about the task for better evaluation

        Returns:
            CouncilDecision with the winning proposal and analysis
        """
        if not proposals:
            raise ValueError("No proposals to evaluate")

        strategy = strategy or self.default_strategy

        logger.info(f"Evaluating {len(proposals)} proposals with strategy: {strategy.value}")

        # If only one proposal, it wins by default
        if len(proposals) == 1:
            return CouncilDecision(
                winning_proposal=proposals[0],
                all_proposals=proposals,
                voting_strategy=strategy,
                agreement_score=1.0,
                consensus_steps=proposals[0].steps,
                dissenting_views=[],
                quality_assessment=self._assess_quality(proposals[0].confidence_score),
                confidence_score=proposals[0].confidence_score,
                deliberation_summary="Single proposal - no deliberation needed",
            )

        # Apply the selected voting strategy
        if strategy == VotingStrategy.MAJORITY_VOTE:
            decision = self._majority_vote(proposals)
        elif strategy == VotingStrategy.CONFIDENCE_WEIGHTED:
            decision = self._confidence_weighted_vote(proposals)
        elif strategy == VotingStrategy.SIMILARITY_CONSENSUS:
            decision = self._similarity_consensus(proposals)
        elif strategy == VotingStrategy.STEP_INTERSECTION:
            decision = self._step_intersection(proposals)
        elif strategy == VotingStrategy.LLM_JUDGE:
            decision = await self._llm_judge_vote(proposals, task_context)
        elif strategy == VotingStrategy.RANKED_CHOICE:
            decision = self._ranked_choice_vote(proposals)
        else:
            # Default to similarity consensus
            decision = self._similarity_consensus(proposals)

        # Update winner's stats
        winner_key = (
            f"{decision.winning_proposal.member.provider}:{decision.winning_proposal.member.model}"
        )
        if winner_key in self.member_stats:
            self.member_stats[winner_key]["wins"] += 1

        # Store in history
        self.deliberation_history.append(decision)

        logger.info(
            f"Council decision: {decision.winning_proposal.member.provider} wins "
            f"(agreement: {decision.agreement_score:.2f})"
        )

        return decision

    def _majority_vote(self, proposals: list[PlanProposal]) -> CouncilDecision:
        """Vote on common steps across proposals"""
        # Count step occurrences (normalized)
        all_steps = []
        for proposal in proposals:
            for step in proposal.steps:
                # Normalize step for comparison
                normalized = step.lower().strip()
                all_steps.append((normalized, step, proposal))

        # Find steps that appear in multiple proposals
        step_counter = Counter(s[0] for s in all_steps)
        consensus_steps = [s for s, count in step_counter.items() if count > 1]

        # Weight proposals by how many consensus steps they include
        scores = {}
        for proposal in proposals:
            normalized_steps = [s.lower().strip() for s in proposal.steps]
            matching = sum(1 for cs in consensus_steps if cs in normalized_steps)
            scores[id(proposal)] = matching * proposal.member.weight

        # Select winner
        winner = max(proposals, key=lambda p: scores.get(id(p), 0))

        # Calculate agreement
        if consensus_steps:
            max_possible = len(proposals)
            avg_agreement = sum(
                step_counter.get(cs, 0) / max_possible for cs in consensus_steps
            ) / len(consensus_steps)
        else:
            avg_agreement = 0.0

        return CouncilDecision(
            winning_proposal=winner,
            all_proposals=proposals,
            voting_strategy=VotingStrategy.MAJORITY_VOTE,
            agreement_score=avg_agreement,
            consensus_steps=[s[1] for s in all_steps if s[0] in consensus_steps][:10],
            dissenting_views=self._find_dissenting_views(proposals, winner),
            quality_assessment=self._assess_quality(winner.confidence_score),
            confidence_score=winner.confidence_score,
            deliberation_summary=f"Majority vote selected {winner.member.provider} with {len(consensus_steps)} consensus steps",
        )

    def _confidence_weighted_vote(self, proposals: list[PlanProposal]) -> CouncilDecision:
        """Weight votes by confidence scores and member weights"""
        # Calculate weighted scores
        scores = {}
        for proposal in proposals:
            weighted_score = (
                proposal.confidence_score
                * proposal.member.weight
                * proposal.member.reliability_score
            )
            scores[id(proposal)] = weighted_score

        # Select winner
        winner = max(proposals, key=lambda p: scores.get(id(p), 0))

        # Calculate agreement based on score distribution
        total_score = sum(scores.values())
        if total_score > 0:
            winner_share = scores[id(winner)] / total_score
            # If winner has >70% of weighted score, high agreement
            agreement = min(1.0, winner_share * 1.4)
        else:
            agreement = 0.5

        return CouncilDecision(
            winning_proposal=winner,
            all_proposals=proposals,
            voting_strategy=VotingStrategy.CONFIDENCE_WEIGHTED,
            agreement_score=agreement,
            consensus_steps=winner.steps,
            dissenting_views=self._find_dissenting_views(proposals, winner),
            quality_assessment=self._assess_quality(winner.confidence_score),
            confidence_score=scores[id(winner)],
            deliberation_summary=(
                f"Confidence-weighted vote: {winner.member.provider} "
                f"(score: {scores[id(winner)]:.2f})"
            ),
        )

    def _similarity_consensus(self, proposals: list[PlanProposal]) -> CouncilDecision:
        """Find the most central/representative proposal"""

        # Calculate pairwise similarity between proposals
        def text_similarity(text1: str, text2: str) -> float:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            return intersection / union if union > 0 else 0.0

        # Find proposal with highest average similarity to others
        best_proposal = proposals[0]
        best_avg_sim = 0.0
        all_similarities = []

        for p1 in proposals:
            sims = []
            for p2 in proposals:
                if p1 is not p2:
                    sim = text_similarity(p1.raw_response, p2.raw_response)
                    sims.append(sim)
                    all_similarities.append(sim)

            if sims:
                avg_sim = sum(sims) / len(sims)
                if avg_sim > best_avg_sim:
                    best_avg_sim = avg_sim
                    best_proposal = p1

        # Overall agreement is average pairwise similarity
        overall_agreement = (
            sum(all_similarities) / len(all_similarities) if all_similarities else 0.5
        )

        return CouncilDecision(
            winning_proposal=best_proposal,
            all_proposals=proposals,
            voting_strategy=VotingStrategy.SIMILARITY_CONSENSUS,
            agreement_score=overall_agreement,
            consensus_steps=best_proposal.steps,
            dissenting_views=self._find_dissenting_views(proposals, best_proposal),
            quality_assessment=self._assess_quality(best_proposal.confidence_score),
            confidence_score=best_proposal.confidence_score,
            deliberation_summary=(
                f"Similarity consensus: {best_proposal.member.provider} "
                f"(avg similarity: {best_avg_sim:.2f})"
            ),
        )

    def _step_intersection(self, proposals: list[PlanProposal]) -> CouncilDecision:
        """Combine common steps from all proposals"""
        # Collect and normalize all steps
        step_sources: dict[str, list[tuple[str, PlanProposal]]] = {}

        for proposal in proposals:
            for step in proposal.steps:
                # Create a normalized key for matching similar steps
                normalized = " ".join(sorted(step.lower().split()[:5]))
                if normalized not in step_sources:
                    step_sources[normalized] = []
                step_sources[normalized].append((step, proposal))

        # Find steps that appear in multiple proposals
        common_steps = []
        for normalized, sources in step_sources.items():
            if len(sources) >= 2:
                # Use the longest version of the step
                best_step = max(sources, key=lambda x: len(x[0]))[0]
                common_steps.append((best_step, len(sources)))

        # Sort by frequency
        common_steps.sort(key=lambda x: x[1], reverse=True)
        consensus_steps = [s[0] for s in common_steps[:10]]

        # Select winner as proposal with most consensus steps
        winner = max(
            proposals,
            key=lambda p: sum(
                1
                for s in p.steps
                if any(cs.lower() in s.lower() or s.lower() in cs.lower() for cs, _ in common_steps)
            ),
        )

        # Agreement based on step overlap
        if common_steps:
            agreement = len(common_steps) / max(len(p.steps) for p in proposals if p.steps)
        else:
            agreement = 0.0

        return CouncilDecision(
            winning_proposal=winner,
            all_proposals=proposals,
            voting_strategy=VotingStrategy.STEP_INTERSECTION,
            agreement_score=min(1.0, agreement),
            consensus_steps=consensus_steps,
            dissenting_views=self._find_dissenting_views(proposals, winner),
            quality_assessment=self._assess_quality(winner.confidence_score),
            confidence_score=winner.confidence_score,
            deliberation_summary=(
                f"Step intersection found {len(consensus_steps)} common steps. "
                f"Winner: {winner.member.provider}"
            ),
        )

    async def _llm_judge_vote(
        self,
        proposals: list[PlanProposal],
        task_context: str | None = None,
    ) -> CouncilDecision:
        """Use an LLM as a judge to evaluate proposals"""
        if not self.llm_client or not self.enable_judge:
            # Fallback to similarity consensus
            return self._similarity_consensus(proposals)

        # Build the judge prompt
        judge_prompt = self._build_judge_prompt(proposals, task_context)

        try:
            # Use the primary model as judge
            response = await self.llm_client.generate(
                prompt=judge_prompt,
                system_prompt=self._get_judge_system_prompt(),
                temperature=0.3,  # Lower temperature for more consistent judging
                max_tokens=2000,
            )

            if response.get("text"):
                verdict = self._parse_judge_verdict(response["text"], proposals)

                if verdict and 0 <= verdict.winner_index < len(proposals):
                    winner = proposals[verdict.winner_index]

                    return CouncilDecision(
                        winning_proposal=winner,
                        all_proposals=proposals,
                        voting_strategy=VotingStrategy.LLM_JUDGE,
                        agreement_score=verdict.confidence,
                        consensus_steps=winner.steps,
                        dissenting_views=verdict.improvement_suggestions,
                        quality_assessment=verdict.quality_assessment,
                        confidence_score=verdict.confidence,
                        deliberation_summary=(
                            f"LLM Judge selected {winner.member.provider}: {verdict.reasoning[:200]}"
                        ),
                    )
        except Exception as e:
            logger.error(f"LLM judge evaluation failed: {e}")

        # Fallback to similarity consensus
        logger.warning("Falling back to similarity consensus")
        return self._similarity_consensus(proposals)

    def _ranked_choice_vote(self, proposals: list[PlanProposal]) -> CouncilDecision:
        """Ranked choice voting based on multiple criteria"""
        # Score each proposal on multiple dimensions
        scores: dict[int, dict[str, float]] = {}

        for i, proposal in enumerate(proposals):
            scores[i] = {
                "confidence": proposal.confidence_score,
                "step_count": min(1.0, len(proposal.steps) / 10),  # Normalize to 0-1
                "structure": 1.0 if proposal.structured_plan else 0.5,
                "weight": proposal.member.weight / max(m.weight for m in self.members),
                "risk_awareness": min(1.0, len(proposal.risks) / 3),
            }

        # Calculate total ranked score
        total_scores = {}
        for i, dim_scores in scores.items():
            total_scores[i] = sum(dim_scores.values()) / len(dim_scores)

        # Select winner
        winner_idx = max(total_scores, key=total_scores.get)
        winner = proposals[winner_idx]

        # Agreement based on score spread
        if len(total_scores) > 1:
            max_score = max(total_scores.values())
            second_score = sorted(total_scores.values(), reverse=True)[1]
            agreement = (max_score - second_score) / max_score if max_score > 0 else 0.5
        else:
            agreement = 1.0

        return CouncilDecision(
            winning_proposal=winner,
            all_proposals=proposals,
            voting_strategy=VotingStrategy.RANKED_CHOICE,
            agreement_score=agreement,
            consensus_steps=winner.steps,
            dissenting_views=self._find_dissenting_views(proposals, winner),
            quality_assessment=self._assess_quality(total_scores[winner_idx]),
            confidence_score=total_scores[winner_idx],
            deliberation_summary=(
                f"Ranked choice: {winner.member.provider} (score: {total_scores[winner_idx]:.2f})"
            ),
        )

    def _build_judge_prompt(
        self,
        proposals: list[PlanProposal],
        task_context: str | None = None,
    ) -> str:
        """Build the prompt for the LLM judge"""
        prompt_parts = [
            "You are an expert judge evaluating multiple proposed plans.",
            "Your task is to select the BEST plan and explain why.",
            "",
        ]

        if task_context:
            prompt_parts.extend(
                [
                    "TASK CONTEXT:",
                    task_context,
                    "",
                ]
            )

        prompt_parts.append("PROPOSALS TO EVALUATE:")
        prompt_parts.append("")

        for i, proposal in enumerate(proposals):
            prompt_parts.extend(
                [
                    f"--- PROPOSAL {i + 1} (from {proposal.member.provider}/{proposal.member.model}) ---",
                    proposal.raw_response[:2000],
                    "",
                ]
            )

        prompt_parts.extend(
            [
                "",
                "EVALUATION CRITERIA:",
                "1. Completeness - Does the plan address all aspects of the task?",
                "2. Clarity - Are the steps clear and actionable?",
                "3. Feasibility - Is the plan realistic and implementable?",
                "4. Risk Awareness - Does the plan consider potential issues?",
                "5. Efficiency - Is the approach appropriately streamlined?",
                "",
                "RESPONSE FORMAT:",
                "Please respond with:",
                "WINNER: [proposal number 1-N]",
                "QUALITY: [excellent/good/acceptable/poor]",
                "CONFIDENCE: [0.0-1.0]",
                "REASONING: [your explanation]",
                "IMPROVEMENTS: [suggestions for the winning plan]",
            ]
        )

        return "\n".join(prompt_parts)

    def _get_judge_system_prompt(self) -> str:
        """Get system prompt for the LLM judge"""
        return """You are an impartial judge evaluating proposed plans.
Your role is to:
1. Carefully analyze each proposal
2. Compare them objectively based on the criteria provided
3. Select the best proposal and explain your reasoning
4. Suggest improvements if any

Be fair, thorough, and provide clear justification for your decision.
Focus on practical value and implementation feasibility."""

    def _parse_judge_verdict(
        self,
        response: str,
        proposals: list[PlanProposal],
    ) -> JudgeVerdict | None:
        """Parse the judge's response into a verdict"""
        try:
            # Extract winner
            winner_match = re.search(r"WINNER:\s*(\d+)", response, re.IGNORECASE)
            winner_idx = int(winner_match.group(1)) - 1 if winner_match else 0

            # Extract quality
            quality_match = re.search(r"QUALITY:\s*(\w+)", response, re.IGNORECASE)
            quality_str = quality_match.group(1).lower() if quality_match else "acceptable"
            quality_map = {
                "excellent": PlanQuality.EXCELLENT,
                "good": PlanQuality.GOOD,
                "acceptable": PlanQuality.ACCEPTABLE,
                "poor": PlanQuality.POOR,
            }
            quality = quality_map.get(quality_str, PlanQuality.ACCEPTABLE)

            # Extract confidence
            conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", response, re.IGNORECASE)
            confidence = float(conf_match.group(1)) if conf_match else 0.7

            # Extract reasoning
            reason_match = re.search(
                r"REASONING:\s*(.+?)(?=IMPROVEMENTS:|$)", response, re.IGNORECASE | re.DOTALL
            )
            reasoning = reason_match.group(1).strip() if reason_match else "No reasoning provided"

            # Extract improvements
            improve_match = re.search(
                r"IMPROVEMENTS:\s*(.+?)$", response, re.IGNORECASE | re.DOTALL
            )
            improvements = []
            if improve_match:
                improvements = [s.strip() for s in improve_match.group(1).split("\n") if s.strip()]

            return JudgeVerdict(
                winner_index=winner_idx,
                winner_provider=proposals[winner_idx].member.provider
                if 0 <= winner_idx < len(proposals)
                else "",
                reasoning=reasoning,
                scores={},  # Could extract individual scores if needed
                improvement_suggestions=improvements[:5],
                quality_assessment=quality,
                confidence=min(1.0, max(0.0, confidence)),
            )

        except Exception as e:
            logger.error(f"Failed to parse judge verdict: {e}")
            return None

    def _find_dissenting_views(
        self,
        proposals: list[PlanProposal],
        winner: PlanProposal,
    ) -> list[str]:
        """Find notable differences from non-winning proposals"""
        dissenting = []

        winner_steps_lower = [s.lower() for s in winner.steps]

        for proposal in proposals:
            if proposal is winner:
                continue

            # Find steps in this proposal not in winner
            for step in proposal.steps:
                step_lower = step.lower()
                if not any(step_lower in ws or ws in step_lower for ws in winner_steps_lower):
                    dissenting.append(f"[{proposal.member.provider}] {step[:100]}")

        return dissenting[:5]  # Limit to top 5

    def _assess_quality(self, score: float) -> PlanQuality:
        """Assess plan quality based on score"""
        if score >= 0.85:
            return PlanQuality.EXCELLENT
        elif score >= 0.70:
            return PlanQuality.GOOD
        elif score >= 0.50:
            return PlanQuality.ACCEPTABLE
        elif score >= 0.30:
            return PlanQuality.POOR
        else:
            return PlanQuality.REJECTED

    def get_member_stats(self) -> dict[str, dict[str, Any]]:
        """Get performance statistics for all members"""
        stats = {}
        for key, data in self.member_stats.items():
            stats[key] = {
                **data,
                "win_rate": data["wins"] / max(1, data["queries"]),
                "avg_confidence": data["total_confidence"] / max(1, data["queries"]),
            }
        return stats

    def update_member_reliability(self, provider: str, model: str, success: bool):
        """Update a member's reliability score based on execution outcome"""
        for member in self.members:
            if member.provider == provider and member.model == model:
                # Exponential moving average
                alpha = 0.1
                outcome = 1.0 if success else 0.0
                member.reliability_score = (1 - alpha) * member.reliability_score + alpha * outcome
                logger.debug(
                    f"Updated reliability for {provider}/{model}: {member.reliability_score:.2f}"
                )
                return


# Convenience function for quick council creation
def create_council(
    llm_client,
    providers: list[str] | None = None,
    strategy: VotingStrategy = VotingStrategy.SIMILARITY_CONSENSUS,
) -> LLMCouncil:
    """
    Create a pre-configured LLM Council.

    Args:
        llm_client: MultiLLMClient instance
        providers: List of providers to include (defaults to all available)
        strategy: Default voting strategy

    Returns:
        Configured LLMCouncil instance
    """
    council = LLMCouncil(llm_client=llm_client, default_strategy=strategy)

    # Default members if not specified
    default_members = [
        ("anthropic", "claude-3-5-sonnet-20241022", 1.2),
        ("openai", "gpt-4-turbo-preview", 1.0),
        ("google", "gemini-pro", 0.9),
    ]

    available = llm_client.get_available_providers() if llm_client else []

    for provider, model, weight in default_members:
        if providers is None or provider in providers:
            if provider in available:
                council.add_member(provider, model, weight)

    return council
