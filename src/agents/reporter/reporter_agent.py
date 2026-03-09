from src.application.dto.agent_io import PlannerOutput, SynthesizerOutput
from src.domain.models.research_brief import ResearchBrief


class ReporterAgent:
    async def run(self, planner_output: PlannerOutput, synthesis: SynthesizerOutput) -> ResearchBrief:
        citations = [chunk.citation for chunk in synthesis.evidence if chunk.citation]
        return ResearchBrief(
            executive_summary=f"Research brief for: {planner_output.plan.question}",
            key_trends=synthesis.claims,
            evidence=[chunk.content for chunk in synthesis.evidence],
            source_comparison=synthesis.comparisons,
            open_questions=synthesis.open_questions,
            citations=citations,
        )
