from src.application.dto.agent_io import PlannerOutput
from src.domain.models.research_plan import ResearchPlan


class PlannerAgent:
    async def run(self, user_query: str) -> PlannerOutput:
        queries = [
            user_query,
            f"{user_query} recent papers",
            f"{user_query} production case study",
        ]
        plan = ResearchPlan(
            question=user_query,
            objective=f"Analyze latest knowledge and practical trends for: {user_query}",
            queries=queries,
            source_types=["papers", "tech_blogs", "news"],
        )
        return PlannerOutput(plan=plan)
