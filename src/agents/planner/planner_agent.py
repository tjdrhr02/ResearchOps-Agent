import logging
from typing import Any, Protocol

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

from src.agents.prompts.templates import PLANNER_PROMPT
from src.application.dto.agent_io import PlannerOutput
from src.domain.models.research_plan import ResearchPlan

logger = logging.getLogger(__name__)


class PlannerRunnable(Protocol):
    async def ainvoke(self, input_data: Any) -> Any:
        raise NotImplementedError


class PlannerPlanSchema(BaseModel):
    research_type: str
    queries: list[str] = Field(default_factory=list)
    focus_topics: list[str] = Field(default_factory=list)
    source_priority: list[str] = Field(default_factory=list)


class PlannerAgent:
    def __init__(self, llm: PlannerRunnable | None = None) -> None:
        self.llm = llm
        self.output_parser = PydanticOutputParser(pydantic_object=PlannerPlanSchema)
        self.prompt_template = PromptTemplate(
            template=PLANNER_PROMPT,
            input_variables=["user_query"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()},
        )

    async def run(self, user_query: str) -> PlannerOutput:
        if not self.llm:
            return self._fallback_plan(user_query=user_query)

        prompt = self.prompt_template.format(user_query=user_query)
        try:
            raw = await self.llm.ainvoke(prompt)
            raw_text = raw.content if hasattr(raw, "content") else str(raw)
            parsed = self.output_parser.parse(raw_text)

            source_priority = parsed.source_priority or ["papers", "tech_blogs", "news"]
            queries = parsed.queries or [user_query, f"{user_query} latest trends"]
            focus_topics = parsed.focus_topics or [user_query]
            plan = ResearchPlan(
                question=user_query,
                objective=f"Create evidence-based brief for query: {user_query}",
                research_type=parsed.research_type,
                queries=queries,
                focus_topics=focus_topics,
                source_priority=source_priority,
                source_types=source_priority,
            )
            return PlannerOutput(plan=plan)
        except Exception as exc:  # noqa: BLE001
            logger.warning("planner_llm_failed fallback_used error=%s", exc)
            return self._fallback_plan(user_query=user_query)

    def _fallback_plan(self, user_query: str) -> PlannerOutput:
        queries = [
            user_query,
            f"{user_query} recent papers",
            f"{user_query} production case study",
        ]
        plan = ResearchPlan(
            question=user_query,
            objective=f"Analyze latest knowledge and practical trends for: {user_query}",
            research_type="trend_analysis",
            queries=queries,
            focus_topics=[user_query, "state of the art", "real-world adoption"],
            source_priority=["papers", "tech_blogs", "news"],
            source_types=["papers", "tech_blogs", "news"],
        )
        return PlannerOutput(plan=plan)
