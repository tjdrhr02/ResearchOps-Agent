import pytest

from src.agents.planner.planner_agent import PlannerAgent


class FakePlannerMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakePlannerLLM:
    async def ainvoke(self, input_data):  # noqa: ANN001
        return FakePlannerMessage(
            content="""
{
  "research_type": "market_landscape",
  "queries": [
    "agentic ai workflow architecture",
    "langchain planner agent best practices"
  ],
  "focus_topics": [
    "workflow design",
    "evaluation metrics"
  ],
  "source_priority": [
    "papers",
    "tech_blogs",
    "news"
  ]
}
""".strip(),
        )


@pytest.mark.asyncio
async def test_planner_agent_with_llm_structured_output() -> None:
    agent = PlannerAgent(llm=FakePlannerLLM())
    output = await agent.run("How to design agentic research workflow?")
    assert output.plan.research_type == "market_landscape"
    assert len(output.plan.queries) == 2
    assert output.plan.source_priority[0] == "papers"


@pytest.mark.asyncio
async def test_planner_agent_fallback_without_llm() -> None:
    agent = PlannerAgent()
    output = await agent.run("RAG observability strategy")
    assert output.plan.research_type == "trend_analysis"
    assert len(output.plan.queries) >= 1
