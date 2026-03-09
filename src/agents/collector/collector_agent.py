from src.application.dto.agent_io import CollectorOutput, PlannerOutput
from src.domain.models.source_document import SourceDocument
from src.domain.ports.tool_port import ToolPort


class CollectorAgent:
    def __init__(self, tools: list[ToolPort]) -> None:
        self.tools = tools

    async def run(self, planner_output: PlannerOutput, max_sources: int) -> CollectorOutput:
        docs: list[SourceDocument] = []
        for query in planner_output.plan.queries:
            for tool in self.tools:
                if tool.name.startswith("search_"):
                    result = await tool.run({"query": query, "limit": max_sources})
                    for item in result.get("items", []):
                        docs.append(SourceDocument(**item))
                        if len(docs) >= max_sources:
                            return CollectorOutput(documents=docs)
        return CollectorOutput(documents=docs)
