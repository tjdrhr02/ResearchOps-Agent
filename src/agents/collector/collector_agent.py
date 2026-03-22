"""
Collector Agent.

Planner 결과로 생성된 queries를 Tool을 통해 실행하고
수집된 raw 데이터를 SourceDocument로 정규화한다.
"""
import logging
from typing import Any

from src.application.dto.agent_io import CollectorOutput, PlannerOutput
from src.domain.models.source_document import SourceDocument
from src.domain.ports.tool_port import ToolPort

logger = logging.getLogger(__name__)


class CollectorAgent:
    def __init__(self, tools: list[ToolPort]) -> None:
        self.tools = tools
        self._search_tools = [t for t in tools if t.name.startswith("search_")]
        self._fetch_tool = next((t for t in tools if t.name == "fetch_article_content"), None)

    async def run(self, planner_output: PlannerOutput, max_sources: int) -> CollectorOutput:
        docs: list[SourceDocument] = []
        plan = planner_output.plan

        logger.info(
            "collector_start query_count=%s max_sources=%s",
            len(plan.queries),
            max_sources,
        )

        for query in plan.queries:
            for tool in self._search_tools:
                if len(docs) >= max_sources:
                    break
                result = await tool.run({"query": query, "limit": max_sources})
                for raw in result.get("items", []):
                    doc = self._normalize(raw)
                    if doc:
                        docs.append(doc)
                    if len(docs) >= max_sources:
                        break

        logger.info("collector_done total_docs=%s", len(docs))
        return CollectorOutput(documents=docs)

    def _normalize(self, raw: Any) -> SourceDocument | None:
        try:
            if isinstance(raw, dict):
                return SourceDocument(**raw)
            if isinstance(raw, SourceDocument):
                return raw
        except Exception as exc:  # noqa: BLE001
            logger.warning("collector_normalize_failed raw=%s error=%s", raw, exc)
        return None
