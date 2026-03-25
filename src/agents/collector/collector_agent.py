"""
Collector Agent.

Planner가 생성한 ResearchPlan을 기반으로 Tool을 호출해 데이터를 수집한다.

실행 흐름:
  1. ResearchPlan.source_priority 순서로 ToolRouter가 Tool을 결정
  2. 각 (source_type, query) 조합으로 Tool 호출
  3. DocumentNormalizer로 raw 결과를 SourceDocument로 변환
  4. DuplicateFilter로 URL/제목 기준 중복 제거
  5. max_sources 도달 시 조기 종료

설계 원칙 (AI_CONTEXT.md):
  - 외부 데이터 접근은 Tool abstraction을 통해 수행
  - side-effect 없음 (Tool 호출 결과만 수집)
  - 수집 실패 시 partial 결과로 계속 진행
"""
import logging
from typing import Any

from src.agents.collector.document_normalizer import DocumentNormalizer
from src.agents.collector.duplicate_filter import DuplicateFilter
from src.agents.collector.tool_router import ToolRouter
from src.application.dto.agent_io import CollectorOutput, PlannerOutput
from src.domain.models.source_document import SourceDocument
from src.domain.ports.tool_port import ToolPort

logger = logging.getLogger(__name__)


class CollectorAgent:
    def __init__(self, tools: list[ToolPort]) -> None:
        self.tools = tools
        self.router = ToolRouter(tools=tools)
        self.normalizer = DocumentNormalizer()
        self.dedup = DuplicateFilter()

    async def run(self, planner_output: PlannerOutput, max_sources: int) -> CollectorOutput:
        plan = planner_output.plan

        source_priority = plan.source_priority or ["papers", "tech_blogs", "news"]
        queries = plan.queries or [plan.question]

        logger.info(
            "collector_start source_priority=%s query_count=%s max_sources=%s",
            source_priority,
            len(queries),
            max_sources,
        )

        # source_priority 순서로 (source_type, tool) 쌍 확보
        tool_pairs = self.router.resolve_ordered(source_priority)

        raw_docs: list[SourceDocument] = []
        for source_type, tool in tool_pairs:
            for query in queries:
                if len(raw_docs) >= max_sources * 2:
                    # dedup 이전 버퍼가 한계에 도달하면 조기 종료
                    break
                collected = await self._call_tool(
                    tool=tool,
                    query=query,
                    limit=max_sources,
                    source_type=source_type,
                )
                raw_docs.extend(collected)

        # 중복 제거 후 max_sources 적용
        unique_docs = self.dedup.filter(raw_docs)
        final_docs = unique_docs[:max_sources]

        logger.info(
            "collector_done raw=%s unique=%s final=%s",
            len(raw_docs),
            len(unique_docs),
            len(final_docs),
        )
        return CollectorOutput(documents=final_docs)

    async def _call_tool(
        self,
        tool: ToolPort,
        query: str,
        limit: int,
        source_type: str,
    ) -> list[SourceDocument]:
        try:
            result = await tool.run({"query": query, "limit": limit})
            return self._extract_docs(
                result=result,
                source_type=source_type,
                query=query,
            )
        except Exception as exc:  # noqa: BLE001
            # 개별 Tool 실패는 경고만 남기고 수집을 계속 진행한다.
            logger.warning(
                "collector_tool_failed tool=%s query=%s error=%s",
                tool.name,
                query,
                exc,
            )
            return []

    def _extract_docs(
        self,
        result: dict[str, Any],
        source_type: str,
        query: str,
    ) -> list[SourceDocument]:
        docs: list[SourceDocument] = []
        for raw in result.get("items", []):
            doc = self.normalizer.normalize(
                raw=raw,
                source_type=source_type,
                query=query,
            )
            if doc:
                docs.append(doc)
        return docs
