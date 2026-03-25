"""
Tool Router.

ResearchPlan의 source_priority를 기반으로
적절한 Tool을 선택해 반환하는 라우터.

source_priority 값과 Tool name 매핑:
  papers      → search_papers
  tech_blogs  → search_tech_blogs
  news        → search_news
"""
import logging
from dataclasses import dataclass, field

from src.domain.ports.tool_port import ToolPort

logger = logging.getLogger(__name__)

_PRIORITY_TO_TOOL: dict[str, str] = {
    "papers": "search_papers",
    "tech_blogs": "search_tech_blogs",
    "news": "search_news",
}


@dataclass
class ToolRouter:
    tools: list[ToolPort]
    _tool_map: dict[str, ToolPort] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._tool_map = {tool.name: tool for tool in self.tools}
        logger.info("tool_router_init available_tools=%s", list(self._tool_map.keys()))

    def resolve(self, source_type: str) -> ToolPort | None:
        """source_priority 값을 Tool 인스턴스로 변환한다."""
        tool_name = _PRIORITY_TO_TOOL.get(source_type)
        if not tool_name:
            logger.warning("tool_router_no_mapping source_type=%s", source_type)
            return None
        tool = self._tool_map.get(tool_name)
        if not tool:
            logger.warning("tool_router_not_found tool_name=%s", tool_name)
            return None
        return tool

    def resolve_ordered(self, source_priority: list[str]) -> list[tuple[str, ToolPort]]:
        """
        source_priority 순서대로 (source_type, Tool) 쌍을 반환한다.
        매핑되지 않는 source_type은 제외된다.
        """
        result: list[tuple[str, ToolPort]] = []
        for source_type in source_priority:
            tool = self.resolve(source_type)
            if tool:
                result.append((source_type, tool))
        logger.info(
            "tool_router_resolved priority=%s matched=%s",
            source_priority,
            [s for s, _ in result],
        )
        return result
