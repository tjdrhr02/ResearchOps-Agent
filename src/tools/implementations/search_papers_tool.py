"""
search_papers Tool.

학술 논문 검색 Tool.
입력: query(검색어), limit(결과 수)
출력: items(SourceDocument 목록)
"""
import logging
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from src.tools.base.tool_contract import ResearchBaseTool

logger = logging.getLogger(__name__)


class SearchPapersInput(BaseModel):
    query: str = Field(description="Research query to search papers for")
    limit: int = Field(default=3, ge=1, le=10, description="Max number of results")


class SearchPapersTool(ResearchBaseTool):
    name: str = "search_papers"
    description: str = (
        "Search for academic papers relevant to the given research query. "
        "Returns a list of paper titles, URLs, and summaries."
    )
    args_schema: type[BaseModel] = SearchPapersInput
    timeout_seconds: float = 10.0
    retry_count: int = 2

    @property
    def output_schema(self) -> dict[str, Any]:
        return {
            "items": "list[SourceDocument]",
            "fields": ["source_id", "source_type", "title", "url", "content", "metadata"],
        }

    async def _execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = payload["query"]
        limit = int(payload.get("limit", 3))
        limit = max(1, min(limit, 10))

        logger.info("search_papers_execute query=%s limit=%s", query, limit)
        items = []
        for i in range(limit):
            sid = str(uuid4())
            items.append(
                {
                    "source_id": sid,
                    "source_type": "paper",
                    "title": f"[Paper {i + 1}] Research on: {query}",
                    "url": f"https://arxiv.org/abs/{sid[:8]}",
                    "content": (
                        f"Abstract: This paper investigates {query}. "
                        f"We propose a novel approach that improves upon prior work."
                    ),
                    "metadata": {
                        "provider": "arxiv-mock",
                        "query": query,
                        "rank": str(i + 1),
                    },
                }
            )
        logger.info("search_papers_done query=%s found=%s", query, len(items))
        return {"items": items}
