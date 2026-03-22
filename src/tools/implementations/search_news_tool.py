"""
search_news Tool.

최신 뉴스 검색 Tool.
입력: query(검색어), limit(결과 수)
출력: items(SourceDocument 목록)
"""
import logging
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from src.tools.base.tool_contract import ResearchBaseTool

logger = logging.getLogger(__name__)


class SearchNewsInput(BaseModel):
    query: str = Field(description="Topic to search recent news articles for")
    limit: int = Field(default=3, ge=1, le=10, description="Max number of results")


class SearchNewsTool(ResearchBaseTool):
    name: str = "search_news"
    description: str = (
        "Search for recent news articles and industry updates about the given topic. "
        "Returns news headlines, URLs, and article summaries."
    )
    args_schema: type[BaseModel] = SearchNewsInput
    timeout_seconds: float = 8.0
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

        logger.info("search_news_execute query=%s limit=%s", query, limit)
        items = []
        for i in range(limit):
            sid = str(uuid4())
            items.append(
                {
                    "source_id": sid,
                    "source_type": "news",
                    "title": f"[News {i + 1}] Industry update: {query}",
                    "url": f"https://news.example.com/articles/{sid[:8]}",
                    "content": (
                        f"Recent industry news: {query} is gaining traction. "
                        f"Key players are accelerating adoption across enterprise segments."
                    ),
                    "metadata": {
                        "provider": "news-mock",
                        "query": query,
                        "rank": str(i + 1),
                    },
                }
            )
        logger.info("search_news_done query=%s found=%s", query, len(items))
        return {"items": items}
