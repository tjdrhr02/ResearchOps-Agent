"""
search_news Tool.

최신 뉴스 검색 Tool — DuckDuckGo News 연동.
입력: query(검색어), limit(결과 수)
출력: items(SourceDocument 목록)
"""
import asyncio
import hashlib
import logging
from typing import Any

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
    timeout_seconds: float = 12.0
    retry_count: int = 2

    @property
    def output_schema(self) -> dict[str, Any]:
        return {
            "items": "list[SourceDocument]",
            "fields": ["source_id", "source_type", "title", "url", "content", "metadata"],
        }

    async def _execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = payload["query"]
        limit = max(1, min(int(payload.get("limit", 3)), 10))

        logger.info("search_news_execute query=%s limit=%s", query, limit)

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, self._ddg_news, query, limit)

        items = []
        for r in results:
            url = r.get("url", "") or r.get("link", "")
            title = r.get("title", "")
            body = r.get("body", "") or r.get("excerpt", "")
            published = r.get("date", "")
            source = r.get("source", "")
            source_id = hashlib.md5(url.encode()).hexdigest()
            items.append(
                {
                    "source_id": source_id,
                    "source_type": "news",
                    "title": title,
                    "url": url,
                    "content": body,
                    "metadata": {
                        "provider": "duckduckgo-news",
                        "query": query,
                        "published": published,
                        "source": source,
                    },
                }
            )

        logger.info("search_news_done query=%s found=%s", query, len(items))
        return {"items": items}

    @staticmethod
    def _ddg_news(query: str, limit: int) -> list[dict]:
        from ddgs import DDGS
        with DDGS() as ddgs:
            return list(ddgs.news(query, max_results=limit))
