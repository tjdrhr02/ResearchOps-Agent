"""
search_tech_blogs Tool.

기술 블로그 검색 Tool — DuckDuckGo 텍스트 검색 연동.
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

_TECH_DOMAINS = (
    "site:medium.com OR site:towardsdatascience.com OR site:dev.to "
    "OR site:blog.research.google OR site:ai.googleblog.com "
    "OR site:openai.com/blog OR site:huggingface.co/blog"
)


class SearchTechBlogsInput(BaseModel):
    query: str = Field(description="Topic to search engineering and tech blogs for")
    limit: int = Field(default=3, ge=1, le=10, description="Max number of results")


class SearchTechBlogsTool(ResearchBaseTool):
    name: str = "search_tech_blogs"
    description: str = (
        "Search for technical blog posts about the given topic. "
        "Returns engineering posts from sources like engineering blogs and Medium."
    )
    args_schema: type[BaseModel] = SearchTechBlogsInput
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

        logger.info("search_tech_blogs_execute query=%s limit=%s", query, limit)

        search_query = f"{query} {_TECH_DOMAINS}"

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, self._ddg_search, search_query, limit)

        items = []
        for r in results:
            url = r.get("href", "") or r.get("url", "")
            title = r.get("title", "")
            body = r.get("body", "") or r.get("snippet", "")
            source_id = hashlib.md5(url.encode()).hexdigest()
            items.append(
                {
                    "source_id": source_id,
                    "source_type": "tech_blog",
                    "title": title,
                    "url": url,
                    "content": body,
                    "metadata": {
                        "provider": "duckduckgo",
                        "query": query,
                    },
                }
            )

        logger.info("search_tech_blogs_done query=%s found=%s", query, len(items))
        return {"items": items}

    @staticmethod
    def _ddg_search(query: str, limit: int) -> list[dict]:
        from ddgs import DDGS
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=limit))
