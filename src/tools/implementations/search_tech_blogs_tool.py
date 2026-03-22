"""
search_tech_blogs Tool.

기술 블로그 검색 Tool.
입력: query(검색어), limit(결과 수)
출력: items(SourceDocument 목록)
"""
import logging
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from src.tools.base.tool_contract import ResearchBaseTool

logger = logging.getLogger(__name__)


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

        logger.info("search_tech_blogs_execute query=%s limit=%s", query, limit)
        items = []
        for i in range(limit):
            sid = str(uuid4())
            items.append(
                {
                    "source_id": sid,
                    "source_type": "tech_blog",
                    "title": f"[Blog {i + 1}] Engineering perspective on: {query}",
                    "url": f"https://engineering.example.com/posts/{sid[:8]}",
                    "content": (
                        f"In this post, we share our team's experience with {query}. "
                        f"We cover implementation details, lessons learned, and best practices."
                    ),
                    "metadata": {
                        "provider": "techblog-mock",
                        "query": query,
                        "rank": str(i + 1),
                    },
                }
            )
        logger.info("search_tech_blogs_done query=%s found=%s", query, len(items))
        return {"items": items}
