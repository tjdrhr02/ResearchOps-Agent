from typing import Any
from uuid import uuid4

from src.tools.base.tool_contract import ToolContract


class SearchTechBlogsTool(ToolContract):
    name = "search_tech_blogs"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"query": "str", "limit": "int"}

    @property
    def output_schema(self) -> dict[str, Any]:
        return {"items": "list[SourceDocument]"}

    async def _run(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = payload["query"]
        limit = int(payload.get("limit", 3))
        items = []
        for i in range(max(1, min(limit, 2))):
            sid = str(uuid4())
            items.append(
                {
                    "source_id": sid,
                    "source_type": "tech_blog",
                    "title": f"Engineering blog {i + 1}: {query}",
                    "url": f"https://example.org/blogs/{sid}",
                    "content": f"Practical implementation notes for {query}.",
                    "metadata": {"provider": "mock-blog-api"},
                }
            )
        return {"items": items}
