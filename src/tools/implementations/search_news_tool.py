from typing import Any
from uuid import uuid4

from src.tools.base.tool_contract import ToolContract


class SearchNewsTool(ToolContract):
    name = "search_news"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"query": "str", "limit": "int"}

    @property
    def output_schema(self) -> dict[str, Any]:
        return {"items": "list[SourceDocument]"}

    async def _run(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = payload["query"]
        sid = str(uuid4())
        return {
            "items": [
                {
                    "source_id": sid,
                    "source_type": "news",
                    "title": f"Industry news: {query}",
                    "url": f"https://example.org/news/{sid}",
                    "content": f"Recent market movement around {query}.",
                    "metadata": {"provider": "mock-news-api"},
                }
            ]
        }
