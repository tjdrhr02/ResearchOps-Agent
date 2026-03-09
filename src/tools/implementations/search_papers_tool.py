from typing import Any
from uuid import uuid4

from src.tools.base.tool_contract import ToolContract


class SearchPapersTool(ToolContract):
    name = "search_papers"

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
        for i in range(max(1, min(limit, 3))):
            sid = str(uuid4())
            items.append(
                {
                    "source_id": sid,
                    "source_type": "paper",
                    "title": f"Paper insight {i + 1} on {query}",
                    "url": f"https://example.org/papers/{sid}",
                    "content": f"Academic content about {query}.",
                    "metadata": {"provider": "mock-paper-api"},
                }
            )
        return {"items": items}
