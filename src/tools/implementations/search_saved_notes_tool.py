from typing import Any

from src.infrastructure.cache.redis_client import InMemoryRedisClient
from src.tools.base.tool_contract import ToolContract


class SearchSavedNotesTool(ToolContract):
    name = "search_saved_notes"

    def __init__(self, note_store: InMemoryRedisClient) -> None:
        self.note_store = note_store

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"keyword": "str"}

    @property
    def output_schema(self) -> dict[str, Any]:
        return {"items": "list[str]"}

    async def _run(self, payload: dict[str, Any]) -> dict[str, Any]:
        keyword = payload.get("keyword", "")
        items = [v for v in self.note_store.values() if keyword.lower() in v.lower()]
        return {"items": items}
