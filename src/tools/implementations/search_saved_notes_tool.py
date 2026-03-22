"""
search_saved_notes Tool.

저장된 연구 노트를 키워드로 검색하는 Tool.
입력: keyword(검색 키워드)
출력: items(매칭된 노트 목록)
"""
from typing import Any

from pydantic import BaseModel, Field

from src.infrastructure.cache.redis_client import InMemoryRedisClient
from src.tools.base.tool_contract import ResearchBaseTool


class SearchNotesInput(BaseModel):
    keyword: str = Field(default="", description="Keyword to search saved notes")


class SearchSavedNotesTool(ResearchBaseTool):
    name: str = "search_saved_notes"
    description: str = "Search previously saved research notes by keyword."
    args_schema: type[BaseModel] = SearchNotesInput

    note_store: Any  # InMemoryRedisClient (Any to avoid Pydantic conflict)

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, note_store: InMemoryRedisClient) -> None:
        super().__init__(note_store=note_store)

    @property
    def output_schema(self) -> dict[str, Any]:
        return {"items": "list[str]"}

    async def _execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        keyword = payload.get("keyword", "")
        items = [v for v in self.note_store.values() if keyword.lower() in v.lower()]
        return {"items": items}
