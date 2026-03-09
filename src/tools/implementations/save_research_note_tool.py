from typing import Any
from uuid import uuid4

from src.infrastructure.cache.redis_client import InMemoryRedisClient
from src.tools.base.tool_contract import ToolContract


class SaveResearchNoteTool(ToolContract):
    name = "save_research_note"

    def __init__(self, note_store: InMemoryRedisClient) -> None:
        self.note_store = note_store

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"note": "str"}

    @property
    def output_schema(self) -> dict[str, Any]:
        return {"note_id": "str"}

    async def _run(self, payload: dict[str, Any]) -> dict[str, Any]:
        note_id = str(uuid4())
        self.note_store.set(f"note:{note_id}", payload["note"])
        return {"note_id": note_id}
