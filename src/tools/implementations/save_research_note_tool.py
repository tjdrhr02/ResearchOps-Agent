"""
save_research_note Tool.

연구 노트를 저장하는 Tool.
입력: note(노트 내용)
출력: note_id
"""
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from src.infrastructure.cache.redis_client import InMemoryRedisClient
from src.tools.base.tool_contract import ResearchBaseTool


class SaveNoteInput(BaseModel):
    note: str = Field(description="Research note content to save")


class SaveResearchNoteTool(ResearchBaseTool):
    name: str = "save_research_note"
    description: str = "Save a research note for later retrieval."
    args_schema: type[BaseModel] = SaveNoteInput

    note_store: Any  # InMemoryRedisClient (Any to avoid Pydantic conflict)

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, note_store: InMemoryRedisClient) -> None:
        super().__init__(note_store=note_store)

    @property
    def output_schema(self) -> dict[str, Any]:
        return {"note_id": "str"}

    async def _execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        note_id = str(uuid4())
        self.note_store.set(f"note:{note_id}", payload["note"])
        return {"note_id": note_id}
