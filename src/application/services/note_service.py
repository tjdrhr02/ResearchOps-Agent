from src.tools.implementations.save_research_note_tool import SaveResearchNoteTool
from src.tools.implementations.search_saved_notes_tool import SearchSavedNotesTool


class NoteService:
    def __init__(self, save_tool: SaveResearchNoteTool, search_tool: SearchSavedNotesTool) -> None:
        self.save_tool = save_tool
        self.search_tool = search_tool

    async def save(self, note: str) -> str:
        result = await self.save_tool.run({"note": note})
        return result["note_id"]

    async def search(self, keyword: str) -> list[str]:
        result = await self.search_tool.run({"keyword": keyword})
        return result.get("items", [])
