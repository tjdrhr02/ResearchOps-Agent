from typing import Any

from src.tools.base.tool_contract import ToolContract


class FetchArticleContentTool(ToolContract):
    name = "fetch_article_content"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"url": "str"}

    @property
    def output_schema(self) -> dict[str, Any]:
        return {"content": "str"}

    async def _run(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = payload.get("url", "")
        return {"content": f"Fetched article body from {url}"}
