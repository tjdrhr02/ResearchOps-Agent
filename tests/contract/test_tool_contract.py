import pytest

from src.tools.implementations.search_papers_tool import SearchPapersTool


@pytest.mark.asyncio
async def test_search_papers_tool_schema_and_output() -> None:
    tool = SearchPapersTool()
    assert "query" in tool.input_schema
    result = await tool.run({"query": "agentic rag", "limit": 2})
    assert "items" in result
    assert len(result["items"]) >= 1
