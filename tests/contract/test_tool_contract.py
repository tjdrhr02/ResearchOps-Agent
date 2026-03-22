"""
Tool 계약 테스트.

모든 Tool이 공통 계약(input/output schema, timeout, retry, logging)을
올바르게 구현하는지 검증한다.
"""
import asyncio

import pytest

from src.domain.errors.exceptions import ToolExecutionError
from src.tools.implementations.fetch_article_content_tool import FetchArticleContentTool
from src.tools.implementations.search_news_tool import SearchNewsTool
from src.tools.implementations.search_papers_tool import SearchPapersTool
from src.tools.implementations.search_tech_blogs_tool import SearchTechBlogsTool


# ──────────────────────────────────────────────
# input schema 계약 검증
# ──────────────────────────────────────────────

def test_all_tools_have_name_and_description() -> None:
    tools = [SearchPapersTool(), SearchTechBlogsTool(), SearchNewsTool(), FetchArticleContentTool()]
    for tool in tools:
        assert tool.name, f"{tool.__class__.__name__} must have name"
        assert tool.description, f"{tool.__class__.__name__} must have description"


def test_all_tools_have_args_schema() -> None:
    tools = [SearchPapersTool(), SearchTechBlogsTool(), SearchNewsTool(), FetchArticleContentTool()]
    for tool in tools:
        assert tool.args_schema is not None, f"{tool.__class__.__name__} must have args_schema"


def test_all_search_tools_have_query_in_input_schema() -> None:
    search_tools = [SearchPapersTool(), SearchTechBlogsTool(), SearchNewsTool()]
    for tool in search_tools:
        fields = tool.args_schema.model_fields
        assert "query" in fields, f"{tool.name} args_schema must have 'query' field"
        assert "limit" in fields, f"{tool.name} args_schema must have 'limit' field"


def test_fetch_tool_has_url_in_input_schema() -> None:
    tool = FetchArticleContentTool()
    assert "url" in tool.args_schema.model_fields


def test_all_tools_have_output_schema() -> None:
    tools = [SearchPapersTool(), SearchTechBlogsTool(), SearchNewsTool(), FetchArticleContentTool()]
    for tool in tools:
        schema = tool.output_schema
        assert isinstance(schema, dict)
        assert len(schema) > 0, f"{tool.name} output_schema must not be empty"


# ──────────────────────────────────────────────
# 기본 실행 검증
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_papers_returns_items() -> None:
    tool = SearchPapersTool()
    result = await tool.run({"query": "agentic RAG", "limit": 2})
    assert "items" in result
    assert len(result["items"]) == 2
    item = result["items"][0]
    assert item["source_type"] == "paper"
    assert item["source_id"]
    assert item["url"].startswith("https://")


@pytest.mark.asyncio
async def test_search_tech_blogs_returns_items() -> None:
    tool = SearchTechBlogsTool()
    result = await tool.run({"query": "LangChain production", "limit": 2})
    assert "items" in result
    assert len(result["items"]) == 2
    assert result["items"][0]["source_type"] == "tech_blog"


@pytest.mark.asyncio
async def test_search_news_returns_items() -> None:
    tool = SearchNewsTool()
    result = await tool.run({"query": "LLM observability", "limit": 2})
    assert "items" in result
    assert result["items"][0]["source_type"] == "news"


@pytest.mark.asyncio
async def test_fetch_article_content_returns_content() -> None:
    tool = FetchArticleContentTool()
    result = await tool.run({"url": "https://example.com/article/rag-patterns"})
    assert "content" in result
    assert "word_count" in result
    assert result["word_count"] > 0
    assert result["url"] == "https://example.com/article/rag-patterns"


# ──────────────────────────────────────────────
# timeout 검증
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tool_raises_on_timeout() -> None:
    class SlowTool(SearchPapersTool):
        name: str = "slow_papers"
        timeout_seconds: float = 0.01
        retry_count: int = 0

        async def _execute(self, payload):  # noqa: ANN001
            await asyncio.sleep(5)
            return {"items": []}

    tool = SlowTool()
    with pytest.raises(ToolExecutionError):
        await tool.run({"query": "will timeout", "limit": 1})


# ──────────────────────────────────────────────
# retry 검증
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tool_retries_on_failure() -> None:
    call_count = 0

    class FlakyTool(SearchPapersTool):
        name: str = "flaky_papers"
        retry_count: int = 2

        async def _execute(self, payload):  # noqa: ANN001
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError(f"flaky error attempt={call_count}")
            return {"items": []}

    tool = FlakyTool()
    result = await tool.run({"query": "retry test", "limit": 1})
    assert result["items"] == []
    assert call_count == 3


# ──────────────────────────────────────────────
# LangChain _arun 호환 검증
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_papers_langchain_arun_compat() -> None:
    tool = SearchPapersTool()
    result_str = await tool._arun(query="LangChain agent", limit=1)
    assert "items" in result_str
