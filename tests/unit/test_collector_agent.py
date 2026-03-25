"""
Collector Agent 테스트.

Tool routing, 중복 제거, metadata 정규화,
partial failure 처리를 검증한다.
"""
import pytest

from src.agents.collector.collector_agent import CollectorAgent
from src.agents.collector.document_normalizer import DocumentNormalizer
from src.agents.collector.duplicate_filter import DuplicateFilter
from src.agents.collector.tool_router import ToolRouter
from src.application.dto.agent_io import PlannerOutput
from src.domain.models.research_plan import ResearchPlan
from src.domain.models.source_document import SourceDocument
from src.domain.ports.tool_port import ToolPort


# ──────────────────────────────────────────────
# Fake Tools
# ──────────────────────────────────────────────

def _make_plan(
    source_priority: list[str] | None = None,
    queries: list[str] | None = None,
) -> PlannerOutput:
    return PlannerOutput(
        plan=ResearchPlan(
            question="agentic RAG",
            objective="test",
            research_type="trend_analysis",
            queries=queries or ["agentic RAG", "RAG evaluation"],
            focus_topics=["RAG"],
            source_priority=source_priority or ["papers", "tech_blogs", "news"],
            source_types=source_priority or ["papers", "tech_blogs", "news"],
        )
    )


class FakeTool(ToolPort):
    def __init__(self, name: str, items: list[dict]) -> None:
        self._name = name
        self._items = items
        self._call_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def input_schema(self) -> dict:
        return {"query": "str", "limit": "int"}

    @property
    def output_schema(self) -> dict:
        return {"items": "list"}

    async def run(self, payload: dict) -> dict:
        self._call_count += 1
        return {"items": self._items}


class FailingTool(ToolPort):
    @property
    def name(self) -> str:
        return "search_news"

    @property
    def input_schema(self) -> dict:
        return {}

    @property
    def output_schema(self) -> dict:
        return {}

    async def run(self, payload: dict) -> dict:
        raise RuntimeError("external API timeout")


# ──────────────────────────────────────────────
# ToolRouter 테스트
# ──────────────────────────────────────────────

def test_tool_router_resolves_known_sources() -> None:
    tools = [
        FakeTool("search_papers", []),
        FakeTool("search_tech_blogs", []),
        FakeTool("search_news", []),
    ]
    router = ToolRouter(tools=tools)

    assert router.resolve("papers") is not None
    assert router.resolve("tech_blogs") is not None
    assert router.resolve("news") is not None


def test_tool_router_returns_none_for_unknown_source() -> None:
    router = ToolRouter(tools=[])
    assert router.resolve("unknown_source") is None


def test_tool_router_resolve_ordered_preserves_priority() -> None:
    tools = [
        FakeTool("search_papers", []),
        FakeTool("search_news", []),
    ]
    router = ToolRouter(tools=tools)
    pairs = router.resolve_ordered(["news", "papers", "tech_blogs"])

    source_types = [s for s, _ in pairs]
    assert source_types == ["news", "papers"]


# ──────────────────────────────────────────────
# DocumentNormalizer 테스트
# ──────────────────────────────────────────────

def test_normalizer_fills_missing_metadata() -> None:
    normalizer = DocumentNormalizer()
    raw = {
        "source_id": "id-1",
        "source_type": "paper",
        "title": "Test Paper",
        "url": "https://example.com/paper/1",
        "content": "  content here  ",
    }
    doc = normalizer.normalize(raw, source_type="papers", query="RAG")
    assert doc is not None
    assert doc.metadata["query"] == "RAG"
    assert doc.metadata["source_type"] == "papers"
    assert doc.metadata["provider"] == "unknown"
    assert doc.content == "content here"


def test_normalizer_generates_source_id_from_url() -> None:
    normalizer = DocumentNormalizer()
    raw = {
        "source_type": "paper",
        "title": "No ID Paper",
        "url": "https://example.com/no-id",
        "content": "some content",
    }
    doc = normalizer.normalize(raw, source_type="papers", query="test")
    assert doc is not None
    assert len(doc.source_id) == 32


def test_normalizer_returns_none_on_invalid_raw() -> None:
    normalizer = DocumentNormalizer()
    result = normalizer.normalize(raw=12345, source_type="papers", query="q")
    assert result is None


# ──────────────────────────────────────────────
# DuplicateFilter 테스트
# ──────────────────────────────────────────────

def _doc(source_id: str, url: str, title: str) -> SourceDocument:
    return SourceDocument(
        source_id=source_id,
        source_type="paper",
        title=title,
        url=url,
        content="",
    )


def test_dedup_removes_same_url() -> None:
    docs = [
        _doc("a", "https://example.com/paper/1", "Paper A"),
        _doc("b", "https://example.com/paper/1", "Paper B"),
        _doc("c", "https://example.com/paper/2", "Paper C"),
    ]
    result = DuplicateFilter().filter(docs)
    assert len(result) == 2
    assert result[0].source_id == "a"


def test_dedup_normalizes_trailing_slash() -> None:
    docs = [
        _doc("a", "https://example.com/paper/1/", "Title A"),
        _doc("b", "https://example.com/paper/1", "Title B"),
    ]
    result = DuplicateFilter().filter(docs)
    assert len(result) == 1


def test_dedup_removes_same_title_when_no_url() -> None:
    docs = [
        _doc("a", "", "Same Title"),
        _doc("b", "", "Same Title"),
    ]
    result = DuplicateFilter().filter(docs)
    assert len(result) == 1


def test_dedup_keeps_all_unique() -> None:
    docs = [_doc(str(i), f"https://example.com/{i}", f"Title {i}") for i in range(5)]
    result = DuplicateFilter().filter(docs)
    assert len(result) == 5


# ──────────────────────────────────────────────
# CollectorAgent 통합 테스트
# ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_collector_routes_by_source_priority() -> None:
    paper_tool = FakeTool("search_papers", [
        {"source_id": "p1", "source_type": "paper",
         "title": "Paper 1", "url": "https://arxiv.org/p1", "content": "c"},
    ])
    blog_tool = FakeTool("search_tech_blogs", [
        {"source_id": "b1", "source_type": "tech_blog",
         "title": "Blog 1", "url": "https://blog.com/b1", "content": "c"},
    ])
    news_tool = FakeTool("search_news", [])

    agent = CollectorAgent(tools=[paper_tool, blog_tool, news_tool])
    plan = _make_plan(
        source_priority=["papers", "tech_blogs"],
        queries=["agentic RAG"],
    )
    output = await agent.run(plan, max_sources=10)

    assert paper_tool._call_count == 1
    assert blog_tool._call_count == 1
    assert news_tool._call_count == 0
    assert len(output.documents) == 2


@pytest.mark.asyncio
async def test_collector_deduplicates_across_queries() -> None:
    shared_url = "https://example.com/paper/same"
    tool = FakeTool("search_papers", [
        {"source_id": "p1", "source_type": "paper",
         "title": "Dup Paper", "url": shared_url, "content": "c"},
    ])
    agent = CollectorAgent(tools=[tool])
    # 두 query 모두 같은 URL을 반환 → dedup 후 1개
    plan = _make_plan(source_priority=["papers"], queries=["query1", "query2"])
    output = await agent.run(plan, max_sources=10)

    assert len(output.documents) == 1


@pytest.mark.asyncio
async def test_collector_continues_on_tool_failure() -> None:
    paper_tool = FakeTool("search_papers", [
        {"source_id": "p1", "source_type": "paper",
         "title": "Paper 1", "url": "https://arxiv.org/p1", "content": "c"},
    ])
    failing_news = FailingTool()

    agent = CollectorAgent(tools=[paper_tool, failing_news])
    plan = _make_plan(source_priority=["papers", "news"], queries=["RAG"])
    output = await agent.run(plan, max_sources=10)

    # news 실패해도 papers 결과는 수집 완료
    assert len(output.documents) == 1


@pytest.mark.asyncio
async def test_collector_respects_max_sources() -> None:
    items = [
        {"source_id": f"p{i}", "source_type": "paper",
         "title": f"Paper {i}", "url": f"https://arxiv.org/p{i}", "content": "c"}
        for i in range(20)
    ]
    tool = FakeTool("search_papers", items)
    agent = CollectorAgent(tools=[tool])
    plan = _make_plan(source_priority=["papers"], queries=["RAG"])
    output = await agent.run(plan, max_sources=5)

    assert len(output.documents) <= 5


@pytest.mark.asyncio
async def test_collector_metadata_enriched() -> None:
    tool = FakeTool("search_papers", [
        {"source_id": "p1", "source_type": "paper",
         "title": "Meta Paper", "url": "https://arxiv.org/p1", "content": "c"},
    ])
    agent = CollectorAgent(tools=[tool])
    plan = _make_plan(source_priority=["papers"], queries=["observability"])
    output = await agent.run(plan, max_sources=5)

    doc = output.documents[0]
    assert doc.metadata["query"] == "observability"
    assert doc.metadata["source_type"] == "papers"
