"""
SynthesizerAgent 유닛 테스트.

테스트 대상:
  - SynthesisContextBuilder: source_context / evidence_context 생성
  - ContextFallback: LLM 없이 rule-based synthesis
  - SynthesizerAgent: fallback 모드 및 LLM 모드 (FakeLLM 사용)

모든 테스트는 외부 LLM 없이 동작한다.
"""
import json

import pytest

from src.agents.synthesizer.synthesizer_agent import (
    ContextFallback,
    SynthesisContextBuilder,
    SynthesisSchema,
    SynthesizerAgent,
)
from src.application.dto.agent_io import CollectorOutput, SynthesizerOutput
from src.domain.models.evidence_chunk import EvidenceChunk
from src.domain.models.source_document import SourceDocument


# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────

def _make_doc(
    source_id: str = "d1",
    source_type: str = "blog",
    title: str = "Test Document",
    content: str = "This is test content about AI systems.",
    url: str = "https://example.com",
) -> SourceDocument:
    return SourceDocument(
        source_id=source_id, source_type=source_type,
        title=title, url=url, content=content,
    )


def _make_chunk(
    chunk_id: str = "c1",
    source_id: str = "d1",
    score: float = 0.8,
    content: str = "Evidence content about AI.",
    citation: str | None = "[1] Test — https://example.com (blog)",
) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk_id, source_id=source_id,
        content=content, score=score, citation=citation,
    )


def _make_collected(docs: list[SourceDocument] | None = None) -> CollectorOutput:
    return CollectorOutput(documents=docs or [_make_doc()])


def _make_synthesis_json(
    trend_summary: str = "AI is rapidly evolving.",
    key_claims: list[str] | None = None,
    source_comparisons: list[str] | None = None,
    open_questions: list[str] | None = None,
) -> str:
    return json.dumps({
        "trend_summary": trend_summary,
        "key_claims": key_claims or ["[paper] Transformers dominate NLP."],
        "source_comparisons": source_comparisons or ["Papers focus on theory, blogs on practice."],
        "open_questions": open_questions or ["What are the scaling limits?"],
    })


class FakeLLMResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeLLM:
    def __init__(self, response: str) -> None:
        self._response = response

    async def ainvoke(self, _prompt: str) -> FakeLLMResponse:
        return FakeLLMResponse(self._response)


class FailingLLM:
    async def ainvoke(self, _prompt: str) -> None:
        raise RuntimeError("LLM connection failed")


# ──────────────────────────────────────────────
# SynthesisContextBuilder 테스트
# ──────────────────────────────────────────────

class TestSynthesisContextBuilder:
    def test_source_context_contains_title(self):
        builder = SynthesisContextBuilder()
        docs = [_make_doc(title="RAG Architecture Overview")]
        ctx = builder.build_source_context(docs)
        assert "RAG Architecture Overview" in ctx

    def test_source_context_groups_by_type(self):
        builder = SynthesisContextBuilder()
        docs = [
            _make_doc("d1", source_type="paper", title="Paper 1"),
            _make_doc("d2", source_type="blog", title="Blog 1"),
            _make_doc("d3", source_type="paper", title="Paper 2"),
        ]
        ctx = builder.build_source_context(docs)
        assert "PAPER" in ctx
        assert "BLOG" in ctx

    def test_source_context_empty_docs(self):
        builder = SynthesisContextBuilder()
        ctx = builder.build_source_context([])
        assert "no sources" in ctx.lower()

    def test_source_context_snippet_length(self):
        builder = SynthesisContextBuilder()
        long_content = "x" * 1000
        docs = [_make_doc(content=long_content)]
        ctx = builder.build_source_context(docs)
        # snippet은 300자로 잘린다
        assert long_content not in ctx
        assert "x" * 300 in ctx

    def test_evidence_context_contains_score(self):
        builder = SynthesisContextBuilder()
        chunks = [_make_chunk(score=0.91)]
        ctx = builder.build_evidence_context(chunks)
        assert "0.91" in ctx

    def test_evidence_context_sorted_by_score(self):
        builder = SynthesisContextBuilder()
        chunks = [
            _make_chunk("c1", score=0.5, content="lower relevance"),
            _make_chunk("c2", score=0.9, content="higher relevance"),
        ]
        ctx = builder.build_evidence_context(chunks)
        assert ctx.index("0.90") < ctx.index("0.50")

    def test_evidence_context_empty(self):
        builder = SynthesisContextBuilder()
        ctx = builder.build_evidence_context([])
        assert "no rag evidence" in ctx.lower()

    def test_evidence_context_includes_citation(self):
        builder = SynthesisContextBuilder()
        chunks = [_make_chunk(citation="[1] MyPaper — https://arxiv.org (paper)")]
        ctx = builder.build_evidence_context(chunks)
        assert "MyPaper" in ctx


# ──────────────────────────────────────────────
# ContextFallback 테스트
# ──────────────────────────────────────────────

class TestContextFallback:
    def _run(
        self,
        docs: list[SourceDocument] | None = None,
        evidence: list[EvidenceChunk] | None = None,
        topic: str = "AI systems",
    ) -> SynthesizerOutput:
        fallback = ContextFallback()
        return fallback.run(
            docs=docs or [_make_doc()],
            evidence=evidence or [_make_chunk()],
            research_topic=topic,
        )

    def test_returns_synthesizer_output(self):
        result = self._run()
        assert isinstance(result, SynthesizerOutput)

    def test_trend_summary_contains_topic(self):
        result = self._run(topic="vector databases")
        assert "vector databases" in result.trend_summary

    def test_trend_summary_not_empty(self):
        result = self._run()
        assert result.trend_summary.strip() != ""

    def test_claims_from_docs(self):
        docs = [_make_doc(title=f"Source {i}") for i in range(5)]
        result = self._run(docs=docs)
        assert len(result.claims) > 0
        assert all(isinstance(c, str) for c in result.claims)

    def test_claims_include_source_type(self):
        docs = [_make_doc(source_type="paper", title="Important Paper")]
        result = self._run(docs=docs)
        assert any("paper" in c for c in result.claims)

    def test_comparisons_for_mixed_types(self):
        docs = [
            _make_doc("d1", source_type="paper"),
            _make_doc("d2", source_type="blog"),
            _make_doc("d3", source_type="news"),
        ]
        result = self._run(docs=docs)
        assert len(result.comparisons) >= 2

    def test_comparisons_paper_mentioned(self):
        docs = [_make_doc("d1", source_type="paper")]
        result = self._run(docs=docs)
        assert any("paper" in c.lower() or "academic" in c.lower() for c in result.comparisons)

    def test_evidence_sorted_by_score(self):
        evidence = [
            _make_chunk("c1", score=0.3),
            _make_chunk("c2", score=0.9),
            _make_chunk("c3", score=0.6),
        ]
        result = self._run(evidence=evidence)
        scores = [e.score for e in result.evidence]
        assert scores == sorted(scores, reverse=True)

    def test_open_questions_not_empty(self):
        result = self._run()
        assert len(result.open_questions) > 0

    def test_empty_docs_no_crash(self):
        result = self._run(docs=[], evidence=[])
        assert isinstance(result, SynthesizerOutput)


# ──────────────────────────────────────────────
# SynthesizerAgent 테스트
# ──────────────────────────────────────────────

class TestSynthesizerAgent:
    @pytest.mark.asyncio
    async def test_fallback_when_no_llm(self):
        agent = SynthesizerAgent(llm=None)
        result = await agent.run(
            collected=_make_collected(),
            retrieved=[_make_chunk()],
            research_topic="RAG systems",
        )
        assert isinstance(result, SynthesizerOutput)
        assert result.trend_summary != ""

    @pytest.mark.asyncio
    async def test_fallback_returns_evidence(self):
        agent = SynthesizerAgent(llm=None)
        chunks = [_make_chunk(f"c{i}", score=0.5 + i * 0.1) for i in range(3)]
        result = await agent.run(
            collected=_make_collected(),
            retrieved=chunks,
        )
        assert len(result.evidence) > 0

    @pytest.mark.asyncio
    async def test_llm_mode_parses_trend_summary(self):
        fake_json = _make_synthesis_json(trend_summary="LLMs are transforming research workflows.")
        agent = SynthesizerAgent(llm=FakeLLM(fake_json))
        result = await agent.run(
            collected=_make_collected(),
            retrieved=[_make_chunk()],
            research_topic="LLM research",
        )
        assert result.trend_summary == "LLMs are transforming research workflows."

    @pytest.mark.asyncio
    async def test_llm_mode_parses_claims(self):
        fake_json = _make_synthesis_json(
            key_claims=["[paper] Attention is all you need.", "[blog] RAG improves grounding."]
        )
        agent = SynthesizerAgent(llm=FakeLLM(fake_json))
        result = await agent.run(
            collected=_make_collected(),
            retrieved=[_make_chunk()],
        )
        assert len(result.claims) == 2
        assert "[paper]" in result.claims[0]

    @pytest.mark.asyncio
    async def test_llm_mode_parses_comparisons(self):
        fake_json = _make_synthesis_json(
            source_comparisons=["Papers focus on theory.", "Blogs focus on practice."]
        )
        agent = SynthesizerAgent(llm=FakeLLM(fake_json))
        result = await agent.run(
            collected=_make_collected(),
            retrieved=[_make_chunk()],
        )
        assert len(result.comparisons) == 2

    @pytest.mark.asyncio
    async def test_llm_mode_parses_open_questions(self):
        fake_json = _make_synthesis_json(
            open_questions=["How does this scale?", "What are the failure modes?"]
        )
        agent = SynthesizerAgent(llm=FakeLLM(fake_json))
        result = await agent.run(
            collected=_make_collected(),
            retrieved=[_make_chunk()],
        )
        assert len(result.open_questions) == 2

    @pytest.mark.asyncio
    async def test_llm_mode_top5_evidence(self):
        chunks = [_make_chunk(f"c{i}", score=i * 0.1) for i in range(10)]
        fake_json = _make_synthesis_json()
        agent = SynthesizerAgent(llm=FakeLLM(fake_json))
        result = await agent.run(
            collected=_make_collected(),
            retrieved=chunks,
        )
        assert len(result.evidence) <= 5
        scores = [e.score for e in result.evidence]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back(self):
        agent = SynthesizerAgent(llm=FailingLLM())
        result = await agent.run(
            collected=_make_collected([_make_doc()]),
            retrieved=[_make_chunk()],
            research_topic="fallback test",
        )
        assert isinstance(result, SynthesizerOutput)
        assert result.trend_summary != ""

    @pytest.mark.asyncio
    async def test_empty_collected_docs(self):
        agent = SynthesizerAgent(llm=None)
        result = await agent.run(
            collected=_make_collected([]),
            retrieved=[],
        )
        assert isinstance(result, SynthesizerOutput)
        assert result.evidence == []

    @pytest.mark.asyncio
    async def test_multi_type_docs_fallback(self):
        docs = [
            _make_doc("d1", source_type="paper", title="Paper A"),
            _make_doc("d2", source_type="blog", title="Blog B"),
            _make_doc("d3", source_type="news", title="News C"),
        ]
        agent = SynthesizerAgent(llm=None)
        result = await agent.run(
            collected=_make_collected(docs),
            retrieved=[_make_chunk()],
            research_topic="multi-source test",
        )
        # 3가지 source type에 대한 comparison이 모두 있어야 한다
        joined = " ".join(result.comparisons).lower()
        assert "paper" in joined or "academic" in joined
        assert "blog" in joined
        assert "news" in joined

    @pytest.mark.asyncio
    async def test_llm_invalid_json_falls_back(self):
        agent = SynthesizerAgent(llm=FakeLLM("this is not valid json"))
        result = await agent.run(
            collected=_make_collected(),
            retrieved=[_make_chunk()],
            research_topic="json error test",
        )
        assert isinstance(result, SynthesizerOutput)
        assert result.trend_summary != ""

    @pytest.mark.asyncio
    async def test_output_has_all_fields(self):
        agent = SynthesizerAgent(llm=None)
        result = await agent.run(
            collected=_make_collected(),
            retrieved=[_make_chunk()],
        )
        assert hasattr(result, "trend_summary")
        assert hasattr(result, "claims")
        assert hasattr(result, "comparisons")
        assert hasattr(result, "open_questions")
        assert hasattr(result, "evidence")
