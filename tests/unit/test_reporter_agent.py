"""
ReporterAgent 유닛 테스트.

테스트 대상:
  - BriefContextBuilder: 섹션별 컨텍스트 문자열 생성
  - CitationFormatter: 중복 제거 + 번호 정규화
  - StructuredFallback: LLM 없이 직접 매핑
  - ReporterAgent: fallback 모드 및 LLM 모드 (FakeLLM 사용)
  - ResearchBrief.to_markdown(): 마크다운 렌더링

모든 테스트는 외부 LLM 없이 동작한다.
"""
import json

import pytest

from src.agents.reporter.reporter_agent import (
    BriefContextBuilder,
    CitationFormatter,
    ReportSchema,
    ReporterAgent,
    StructuredFallback,
)
from src.application.dto.agent_io import PlannerOutput, SynthesizerOutput
from src.domain.models.evidence_chunk import EvidenceChunk
from src.domain.models.research_brief import BriefMetadata, ResearchBrief
from src.domain.models.research_plan import ResearchPlan
from src.domain.models.source_document import SourceDocument


# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────

def _make_plan(
    question: str = "What is RAG?",
    research_type: str = "trend_analysis",
    objective: str = "Understand RAG adoption trends.",
) -> ResearchPlan:
    return ResearchPlan(
        question=question,
        objective=objective,
        research_type=research_type,
        queries=[question, f"{question} recent"],
        focus_topics=["RAG", "retrieval"],
        source_priority=["papers", "blogs"],
    )


def _make_planner_output(question: str = "What is RAG?") -> PlannerOutput:
    return PlannerOutput(plan=_make_plan(question=question))


def _make_chunk(
    chunk_id: str = "c1",
    source_id: str = "s1",
    score: float = 0.8,
    content: str = "RAG combines retrieval with generation.",
    citation: str | None = "[1] RAG Paper — https://arxiv.org (paper, score: 0.80)",
) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk_id, source_id=source_id,
        content=content, score=score, citation=citation,
    )


def _make_synthesis(
    trend_summary: str = "RAG is rapidly becoming the standard for grounded AI.",
    claims: list[str] | None = None,
    comparisons: list[str] | None = None,
    open_questions: list[str] | None = None,
    evidence: list[EvidenceChunk] | None = None,
) -> SynthesizerOutput:
    return SynthesizerOutput(
        trend_summary=trend_summary,
        claims=claims or ["[paper] Transformers dominate.", "[blog] RAG is production-ready."],
        comparisons=comparisons or ["Papers focus on benchmarks, blogs on engineering."],
        open_questions=open_questions or ["How does RAG scale with millions of docs?"],
        evidence=evidence or [_make_chunk()],
    )


def _make_report_json(
    executive_summary: str = "RAG represents a paradigm shift in AI-assisted research.",
    key_trends: list[str] | None = None,
    evidence_highlights: list[str] | None = None,
    source_comparison: list[str] | None = None,
    open_questions: list[str] | None = None,
) -> str:
    return json.dumps({
        "executive_summary": executive_summary,
        "key_trends": key_trends or ["RAG adoption is accelerating."],
        "evidence_highlights": evidence_highlights or ["[1] RAG Paper — key finding."],
        "source_comparison": source_comparison or ["Papers emphasize theory, blogs emphasize practice."],
        "open_questions": open_questions or ["What are the limits of RAG?"],
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
        raise RuntimeError("LLM unavailable")


# ──────────────────────────────────────────────
# BriefContextBuilder 테스트
# ──────────────────────────────────────────────

class TestBriefContextBuilder:
    def test_claims_text_contains_claims(self):
        builder = BriefContextBuilder()
        claims = ["[paper] Claim A", "[blog] Claim B"]
        text = builder.build_claims_text(claims)
        assert "Claim A" in text
        assert "Claim B" in text

    def test_claims_text_empty(self):
        builder = BriefContextBuilder()
        text = builder.build_claims_text([])
        assert "no key claims" in text.lower()

    def test_comparisons_text_contains_comparisons(self):
        builder = BriefContextBuilder()
        text = builder.build_comparisons_text(["Papers focus on benchmarks."])
        assert "benchmarks" in text

    def test_comparisons_text_empty(self):
        builder = BriefContextBuilder()
        text = builder.build_comparisons_text([])
        assert "no source comparison" in text.lower()

    def test_evidence_text_contains_score_order(self):
        builder = BriefContextBuilder()
        chunks = [
            _make_chunk("c1", score=0.5, content="low relevance"),
            _make_chunk("c2", score=0.9, content="high relevance"),
        ]
        text = builder.build_evidence_text(chunks)
        assert text.index("high relevance") < text.index("low relevance")

    def test_evidence_text_empty(self):
        builder = BriefContextBuilder()
        text = builder.build_evidence_text([])
        assert "no evidence" in text.lower()

    def test_evidence_text_truncates_content(self):
        builder = BriefContextBuilder()
        long_content = "word " * 200
        chunks = [_make_chunk(content=long_content)]
        text = builder.build_evidence_text(chunks)
        assert len(text) < len(long_content)

    def test_open_questions_text(self):
        builder = BriefContextBuilder()
        questions = ["How does RAG scale?", "What are failure modes?"]
        text = builder.build_open_questions_text(questions)
        assert "RAG scale" in text
        assert "failure modes" in text

    def test_open_questions_empty(self):
        builder = BriefContextBuilder()
        text = builder.build_open_questions_text([])
        assert "no open questions" in text.lower()


# ──────────────────────────────────────────────
# CitationFormatter 테스트
# ──────────────────────────────────────────────

class TestCitationFormatter:
    def test_formats_citations_from_evidence(self):
        formatter = CitationFormatter()
        chunks = [_make_chunk("c1", "s1", citation="[1] Paper — url (paper)")]
        result = formatter.format(chunks)
        assert len(result) == 1
        assert "Paper" in result[0]

    def test_deduplicates_by_source_id(self):
        formatter = CitationFormatter()
        chunks = [
            _make_chunk("c1", "s1", citation="[1] Paper — url (paper)"),
            _make_chunk("c2", "s1", citation="[1] Paper — url (paper)"),  # 같은 source_id
        ]
        result = formatter.format(chunks)
        assert len(result) == 1

    def test_different_sources_all_included(self):
        formatter = CitationFormatter()
        chunks = [
            _make_chunk("c1", "s1", citation="[1] Source A — url-a (paper)"),
            _make_chunk("c2", "s2", citation="[2] Source B — url-b (blog)"),
        ]
        result = formatter.format(chunks)
        assert len(result) == 2

    def test_adds_number_to_unnumbered_citation(self):
        formatter = CitationFormatter()
        chunks = [_make_chunk("c1", "s1", citation="Some Paper — https://example.com (paper)")]
        result = formatter.format(chunks)
        assert result[0].startswith("[1]")

    def test_preserves_existing_number(self):
        formatter = CitationFormatter()
        chunks = [_make_chunk("c1", "s1", citation="[42] Paper — url (paper)")]
        result = formatter.format(chunks)
        assert result[0].startswith("[42]")

    def test_none_citation_skipped(self):
        formatter = CitationFormatter()
        chunks = [_make_chunk("c1", "s1", citation=None)]
        result = formatter.format(chunks)
        assert result == []

    def test_empty_evidence(self):
        formatter = CitationFormatter()
        result = CitationFormatter().format([])
        assert result == []


# ──────────────────────────────────────────────
# StructuredFallback 테스트
# ──────────────────────────────────────────────

class TestStructuredFallback:
    def _run(
        self,
        question: str = "What is RAG?",
        synthesis: SynthesizerOutput | None = None,
    ) -> ResearchBrief:
        fallback = StructuredFallback(CitationFormatter())
        return fallback.run(
            planner_output=_make_planner_output(question),
            synthesis=synthesis or _make_synthesis(),
        )

    def test_returns_research_brief(self):
        result = self._run()
        assert isinstance(result, ResearchBrief)

    def test_executive_summary_uses_trend_summary(self):
        result = self._run(synthesis=_make_synthesis(trend_summary="RAG is the future."))
        assert "RAG is the future." in result.executive_summary

    def test_executive_summary_fallback_when_empty_trend(self):
        result = self._run(synthesis=_make_synthesis(trend_summary=""))
        assert result.executive_summary.strip() != ""

    def test_key_trends_from_claims(self):
        claims = ["[paper] Claim A", "[blog] Claim B"]
        result = self._run(synthesis=_make_synthesis(claims=claims))
        assert result.key_trends == claims

    def test_source_comparison_preserved(self):
        comparisons = ["Papers focus on theory.", "Blogs on practice."]
        result = self._run(synthesis=_make_synthesis(comparisons=comparisons))
        assert result.source_comparison == comparisons

    def test_open_questions_preserved(self):
        questions = ["How does this scale?"]
        result = self._run(synthesis=_make_synthesis(open_questions=questions))
        assert result.open_questions == questions

    def test_citations_deduplicated(self):
        chunks = [
            _make_chunk("c1", "s1", citation="[1] Paper A"),
            _make_chunk("c2", "s1", citation="[1] Paper A"),  # 중복
            _make_chunk("c3", "s2", citation="[2] Paper B"),
        ]
        result = self._run(synthesis=_make_synthesis(evidence=chunks))
        assert len(result.citations) == 2

    def test_metadata_research_question(self):
        result = self._run(question="What is pgvector?")
        assert result.metadata.research_question == "What is pgvector?"

    def test_metadata_evidence_count(self):
        chunks = [_make_chunk(f"c{i}") for i in range(4)]
        result = self._run(synthesis=_make_synthesis(evidence=chunks))
        assert result.metadata.evidence_count == 4

    def test_evidence_sorted_by_score(self):
        chunks = [
            _make_chunk("c1", score=0.3, content="low"),
            _make_chunk("c2", score=0.9, content="high"),
        ]
        result = self._run(synthesis=_make_synthesis(evidence=chunks))
        if len(result.evidence) >= 2:
            assert "high" in result.evidence[0]


# ──────────────────────────────────────────────
# ResearchBrief.to_markdown() 테스트
# ──────────────────────────────────────────────

class TestResearchBriefMarkdown:
    def _make_brief(self) -> ResearchBrief:
        return ResearchBrief(
            executive_summary="RAG is transforming AI research workflows.",
            key_trends=["RAG adoption is accelerating.", "Vector databases are maturing."],
            evidence=["[1] Evidence snippet one.", "[2] Evidence snippet two."],
            source_comparison=["Papers focus on benchmarks, blogs on practice."],
            open_questions=["What are the scaling limits?"],
            citations=["[1] Paper — https://arxiv.org (paper)", "[2] Blog — https://dev.to (blog)"],
            metadata=BriefMetadata(
                research_question="What is RAG?",
                research_type="trend_analysis",
                source_count=10,
                evidence_count=5,
            ),
        )

    def test_markdown_contains_title(self):
        md = self._make_brief().to_markdown()
        assert "# Research Brief" in md
        assert "What is RAG?" in md

    def test_markdown_contains_executive_summary_section(self):
        md = self._make_brief().to_markdown()
        assert "## Executive Summary" in md
        assert "transforming AI research" in md

    def test_markdown_contains_key_trends(self):
        md = self._make_brief().to_markdown()
        assert "## Key Trends" in md
        assert "RAG adoption is accelerating" in md

    def test_markdown_contains_evidence(self):
        md = self._make_brief().to_markdown()
        assert "## Evidence" in md
        assert "Evidence snippet one" in md

    def test_markdown_contains_source_comparison(self):
        md = self._make_brief().to_markdown()
        assert "## Source Perspective Comparison" in md
        assert "benchmarks" in md

    def test_markdown_contains_open_questions(self):
        md = self._make_brief().to_markdown()
        assert "## Open Questions" in md
        assert "scaling limits" in md

    def test_markdown_contains_citations(self):
        md = self._make_brief().to_markdown()
        assert "## Citations" in md
        assert "[1]" in md

    def test_markdown_contains_metadata_line(self):
        md = self._make_brief().to_markdown()
        assert "Sources: 10" in md
        assert "trend_analysis" in md

    def test_markdown_empty_sections_skipped(self):
        brief = ResearchBrief(
            executive_summary="Summary only.",
            key_trends=[],
            evidence=[],
            source_comparison=[],
            open_questions=[],
            citations=[],
        )
        md = brief.to_markdown()
        assert "## Key Trends" not in md
        assert "## Evidence" not in md
        assert "## Citations" not in md


# ──────────────────────────────────────────────
# ReporterAgent 테스트
# ──────────────────────────────────────────────

class TestReporterAgent:
    @pytest.mark.asyncio
    async def test_fallback_when_no_llm(self):
        agent = ReporterAgent(llm=None)
        result = await agent.run(
            planner_output=_make_planner_output(),
            synthesis=_make_synthesis(),
        )
        assert isinstance(result, ResearchBrief)
        assert result.executive_summary.strip() != ""

    @pytest.mark.asyncio
    async def test_fallback_returns_metadata(self):
        agent = ReporterAgent(llm=None)
        result = await agent.run(
            planner_output=_make_planner_output("Agentic AI"),
            synthesis=_make_synthesis(),
            source_count=7,
        )
        assert result.metadata.research_question == "Agentic AI"
        assert result.metadata.source_count == 7

    @pytest.mark.asyncio
    async def test_llm_mode_executive_summary(self):
        fake_json = _make_report_json(executive_summary="LLMs are transforming enterprise workflows.")
        agent = ReporterAgent(llm=FakeLLM(fake_json))
        result = await agent.run(
            planner_output=_make_planner_output(),
            synthesis=_make_synthesis(),
        )
        assert result.executive_summary == "LLMs are transforming enterprise workflows."

    @pytest.mark.asyncio
    async def test_llm_mode_key_trends(self):
        fake_json = _make_report_json(
            key_trends=["Trend A is emerging.", "Trend B is declining."]
        )
        agent = ReporterAgent(llm=FakeLLM(fake_json))
        result = await agent.run(
            planner_output=_make_planner_output(),
            synthesis=_make_synthesis(),
        )
        assert len(result.key_trends) == 2
        assert "Trend A" in result.key_trends[0]

    @pytest.mark.asyncio
    async def test_llm_mode_evidence_highlights(self):
        fake_json = _make_report_json(
            evidence_highlights=["[1] Key finding about RAG.", "[2] Another finding."]
        )
        agent = ReporterAgent(llm=FakeLLM(fake_json))
        result = await agent.run(
            planner_output=_make_planner_output(),
            synthesis=_make_synthesis(),
        )
        assert len(result.evidence) == 2
        assert "[1]" in result.evidence[0]

    @pytest.mark.asyncio
    async def test_llm_mode_source_comparison(self):
        fake_json = _make_report_json(
            source_comparison=["Papers use rigorous evaluation.", "Blogs use anecdotal evidence."]
        )
        agent = ReporterAgent(llm=FakeLLM(fake_json))
        result = await agent.run(
            planner_output=_make_planner_output(),
            synthesis=_make_synthesis(),
        )
        assert len(result.source_comparison) == 2

    @pytest.mark.asyncio
    async def test_llm_mode_open_questions(self):
        fake_json = _make_report_json(
            open_questions=["What is the latency tradeoff?", "How does chunking affect recall?"]
        )
        agent = ReporterAgent(llm=FakeLLM(fake_json))
        result = await agent.run(
            planner_output=_make_planner_output(),
            synthesis=_make_synthesis(),
        )
        assert len(result.open_questions) == 2

    @pytest.mark.asyncio
    async def test_llm_mode_citations_deduplicated(self):
        chunks = [
            _make_chunk("c1", "s1", citation="[1] Paper A"),
            _make_chunk("c2", "s1", citation="[1] Paper A"),
            _make_chunk("c3", "s2", citation="[2] Paper B"),
        ]
        fake_json = _make_report_json()
        agent = ReporterAgent(llm=FakeLLM(fake_json))
        result = await agent.run(
            planner_output=_make_planner_output(),
            synthesis=_make_synthesis(evidence=chunks),
        )
        assert len(result.citations) == 2

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back(self):
        agent = ReporterAgent(llm=FailingLLM())
        result = await agent.run(
            planner_output=_make_planner_output(),
            synthesis=_make_synthesis(),
        )
        assert isinstance(result, ResearchBrief)
        assert result.executive_summary.strip() != ""

    @pytest.mark.asyncio
    async def test_llm_invalid_json_falls_back(self):
        agent = ReporterAgent(llm=FakeLLM("not json at all"))
        result = await agent.run(
            planner_output=_make_planner_output(),
            synthesis=_make_synthesis(),
        )
        assert isinstance(result, ResearchBrief)

    @pytest.mark.asyncio
    async def test_brief_has_all_required_fields(self):
        agent = ReporterAgent(llm=None)
        result = await agent.run(
            planner_output=_make_planner_output(),
            synthesis=_make_synthesis(),
        )
        for field in ["executive_summary", "key_trends", "evidence",
                      "source_comparison", "open_questions", "citations", "metadata"]:
            assert hasattr(result, field)

    @pytest.mark.asyncio
    async def test_brief_to_markdown_callable(self):
        agent = ReporterAgent(llm=None)
        result = await agent.run(
            planner_output=_make_planner_output("What is pgvector?"),
            synthesis=_make_synthesis(),
        )
        md = result.to_markdown()
        assert "# Research Brief" in md
        assert "pgvector" in md

    @pytest.mark.asyncio
    async def test_metadata_generated_at_set(self):
        agent = ReporterAgent(llm=None)
        result = await agent.run(
            planner_output=_make_planner_output(),
            synthesis=_make_synthesis(),
        )
        assert result.metadata.generated_at != ""
        assert "T" in result.metadata.generated_at  # ISO 형식 확인
