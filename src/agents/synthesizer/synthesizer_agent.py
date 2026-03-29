"""
SynthesizerAgent.

수집된 SourceDocument와 RAG EvidenceChunk를 분석하여 구조화된 합성 결과를 생성한다.

처리 흐름:
  CollectorOutput + EvidenceChunk[]
    → ContextBuilder  (소스 유형별 컨텍스트 문자열 구성)
    → PromptTemplate  (SYNTHESIZER_PROMPT 렌더링)
    → LLM.ainvoke()   (비동기 LLM 호출)
    → PydanticOutputParser (JSON → SynthesisSchema)
    → SynthesizerOutput

출력 구조:
  trend_summary  — 전체 트렌드 요약 (2-3문장)
  claims         — 출처 기반 핵심 주장 목록
  comparisons    — source type별 관점 차이 분석
  open_questions — 미해결 질문 목록
  evidence       — 관련성 높은 EvidenceChunk 목록

LLM 없이도 동작:
  ContextFallback이 source_type별 분류와 evidence 점수를 기반으로
  rule-based synthesis를 수행한다.
"""
import logging
from collections import defaultdict
from typing import Any, Protocol

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from src.agents.prompts.templates import SYNTHESIZER_PROMPT
from src.application.dto.agent_io import CollectorOutput, SynthesizerOutput
from src.domain.models.evidence_chunk import EvidenceChunk
from src.domain.models.source_document import SourceDocument

logger = logging.getLogger(__name__)

_SNIPPET_CHARS = 300  # 컨텍스트에 포함할 소스 당 최대 문자 수
_MAX_SOURCES_IN_CONTEXT = 12  # 컨텍스트에 포함할 최대 소스 수
_MAX_EVIDENCE_IN_CONTEXT = 6  # 컨텍스트에 포함할 최대 evidence 수


# ──────────────────────────────────────────────
# LangChain 호환 Protocol
# ──────────────────────────────────────────────

class SynthesizerRunnable(Protocol):
    async def ainvoke(self, input_data: Any) -> Any:
        raise NotImplementedError


# ──────────────────────────────────────────────
# LLM 출력 파싱용 Pydantic 스키마
# ──────────────────────────────────────────────

class SynthesisSchema(BaseModel):
    """LLM이 반환하는 JSON 구조. PydanticOutputParser가 이 스키마를 기준으로 파싱한다."""

    trend_summary: str = Field(
        description="2-3 sentence summary of the dominant trend across all sources."
    )
    key_claims: list[str] = Field(
        description="3-5 key claims directly supported by the sources. "
                    "Each claim must reference its source type in brackets, e.g. [paper], [blog], [news].",
        default_factory=list,
    )
    source_comparisons: list[str] = Field(
        description="2-3 sentences describing how papers, blogs, and news differ in perspective.",
        default_factory=list,
    )
    open_questions: list[str] = Field(
        description="2-3 open questions that remain unanswered by the sources.",
        default_factory=list,
    )


# ──────────────────────────────────────────────
# 컨텍스트 빌더
# ──────────────────────────────────────────────

class SynthesisContextBuilder:
    """
    SourceDocument와 EvidenceChunk를 LLM에 전달하기 위한 텍스트 컨텍스트로 변환한다.

    source_context: 소스 유형별로 그룹화된 문서 목록
    evidence_context: relevance score 순서의 evidence 스니펫
    """

    def build_source_context(self, docs: list[SourceDocument]) -> str:
        by_type: dict[str, list[SourceDocument]] = defaultdict(list)
        for doc in docs[:_MAX_SOURCES_IN_CONTEXT]:
            by_type[doc.source_type].append(doc)

        lines: list[str] = []
        for source_type, type_docs in sorted(by_type.items()):
            lines.append(f"### {source_type.upper()} ({len(type_docs)} sources)")
            for i, doc in enumerate(type_docs, start=1):
                snippet = (doc.content or doc.title)[:_SNIPPET_CHARS]
                lines.append(f"[{i}] {doc.title}")
                lines.append(f"    URL: {doc.url}")
                lines.append(f"    Snippet: {snippet}")
                lines.append("")

        return "\n".join(lines) if lines else "(no sources collected)"

    def build_evidence_context(self, evidence: list[EvidenceChunk]) -> str:
        top = sorted(evidence, key=lambda e: e.score, reverse=True)[:_MAX_EVIDENCE_IN_CONTEXT]

        if not top:
            return "(no RAG evidence retrieved)"

        lines: list[str] = []
        for i, chunk in enumerate(top, start=1):
            citation = chunk.citation or "unknown source"
            lines.append(f"[E{i}] score={chunk.score:.2f} | {citation}")
            lines.append(f"     {chunk.content[:_SNIPPET_CHARS]}")
            lines.append("")

        return "\n".join(lines)


# ──────────────────────────────────────────────
# Fallback Synthesizer (LLM 없을 때)
# ──────────────────────────────────────────────

class ContextFallback:
    """
    LLM 없이 source_type 분류와 evidence score를 기반으로
    rule-based synthesis를 수행한다.
    """

    def run(
        self,
        docs: list[SourceDocument],
        evidence: list[EvidenceChunk],
        research_topic: str,
    ) -> SynthesizerOutput:
        by_type: dict[str, list[SourceDocument]] = defaultdict(list)
        for doc in docs:
            by_type[doc.source_type].append(doc)

        type_counts = {t: len(d) for t, d in by_type.items()}
        dominant_type = max(type_counts, key=type_counts.get) if type_counts else "general"

        trend_summary = (
            f"Based on {len(docs)} collected sources across "
            f"{len(by_type)} source type(s), the topic '{research_topic}' "
            f"is primarily covered through {dominant_type} sources."
        )

        claims = [
            f"[{doc.source_type}] {doc.title}"
            for doc in docs[:5]
        ]

        comparisons = self._build_comparisons(by_type)
        open_questions = [
            f"What are the practical limitations of {research_topic} in production?",
            f"How does {research_topic} scale under real-world constraints?",
        ]

        top_evidence = sorted(evidence, key=lambda e: e.score, reverse=True)[:5]

        return SynthesizerOutput(
            trend_summary=trend_summary,
            claims=claims,
            comparisons=comparisons,
            open_questions=open_questions,
            evidence=top_evidence,
        )

    def _build_comparisons(self, by_type: dict[str, list[SourceDocument]]) -> list[str]:
        comparisons: list[str] = []
        if "paper" in by_type:
            comparisons.append(
                f"Academic papers ({len(by_type['paper'])} sources) focus on "
                "theoretical foundations and empirical benchmarks."
            )
        if "blog" in by_type:
            comparisons.append(
                f"Tech blogs ({len(by_type['blog'])} sources) emphasize "
                "practical implementation and engineering tradeoffs."
            )
        if "news" in by_type:
            comparisons.append(
                f"News articles ({len(by_type['news'])} sources) highlight "
                "industry adoption and business impact."
            )
        if not comparisons:
            comparisons.append("No multi-source comparison available with current data.")
        return comparisons


# ──────────────────────────────────────────────
# SynthesizerAgent
# ──────────────────────────────────────────────

class SynthesizerAgent:
    """
    LangChain 기반 Synthesizer Agent.

    llm이 주입되면 실제 LLM을 사용하고, 없으면 ContextFallback을 실행한다.
    두 경로 모두 동일한 SynthesizerOutput 형식을 반환한다.
    """

    def __init__(self, llm: SynthesizerRunnable | None = None) -> None:
        self.llm = llm
        self.output_parser = PydanticOutputParser(pydantic_object=SynthesisSchema)
        self.prompt_template = PromptTemplate(
            template=SYNTHESIZER_PROMPT,
            input_variables=[
                "source_context",
                "evidence_context",
                "research_topic",
                "research_objective",
            ],
            partial_variables={
                "format_instructions": self.output_parser.get_format_instructions()
            },
        )
        self.context_builder = SynthesisContextBuilder()
        self.fallback = ContextFallback()

    async def run(
        self,
        collected: CollectorOutput,
        retrieved: list[EvidenceChunk],
        research_topic: str = "",
        research_objective: str = "",
    ) -> SynthesizerOutput:
        docs = collected.documents

        if not self.llm:
            logger.info("synthesizer_fallback_mode doc=%s evidence=%s", len(docs), len(retrieved))
            return self.fallback.run(
                docs=docs,
                evidence=retrieved,
                research_topic=research_topic or "the research topic",
            )

        return await self._run_with_llm(
            docs=docs,
            evidence=retrieved,
            research_topic=research_topic,
            research_objective=research_objective,
        )

    async def _run_with_llm(
        self,
        docs: list[SourceDocument],
        evidence: list[EvidenceChunk],
        research_topic: str,
        research_objective: str,
    ) -> SynthesizerOutput:
        source_context = self.context_builder.build_source_context(docs)
        evidence_context = self.context_builder.build_evidence_context(evidence)

        prompt = self.prompt_template.format(
            source_context=source_context,
            evidence_context=evidence_context,
            research_topic=research_topic or "the research topic",
            research_objective=research_objective or "Synthesize available evidence.",
        )

        try:
            raw = await self.llm.ainvoke(prompt)
            raw_text = raw.content if hasattr(raw, "content") else str(raw)
            parsed: SynthesisSchema = self.output_parser.parse(raw_text)

            top_evidence = sorted(evidence, key=lambda e: e.score, reverse=True)[:5]

            output = SynthesizerOutput(
                trend_summary=parsed.trend_summary,
                claims=parsed.key_claims,
                comparisons=parsed.source_comparisons,
                open_questions=parsed.open_questions,
                evidence=top_evidence,
            )
            logger.info(
                "synthesizer_llm_success claims=%s comparisons=%s evidence=%s",
                len(output.claims),
                len(output.comparisons),
                len(output.evidence),
            )
            return output

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "synthesizer_llm_failed fallback_used error=%s", exc
            )
            return self.fallback.run(
                docs=docs,
                evidence=evidence,
                research_topic=research_topic or "the research topic",
            )
