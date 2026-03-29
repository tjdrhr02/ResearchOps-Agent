"""
ReporterAgent.

SynthesizerOutput과 PlannerOutput을 받아
최종 Research Brief를 LangChain 기반으로 생성한다.

처리 흐름:
  PlannerOutput + SynthesizerOutput
    → BriefContextBuilder  (섹션별 컨텍스트 문자열 구성)
    → PromptTemplate       (REPORTER_PROMPT 렌더링)
    → LLM.ainvoke()        (비동기 LLM 호출)
    → PydanticOutputParser (JSON → ReportSchema)
    → CitationFormatter    (중복 제거 + 번호 정규화)
    → ResearchBrief        (metadata 포함 최종 출력)

출력 구조:
  executive_summary  — 의사결정자용 3-4문장 요약
  key_trends         — 증거 기반 주요 트렌드
  evidence           — 인용 포함 근거 스니펫
  source_comparison  — source type 관점 차이
  open_questions     — 미해결 연구 방향
  citations          — 번호 기반 참고문헌
  metadata           — BriefMetadata (생성 컨텍스트)

LLM 없이도 동작:
  StructuredFallback이 Synthesizer 결과를 직접 매핑하여
  형식은 다르지만 내용이 완전한 Brief를 반환한다.
"""
import logging
import re
from typing import Any, Protocol

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from src.agents.prompts.templates import REPORTER_PROMPT
from src.application.dto.agent_io import PlannerOutput, SynthesizerOutput
from src.domain.models.evidence_chunk import EvidenceChunk
from src.domain.models.research_brief import BriefMetadata, ResearchBrief

logger = logging.getLogger(__name__)

_MAX_EVIDENCE_IN_PROMPT = 5
_MAX_CLAIMS_IN_PROMPT = 8


# ──────────────────────────────────────────────
# LangChain 호환 Protocol
# ──────────────────────────────────────────────

class ReporterRunnable(Protocol):
    async def ainvoke(self, input_data: Any) -> Any:
        raise NotImplementedError


# ──────────────────────────────────────────────
# LLM 출력 파싱용 Pydantic 스키마
# ──────────────────────────────────────────────

class ReportSchema(BaseModel):
    """LLM이 반환하는 JSON 구조. PydanticOutputParser가 이 스키마를 기준으로 파싱한다."""

    executive_summary: str = Field(
        description="3-4 sentence summary for a non-technical decision-maker."
    )
    key_trends: list[str] = Field(
        description="3-5 key trends, each a single clear sentence with optional citation.",
        default_factory=list,
    )
    evidence_highlights: list[str] = Field(
        description="Top evidence snippets, preserving citation markers like [1], [2].",
        default_factory=list,
    )
    source_comparison: list[str] = Field(
        description="2-3 sentences explaining WHY papers, blogs, and news differ in perspective.",
        default_factory=list,
    )
    open_questions: list[str] = Field(
        description="2-3 actionable research directions framed as questions.",
        default_factory=list,
    )


# ──────────────────────────────────────────────
# 컨텍스트 빌더
# ──────────────────────────────────────────────

class BriefContextBuilder:
    """
    SynthesizerOutput 각 섹션을 REPORTER_PROMPT에 주입할 텍스트로 변환한다.
    """

    def build_claims_text(self, claims: list[str]) -> str:
        if not claims:
            return "(no key claims provided)"
        top = claims[:_MAX_CLAIMS_IN_PROMPT]
        return "\n".join(f"- {c}" for c in top)

    def build_comparisons_text(self, comparisons: list[str]) -> str:
        if not comparisons:
            return "(no source comparison provided)"
        return "\n".join(f"- {c}" for c in comparisons)

    def build_evidence_text(self, evidence: list[EvidenceChunk]) -> str:
        if not evidence:
            return "(no evidence available)"
        top = sorted(evidence, key=lambda e: e.score, reverse=True)[:_MAX_EVIDENCE_IN_PROMPT]
        lines: list[str] = []
        for i, chunk in enumerate(top, start=1):
            citation = chunk.citation or "unknown source"
            lines.append(f"[E{i}] {citation}")
            lines.append(f"     \"{chunk.content[:250]}\"")
            lines.append("")
        return "\n".join(lines)

    def build_open_questions_text(self, questions: list[str]) -> str:
        if not questions:
            return "(no open questions provided)"
        return "\n".join(f"- {q}" for q in questions)


# ──────────────────────────────────────────────
# Citation Formatter
# ──────────────────────────────────────────────

class CitationFormatter:
    """
    EvidenceChunk의 citation 목록에서 중복을 제거하고
    [번호] 형식으로 정규화된 참고문헌 목록을 생성한다.
    """

    def format(self, evidence: list[EvidenceChunk]) -> list[str]:
        seen: dict[str, str] = {}  # source_id → citation
        for chunk in evidence:
            if chunk.citation and chunk.source_id not in seen:
                seen[chunk.source_id] = chunk.citation

        # 번호가 없는 citation에 번호 부여
        result: list[str] = []
        for idx, citation in enumerate(seen.values(), start=1):
            normalized = self._ensure_numbered(citation, idx)
            result.append(normalized)
        return result

    def _ensure_numbered(self, citation: str, num: int) -> str:
        """citation이 [N] 으로 시작하지 않으면 번호를 붙인다."""
        if re.match(r"^\[\d+\]", citation.strip()):
            return citation.strip()
        return f"[{num}] {citation.strip()}"


# ──────────────────────────────────────────────
# Fallback Reporter (LLM 없을 때)
# ──────────────────────────────────────────────

class StructuredFallback:
    """
    LLM 없이 SynthesizerOutput 필드를 직접 매핑하여 ResearchBrief를 생성한다.
    형식은 단순하지만 내용 손실 없이 구조를 유지한다.
    """

    def __init__(self, citation_formatter: CitationFormatter) -> None:
        self.citation_formatter = citation_formatter

    def run(
        self,
        planner_output: PlannerOutput,
        synthesis: SynthesizerOutput,
    ) -> ResearchBrief:
        plan = planner_output.plan

        executive_summary = (
            synthesis.trend_summary
            if synthesis.trend_summary
            else (
                f"This brief covers '{plan.question}' based on "
                f"{len(synthesis.claims)} key claims and "
                f"{len(synthesis.evidence)} evidence items. "
                f"Sources were analyzed across multiple perspectives to identify trends."
            )
        )

        evidence_snippets = [
            f"{chunk.content[:200]} {('— ' + chunk.citation) if chunk.citation else ''}"
            for chunk in sorted(synthesis.evidence, key=lambda e: e.score, reverse=True)[:5]
        ]

        citations = self.citation_formatter.format(synthesis.evidence)

        return ResearchBrief(
            executive_summary=executive_summary,
            key_trends=synthesis.claims,
            evidence=evidence_snippets,
            source_comparison=synthesis.comparisons,
            open_questions=synthesis.open_questions,
            citations=citations,
            metadata=BriefMetadata(
                research_question=plan.question,
                research_type=plan.research_type,
                source_count=0,
                evidence_count=len(synthesis.evidence),
            ),
        )


# ──────────────────────────────────────────────
# ReporterAgent
# ──────────────────────────────────────────────

class ReporterAgent:
    """
    LangChain 기반 Reporter Agent.

    llm이 주입되면 실제 LLM으로 professional brief를 작성하고,
    없으면 StructuredFallback으로 SynthesizerOutput을 직접 매핑한다.
    두 경로 모두 동일한 ResearchBrief 구조를 반환한다.
    """

    def __init__(self, llm: ReporterRunnable | None = None) -> None:
        self.llm = llm
        self.output_parser = PydanticOutputParser(pydantic_object=ReportSchema)
        self.prompt_template = PromptTemplate(
            template=REPORTER_PROMPT,
            input_variables=[
                "research_question",
                "research_type",
                "research_objective",
                "trend_summary",
                "claims_text",
                "comparisons_text",
                "evidence_text",
                "open_questions_text",
            ],
            partial_variables={
                "format_instructions": self.output_parser.get_format_instructions()
            },
        )
        self.context_builder = BriefContextBuilder()
        self.citation_formatter = CitationFormatter()
        self.fallback = StructuredFallback(self.citation_formatter)

    async def run(
        self,
        planner_output: PlannerOutput,
        synthesis: SynthesizerOutput,
        source_count: int = 0,
    ) -> ResearchBrief:
        if not self.llm:
            logger.info(
                "reporter_fallback_mode claims=%s evidence=%s",
                len(synthesis.claims),
                len(synthesis.evidence),
            )
            brief = self.fallback.run(planner_output, synthesis)
            brief.metadata.source_count = source_count
            return brief

        return await self._run_with_llm(planner_output, synthesis, source_count)

    async def _run_with_llm(
        self,
        planner_output: PlannerOutput,
        synthesis: SynthesizerOutput,
        source_count: int,
    ) -> ResearchBrief:
        plan = planner_output.plan

        prompt = self.prompt_template.format(
            research_question=plan.question,
            research_type=plan.research_type,
            research_objective=plan.objective,
            trend_summary=synthesis.trend_summary or "(not provided)",
            claims_text=self.context_builder.build_claims_text(synthesis.claims),
            comparisons_text=self.context_builder.build_comparisons_text(synthesis.comparisons),
            evidence_text=self.context_builder.build_evidence_text(synthesis.evidence),
            open_questions_text=self.context_builder.build_open_questions_text(synthesis.open_questions),
        )

        try:
            raw = await self.llm.ainvoke(prompt)
            raw_text = raw.content if hasattr(raw, "content") else str(raw)
            parsed: ReportSchema = self.output_parser.parse(raw_text)

            citations = self.citation_formatter.format(synthesis.evidence)

            brief = ResearchBrief(
                executive_summary=parsed.executive_summary,
                key_trends=parsed.key_trends,
                evidence=parsed.evidence_highlights,
                source_comparison=parsed.source_comparison,
                open_questions=parsed.open_questions,
                citations=citations,
                metadata=BriefMetadata(
                    research_question=plan.question,
                    research_type=plan.research_type,
                    source_count=source_count,
                    evidence_count=len(synthesis.evidence),
                ),
            )
            logger.info(
                "reporter_llm_success trends=%s evidence=%s citations=%s",
                len(brief.key_trends),
                len(brief.evidence),
                len(brief.citations),
            )
            return brief

        except Exception as exc:  # noqa: BLE001
            logger.warning("reporter_llm_failed fallback_used error=%s", exc)
            brief = self.fallback.run(planner_output, synthesis)
            brief.metadata.source_count = source_count
            return brief
