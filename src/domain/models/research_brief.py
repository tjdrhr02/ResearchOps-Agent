"""
ResearchBrief.

ResearchOps Agent 최종 출력 도메인 모델.

구조:
  executive_summary  — 의사결정자용 핵심 요약 (3-4문장)
  key_trends         — 증거 기반 주요 트렌드 목록
  evidence           — 인용 포함 주요 근거 스니펫 목록
  source_comparison  — source type별 관점 차이 설명
  open_questions     — 미해결 연구 방향 질문
  citations          — 번호 기반 참고문헌 목록
  metadata           — 쿼리, 소스 수, 생성 정보 등

to_markdown()은 구조화된 마크다운 문서를 반환한다.
"""
from datetime import UTC, datetime

from pydantic import BaseModel, Field


class BriefMetadata(BaseModel):
    """Brief 생성 컨텍스트 정보."""

    research_question: str = ""
    research_type: str = "general_research"
    source_count: int = 0
    evidence_count: int = 0
    generated_at: str = Field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    )


class ResearchBrief(BaseModel):
    executive_summary: str
    key_trends: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    source_comparison: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    metadata: BriefMetadata = Field(default_factory=BriefMetadata)

    def to_markdown(self) -> str:
        """구조화된 마크다운 Research Brief를 반환한다."""
        sections: list[str] = []

        # 헤더
        title = self.metadata.research_question or "Research Brief"
        sections.append(f"# Research Brief: {title}")
        sections.append(
            f"*Generated: {self.metadata.generated_at} | "
            f"Sources: {self.metadata.source_count} | "
            f"Type: {self.metadata.research_type}*"
        )
        sections.append("")

        # Executive Summary
        sections.append("## Executive Summary")
        sections.append(self.executive_summary)
        sections.append("")

        # Key Trends
        if self.key_trends:
            sections.append("## Key Trends")
            for i, trend in enumerate(self.key_trends, start=1):
                sections.append(f"{i}. {trend}")
            sections.append("")

        # Evidence
        if self.evidence:
            sections.append("## Evidence")
            for snippet in self.evidence:
                sections.append(f"> {snippet}")
                sections.append("")

        # Source Comparison
        if self.source_comparison:
            sections.append("## Source Perspective Comparison")
            for comparison in self.source_comparison:
                sections.append(f"- {comparison}")
            sections.append("")

        # Open Questions
        if self.open_questions:
            sections.append("## Open Questions")
            for question in self.open_questions:
                sections.append(f"- {question}")
            sections.append("")

        # Citations
        if self.citations:
            sections.append("## Citations")
            for citation in self.citations:
                sections.append(citation)
            sections.append("")

        return "\n".join(sections)
