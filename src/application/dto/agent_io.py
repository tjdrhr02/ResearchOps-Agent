from pydantic import BaseModel, Field

from src.domain.models.evidence_chunk import EvidenceChunk
from src.domain.models.research_brief import ResearchBrief
from src.domain.models.research_plan import ResearchPlan
from src.domain.models.source_document import SourceDocument


class PlannerOutput(BaseModel):
    plan: ResearchPlan


class CollectorOutput(BaseModel):
    documents: list[SourceDocument] = Field(default_factory=list)


class SynthesizerOutput(BaseModel):
    trend_summary: str = ""
    claims: list[str] = Field(default_factory=list)
    comparisons: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    evidence: list[EvidenceChunk] = Field(default_factory=list)


class StepResult(BaseModel):
    """워크플로우 단계 하나의 실행 결과."""

    step: str
    status: str          # "success" | "failed" | "partial"
    elapsed_ms: float
    detail: str = ""
    error: str | None = None


class WorkflowTrace(BaseModel):
    """전체 워크플로우의 단계별 실행 기록."""

    steps: list[StepResult] = Field(default_factory=list)

    def add(
        self,
        step: str,
        status: str,
        elapsed_ms: float,
        detail: str = "",
        error: str | None = None,
    ) -> None:
        self.steps.append(
            StepResult(
                step=step,
                status=status,
                elapsed_ms=round(elapsed_ms, 2),
                detail=detail,
                error=error,
            )
        )

    def has_failure(self) -> bool:
        return any(s.status == "failed" for s in self.steps)

    def summary(self) -> str:
        return " | ".join(
            f"{s.step}={s.status}({s.elapsed_ms:.0f}ms)" for s in self.steps
        )


class WorkflowResult(BaseModel):
    brief: ResearchBrief
    sources: list[SourceDocument] = Field(default_factory=list)
    trace: WorkflowTrace = Field(default_factory=WorkflowTrace)
