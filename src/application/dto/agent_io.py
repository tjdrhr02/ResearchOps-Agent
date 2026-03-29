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


class WorkflowResult(BaseModel):
    brief: ResearchBrief
    sources: list[SourceDocument] = Field(default_factory=list)
