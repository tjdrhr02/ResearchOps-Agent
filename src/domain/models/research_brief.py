from pydantic import BaseModel, Field


class ResearchBrief(BaseModel):
    executive_summary: str
    key_trends: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    source_comparison: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
