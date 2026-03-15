from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    question: str
    objective: str
    research_type: str = "general_research"
    queries: list[str] = Field(default_factory=list)
    focus_topics: list[str] = Field(default_factory=list)
    source_priority: list[str] = Field(default_factory=list)
    source_types: list[str] = Field(default_factory=list)
