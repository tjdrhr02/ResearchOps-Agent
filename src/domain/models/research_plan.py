from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    question: str
    objective: str
    queries: list[str] = Field(default_factory=list)
    source_types: list[str] = Field(default_factory=list)
