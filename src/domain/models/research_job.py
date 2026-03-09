from pydantic import BaseModel, Field

from src.domain.models.research_brief import ResearchBrief
from src.domain.models.source_document import SourceDocument


class ResearchJob(BaseModel):
    job_id: str
    query: str
    status: str = "completed"
    brief: ResearchBrief
    sources: list[SourceDocument] = Field(default_factory=list)
