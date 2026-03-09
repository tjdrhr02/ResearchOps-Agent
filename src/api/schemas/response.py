from pydantic import BaseModel

from src.domain.models.research_job import ResearchJob
from src.domain.models.research_brief import ResearchBrief
from src.domain.models.source_document import SourceDocument


class ResearchRunResponse(BaseModel):
    job_id: str
    status: str
    brief: ResearchBrief


class ResearchJobResponse(BaseModel):
    job: ResearchJob


class ResearchSourcesResponse(BaseModel):
    job_id: str
    sources: list[SourceDocument]


class SaveNoteResponse(BaseModel):
    note_id: str


class SearchNotesResponse(BaseModel):
    items: list[str]
