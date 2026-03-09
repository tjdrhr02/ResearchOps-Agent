from pydantic import BaseModel, Field


class EvidenceChunk(BaseModel):
    chunk_id: str
    source_id: str
    content: str
    score: float
    citation: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
