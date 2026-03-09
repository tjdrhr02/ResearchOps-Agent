from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    source_id: str
    source_type: str
    title: str
    url: str
    content: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)
