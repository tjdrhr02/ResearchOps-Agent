from pydantic import BaseModel, Field


class ResearchRunRequest(BaseModel):
    user_query: str = Field(min_length=3, description="Research question from user")
    max_sources: int = Field(default=8, ge=1, le=30)


class SaveNoteRequest(BaseModel):
    note: str = Field(min_length=1, description="Research note content")
