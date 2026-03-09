from pydantic import BaseModel, Field


class Settings(BaseModel):
    app_name: str = "ResearchOps Agent"
    postgres_dsn: str = Field(default="postgresql://localhost:5432/researchops")
    redis_dsn: str = Field(default="redis://localhost:6379/0")
