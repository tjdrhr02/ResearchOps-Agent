from pydantic import BaseModel, Field


class Settings(BaseModel):
    app_name: str = "ResearchOps Agent"
    postgres_dsn: str = Field(default="postgresql://localhost:5432/researchops")
    redis_dsn: str = Field(default="redis://localhost:6379/0")
    llm_default_model: str = Field(default="gpt-4o-mini")
    llm_planner_model: str = Field(default="gpt-4o-mini")
    llm_synthesizer_model: str = Field(default="gpt-4o")
