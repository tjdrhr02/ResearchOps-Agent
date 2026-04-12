import os

from pydantic import BaseModel, Field


class Settings(BaseModel):
    app_name: str = "ResearchOps Agent"
    postgres_dsn: str = Field(default="postgresql://localhost:5432/researchops")
    redis_dsn: str = Field(default="redis://localhost:6379/0")

    # Google API
    google_api_key: str = Field(default="")

    # LLM 모델 설정 — .env의 LLM_MODEL로 전체 교체 가능
    llm_model: str = Field(default="gemini-2.5-flash")


def get_settings() -> Settings:
    return Settings(
        google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
        llm_model=os.environ.get("LLM_MODEL", "gemini-2.5-flash"),
    )
