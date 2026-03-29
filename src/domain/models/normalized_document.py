"""
NormalizedDocument.

HTML 파싱 / 텍스트 정제 / 유형별 처리를 마친 문서 모델.
RAG 파이프라인(chunking → embedding → retrieval)의 입력으로 사용된다.
"""
from pydantic import BaseModel, Field


class NormalizedDocument(BaseModel):
    source_id: str
    source_type: str
    title: str
    url: str
    clean_content: str
    word_count: int = 0
    language: str = "en"
    metadata: dict[str, str] = Field(default_factory=dict)
