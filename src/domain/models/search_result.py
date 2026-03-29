"""
SearchResult.

VectorStore 검색의 단건 결과 타입.
EmbeddedDocument와 코사인 유사도 score를 묶어 retrieval 계층에 전달한다.

score 범위: 0.0 (무관) ~ 1.0 (완전 일치)
"""
from pydantic import BaseModel, Field

from src.domain.models.embedded_document import EmbeddedDocument


class SearchResult(BaseModel):
    document: EmbeddedDocument
    score: float = Field(ge=0.0, le=1.0, description="코사인 유사도 (0~1)")
