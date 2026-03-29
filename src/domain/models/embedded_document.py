"""
EmbeddedDocument.

텍스트 청크와 임베딩 벡터를 묶은 모델.
VectorStore 저장 단위이며 retrieval 결과의 원본이 된다.
"""
from pydantic import BaseModel, Field


class TextChunk(BaseModel):
    chunk_id: str
    source_id: str
    source_type: str
    content: str
    chunk_index: int = 0
    metadata: dict[str, str] = Field(default_factory=dict)


class EmbeddedDocument(BaseModel):
    chunk_id: str
    source_id: str
    source_type: str
    content: str
    embedding: list[float]
    chunk_index: int = 0
    metadata: dict[str, str] = Field(default_factory=dict)
