"""
VectorStorePort.

벡터 저장소의 인터페이스.
similarity_search는 실제 코사인 유사도 score를 포함한 SearchResult를 반환한다.
"""
from abc import ABC, abstractmethod

from src.domain.models.embedded_document import EmbeddedDocument
from src.domain.models.search_result import SearchResult


class VectorStorePort(ABC):
    @abstractmethod
    async def upsert(self, docs: list[EmbeddedDocument]) -> int:
        """문서 임베딩을 저장하고 저장된 수를 반환한다."""
        raise NotImplementedError

    @abstractmethod
    async def similarity_search(
        self,
        query_vector: list[float],
        k: int = 5,
        source_type_filter: str | None = None,
    ) -> list[SearchResult]:
        """
        코사인 유사도 기준으로 상위 k개 청크를 반환한다.

        Returns:
            score 내림차순으로 정렬된 SearchResult 목록
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_by_source(self, source_id: str) -> int:
        """특정 source_id의 모든 청크를 삭제하고 삭제된 수를 반환한다."""
        raise NotImplementedError
