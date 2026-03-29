"""
RetrieverPort.

RAG retrieval 계층의 추상 인터페이스.
index_documents → retrieve 순서로 호출한다.
"""
from abc import ABC, abstractmethod

from src.domain.models.evidence_chunk import EvidenceChunk
from src.domain.models.source_document import SourceDocument


class RetrieverPort(ABC):
    @abstractmethod
    async def index_documents(self, docs: list[SourceDocument]) -> int:
        """문서를 색인하고 저장된 청크 수를 반환한다."""
        raise NotImplementedError

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.0,
        source_type_filter: str | None = None,
    ) -> list[EvidenceChunk]:
        """
        쿼리와 유사한 청크를 반환한다.

        Args:
            query: 검색 쿼리 문자열
            k: 반환할 최대 청크 수
            min_score: 이 값 미만의 결과를 제외한다 (0.0 = 필터 없음)
            source_type_filter: "paper" | "blog" | "news" 등으로 소스 유형 한정

        Returns:
            score 내림차순으로 정렬된 EvidenceChunk 목록
        """
        raise NotImplementedError

    async def retrieve_multi_query(
        self,
        queries: list[str],
        k: int = 5,
        min_score: float = 0.0,
        source_type_filter: str | None = None,
    ) -> list[EvidenceChunk]:
        """
        여러 쿼리로 검색하고 결과를 병합한다. 기본 구현은 단건 retrieve를 순차 호출한다.
        Retriever 구현체에서 오버라이드해 병렬 처리할 수 있다.
        """
        seen_ids: set[str] = set()
        combined: list[EvidenceChunk] = []
        for query in queries:
            results = await self.retrieve(
                query=query,
                k=k,
                min_score=min_score,
                source_type_filter=source_type_filter,
            )
            for chunk in results:
                if chunk.chunk_id not in seen_ids:
                    seen_ids.add(chunk.chunk_id)
                    combined.append(chunk)
        return sorted(combined, key=lambda c: c.score, reverse=True)[:k]
