from abc import ABC, abstractmethod

from src.domain.models.evidence_chunk import EvidenceChunk
from src.domain.models.source_document import SourceDocument


class RetrieverPort(ABC):
    @abstractmethod
    async def index_documents(self, docs: list[SourceDocument]) -> int:
        raise NotImplementedError

    @abstractmethod
    async def retrieve(self, query: str, k: int = 5) -> list[EvidenceChunk]:
        raise NotImplementedError
