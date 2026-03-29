"""
InMemoryVectorStore.

개발/테스트 환경에서 pgvector 없이 동작하는 인메모리 벡터 저장소.
VectorStorePort를 구현하므로 PgVectorStore와 교체 가능하다.

유사도: 코사인 유사도 (dot product / (|a| * |b|)), 범위 0.0 ~ 1.0
"""
import logging
import math

from src.domain.models.embedded_document import EmbeddedDocument
from src.domain.models.search_result import SearchResult
from src.domain.ports.vector_store_port import VectorStorePort

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryVectorStore(VectorStorePort):
    def __init__(self) -> None:
        self._store: dict[str, EmbeddedDocument] = {}

    async def upsert(self, docs: list[EmbeddedDocument]) -> int:
        for doc in docs:
            self._store[doc.chunk_id] = doc
        logger.info("vector_store_upsert count=%s total=%s", len(docs), len(self._store))
        return len(docs)

    async def similarity_search(
        self,
        query_vector: list[float],
        k: int = 5,
        source_type_filter: str | None = None,
    ) -> list[SearchResult]:
        candidates = list(self._store.values())
        if source_type_filter:
            candidates = [d for d in candidates if d.source_type == source_type_filter]

        scored = [
            SearchResult(
                document=doc,
                score=max(0.0, min(1.0, _cosine_similarity(query_vector, doc.embedding))),
            )
            for doc in candidates
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        results = scored[:k]

        logger.info(
            "vector_store_search k=%s filter=%s candidates=%s found=%s",
            k,
            source_type_filter,
            len(candidates),
            len(results),
        )
        return results

    async def delete_by_source(self, source_id: str) -> int:
        keys = [k for k, v in self._store.items() if v.source_id == source_id]
        for k in keys:
            del self._store[k]
        logger.info("vector_store_delete source_id=%s removed=%s", source_id, len(keys))
        return len(keys)

    def __len__(self) -> int:
        return len(self._store)


# 기존 코드와의 호환을 위한 alias
InMemoryPgVectorStore = InMemoryVectorStore
