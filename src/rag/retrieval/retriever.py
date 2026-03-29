"""
Retriever.

IngestionPipeline으로 문서를 색인하고,
쿼리 임베딩 기반 유사도 검색 → RelevanceFilter → CitationBuilder
순서로 EvidenceChunk를 생성한다.

retrieve_multi_query:
  ResearchPlan.queries[] 를 병렬 검색하고 ScoreAggregator로 병합한다.
  같은 chunk가 여러 쿼리에 매칭될 경우 최고 score를 유지한다.
"""
import asyncio
import logging

from src.domain.models.embedded_document import EmbeddedDocument
from src.domain.models.evidence_chunk import EvidenceChunk
from src.domain.models.search_result import SearchResult
from src.domain.models.source_document import SourceDocument
from src.domain.ports.retriever_port import RetrieverPort
from src.domain.ports.vector_store_port import VectorStorePort
from src.rag.citation.citation_builder import CitationBuilder
from src.rag.embedding.embedder import HashEmbedder
from src.rag.ingestion.ingestion_pipeline import IngestionPipeline
from src.rag.retrieval.relevance_filter import RelevanceFilter

logger = logging.getLogger(__name__)


class Retriever(RetrieverPort):
    def __init__(
        self,
        pipeline: IngestionPipeline,
        embedder: HashEmbedder,
        vector_store: VectorStorePort,
        citation_builder: CitationBuilder,
        relevance_filter: RelevanceFilter | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.embedder = embedder
        self.vector_store = vector_store
        self.citation_builder = citation_builder
        self.relevance_filter = relevance_filter or RelevanceFilter()
        self._source_by_id: dict[str, SourceDocument] = {}

    async def index_documents(self, docs: list[SourceDocument]) -> int:
        for doc in docs:
            self._source_by_id[doc.source_id] = doc

        embedded = await self.pipeline.run(docs)
        logger.info("retriever_indexed doc=%s chunks=%s", len(docs), len(embedded))
        return len(embedded)

    async def retrieve(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.0,
        source_type_filter: str | None = None,
    ) -> list[EvidenceChunk]:
        """
        단일 쿼리로 검색한다.
        결과에 실제 코사인 유사도 score를 담아 반환한다.
        """
        query_vector = await self.embedder.embed(query)
        raw_results: list[SearchResult] = await self.vector_store.similarity_search(
            query_vector=query_vector,
            k=k * 2,  # 필터 후 k개 확보를 위해 넉넉하게 요청
            source_type_filter=source_type_filter,
        )

        filtered = self.relevance_filter.filter(raw_results)
        if min_score > 0.0:
            filtered = [r for r in filtered if r.score >= min_score]

        top = filtered[:k]
        chunks = self._to_evidence_chunks(top)

        logger.info(
            "retriever_retrieve query_len=%s k=%s raw=%s filtered=%s returned=%s",
            len(query),
            k,
            len(raw_results),
            len(filtered),
            len(chunks),
        )
        return chunks

    async def retrieve_multi_query(
        self,
        queries: list[str],
        k: int = 5,
        min_score: float = 0.0,
        source_type_filter: str | None = None,
    ) -> list[EvidenceChunk]:
        """
        복수 쿼리를 병렬로 검색하고 결과를 병합한다.

        병합 전략:
          - 동일 chunk_id가 여러 쿼리에서 나타나면 score 최대값을 유지
          - 병합 후 전체에 RelevanceFilter 적용
          - 최종 score 내림차순으로 상위 k개 반환
        """
        if not queries:
            return []

        query_vectors = await asyncio.gather(
            *[self.embedder.embed(q) for q in queries]
        )

        search_tasks = [
            self.vector_store.similarity_search(
                query_vector=qv,
                k=k * 2,
                source_type_filter=source_type_filter,
            )
            for qv in query_vectors
        ]
        per_query_results: list[list[SearchResult]] = await asyncio.gather(*search_tasks)

        merged = self._aggregate_scores(per_query_results)
        filtered = self.relevance_filter.filter(merged)
        if min_score > 0.0:
            filtered = [r for r in filtered if r.score >= min_score]

        top = filtered[:k]
        chunks = self._to_evidence_chunks(top)

        logger.info(
            "retriever_multi_query queries=%s k=%s merged=%s filtered=%s returned=%s",
            len(queries),
            k,
            len(merged),
            len(filtered),
            len(chunks),
        )
        return chunks

    def _aggregate_scores(
        self,
        per_query_results: list[list[SearchResult]],
    ) -> list[SearchResult]:
        """
        여러 쿼리 결과를 병합한다.
        동일 chunk_id는 score 최대값으로 유지한다.
        """
        best: dict[str, SearchResult] = {}
        for results in per_query_results:
            for result in results:
                cid = result.document.chunk_id
                if cid not in best or result.score > best[cid].score:
                    best[cid] = result
        return list(best.values())

    def _to_evidence_chunks(self, results: list[SearchResult]) -> list[EvidenceChunk]:
        """SearchResult 목록을 EvidenceChunk 목록으로 변환하고 citation을 부여한다."""
        chunks: list[EvidenceChunk] = []
        for result in results:
            doc = result.document
            source = self._source_by_id.get(doc.source_id)

            citation = None
            if source:
                citation = self.citation_builder.build(
                    title=source.title,
                    url=source.url,
                    source_type=source.source_type,
                    source_id=source.source_id,
                    score=result.score,
                )

            chunks.append(
                EvidenceChunk(
                    chunk_id=doc.chunk_id,
                    source_id=doc.source_id,
                    content=doc.content,
                    score=result.score,
                    citation=citation,
                    metadata=doc.metadata,
                )
            )

        return self.citation_builder.build_for_chunks(chunks)
