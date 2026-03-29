"""
IngestionPipeline.

수집된 SourceDocument를 RAG 저장소에 색인하는 파이프라인 진입점.

처리 흐름:
  SourceDocument[]
    → DocumentIngestor   (HTML 파싱 + 텍스트 정제)
    → SemanticChunker    (단락 경계 청킹 + overlap)
    → EmbeddingService   (벡터 변환, 배치 처리)
    → VectorStorePort    (pgvector 저장)
    → list[EmbeddedDocument]  (저장된 청크 반환)

설계 원칙:
  - 각 단계는 독립 교체 가능 (DI)
  - 청크 단위 배치 embed로 API 호출 횟수 최소화
  - 단계별 오류 격리 및 로깅
  - MetricsPort를 통한 ingestion 지표 수집
"""
import logging
import time

from src.domain.models.embedded_document import EmbeddedDocument
from src.domain.models.source_document import SourceDocument
from src.domain.ports.metrics_port import MetricsPort
from src.domain.ports.vector_store_port import VectorStorePort
from src.rag.chunking.chunker import SemanticChunker
from src.rag.embedding.embedder import HashEmbedder
from src.rag.ingestion.ingestor import DocumentIngestor

logger = logging.getLogger(__name__)

_EMBED_BATCH_SIZE = 32


class IngestionPipeline:
    """
    RAG 색인 파이프라인.

    의존성은 모두 주입받아 단계별 교체 및 테스트를 지원한다.
    """

    def __init__(
        self,
        ingestor: DocumentIngestor,
        chunker: SemanticChunker,
        embedder: HashEmbedder,
        vector_store: VectorStorePort,
        metrics: MetricsPort | None = None,
    ) -> None:
        self.ingestor = ingestor
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.metrics = metrics

    async def run(self, docs: list[SourceDocument]) -> list[EmbeddedDocument]:
        """
        문서 목록을 색인하고 저장된 EmbeddedDocument 목록을 반환한다.

        Args:
            docs: Collector Agent가 수집한 SourceDocument 목록

        Returns:
            벡터까지 저장된 EmbeddedDocument 목록
        """
        t0 = time.monotonic()
        logger.info("ingestion_pipeline_start doc_count=%s", len(docs))

        # 1단계: HTML 파싱 + 텍스트 정제
        ingested = await self.ingestor.ingest(docs)
        logger.info("ingestion_stage_ingest done count=%s", len(ingested))

        # 2단계: 청킹
        chunks = await self.chunker.chunk(ingested)
        logger.info("ingestion_stage_chunk done count=%s", len(chunks))

        if not chunks:
            logger.warning("ingestion_pipeline_empty_chunks doc_count=%s", len(docs))
            return []

        # 3단계: 배치 임베딩
        embedded_docs = await self._embed_chunks(chunks)
        logger.info("ingestion_stage_embed done count=%s", len(embedded_docs))

        # 4단계: 벡터 저장
        saved = await self.vector_store.upsert(embedded_docs)
        elapsed = time.monotonic() - t0
        logger.info(
            "ingestion_pipeline_done doc=%s chunk=%s saved=%s elapsed=%.3fs",
            len(docs),
            len(chunks),
            saved,
            elapsed,
        )

        if self.metrics:
            self.metrics.observe_latency("ingestion_pipeline", elapsed * 1000)
            self.metrics.increment("ingestion_chunks_total", saved)

        return embedded_docs

    async def _embed_chunks(self, chunks: list) -> list[EmbeddedDocument]:
        """청크를 _EMBED_BATCH_SIZE 단위로 나눠 임베딩하고 EmbeddedDocument를 생성한다."""
        embedded: list[EmbeddedDocument] = []

        for batch_start in range(0, len(chunks), _EMBED_BATCH_SIZE):
            batch = chunks[batch_start : batch_start + _EMBED_BATCH_SIZE]
            texts = [c.content for c in batch]
            vectors = await self.embedder.embed_batch(texts)

            for chunk, vector in zip(batch, vectors):
                embedded.append(
                    EmbeddedDocument(
                        chunk_id=chunk.chunk_id,
                        source_id=chunk.source_id,
                        source_type=chunk.source_type,
                        content=chunk.content,
                        embedding=vector,
                        chunk_index=chunk.chunk_index,
                        metadata=chunk.metadata,
                    )
                )

        return embedded

    async def delete_source(self, source_id: str) -> int:
        """특정 source_id의 모든 청크를 벡터 저장소에서 삭제한다."""
        removed = await self.vector_store.delete_by_source(source_id)
        logger.info("ingestion_pipeline_delete source_id=%s removed=%s", source_id, removed)
        return removed
