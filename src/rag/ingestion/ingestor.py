"""
DocumentIngestor.

Collector가 수집한 SourceDocument를 RAG 파이프라인에
입력 가능한 형태로 변환한다.

처리 흐름:
  SourceDocument → DocumentProcessor → NormalizedDocument → SourceDocument(clean)

NormalizedDocument를 SourceDocument로 다시 매핑하는 이유:
  하위 RAG(chunker/embedder) 인터페이스가 SourceDocument를 기대하므로
  clean_content를 content 필드로 교체한 SourceDocument를 반환한다.
  NormalizedDocument는 metadata에 전체 정제 정보를 담아 보존한다.
"""
import logging

from src.domain.models.source_document import SourceDocument
from src.rag.ingestion.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


class DocumentIngestor:
    def __init__(self) -> None:
        self.processor = DocumentProcessor()

    async def ingest(self, docs: list[SourceDocument]) -> list[SourceDocument]:
        normalized_docs = self.processor.process_many(docs)

        result: list[SourceDocument] = []
        for nd in normalized_docs:
            result.append(
                SourceDocument(
                    source_id=nd.source_id,
                    source_type=nd.source_type,
                    title=nd.title,
                    url=nd.url,
                    content=nd.clean_content,
                    metadata=nd.metadata,
                )
            )

        logger.info(
            "ingestor_done input=%s output=%s",
            len(docs),
            len(result),
        )
        return result
