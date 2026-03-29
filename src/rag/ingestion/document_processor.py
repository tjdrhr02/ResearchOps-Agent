"""
DocumentProcessor.

SourceDocument를 NormalizedDocument로 변환하는 파이프라인 조율자.

실행 순서:
  1. HtmlParser   → HTML 태그 제거, 본문 추출
  2. TextCleaner  → 유니코드/공백/상용구 정제
  3. SourceTypeProcessor → paper/blog/news 유형별 처리
  4. NormalizedDocument 생성

설계 원칙:
  - 개별 문서 처리 실패는 경고 후 None 반환 (전체 파이프라인 중단 금지)
  - SourceDocument는 불변, NormalizedDocument로만 새 데이터를 만든다
"""
import logging

from src.domain.models.normalized_document import NormalizedDocument
from src.domain.models.source_document import SourceDocument
from src.rag.ingestion.html_parser import HtmlParser
from src.rag.ingestion.source_type_processors import get_processor
from src.rag.ingestion.text_cleaner import TextCleaner

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self) -> None:
        self.html_parser = HtmlParser()
        self.text_cleaner = TextCleaner()

    def process(self, doc: SourceDocument) -> NormalizedDocument | None:
        try:
            return self._run_pipeline(doc)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "document_processor_failed source_id=%s source_type=%s error=%s",
                doc.source_id,
                doc.source_type,
                exc,
            )
            return None

    def process_many(self, docs: list[SourceDocument]) -> list[NormalizedDocument]:
        results: list[NormalizedDocument] = []
        for doc in docs:
            normalized = self.process(doc)
            if normalized:
                results.append(normalized)

        logger.info(
            "document_processor_batch total=%s normalized=%s failed=%s",
            len(docs),
            len(results),
            len(docs) - len(results),
        )
        return results

    def _run_pipeline(self, doc: SourceDocument) -> NormalizedDocument:
        # 1단계: HTML 파싱 (content가 없으면 title을 사용하되 word_count는 0 처리를 위해 빈 상태 보존)
        raw_content = doc.content or ""
        parsed_text = self.html_parser.parse(raw_content) if raw_content else ""
        logger.debug("html_parsed source_id=%s length=%s", doc.source_id, len(parsed_text))

        # 2단계: 텍스트 정제
        clean_text = self.text_cleaner.clean(parsed_text)
        logger.debug("text_cleaned source_id=%s length=%s", doc.source_id, len(clean_text))

        # 3단계: source_type별 처리
        meta = dict(doc.metadata)
        processor = get_processor(doc.source_type)
        if processor:
            clean_text, meta = processor.process(clean_text, meta)
            logger.debug(
                "source_type_processed source_id=%s type=%s",
                doc.source_id,
                doc.source_type,
            )

        word_count = len(clean_text.split()) if clean_text else 0
        meta["word_count"] = str(word_count)
        meta["processed"] = "true"

        logger.info(
            "document_normalized source_id=%s type=%s words=%s",
            doc.source_id,
            doc.source_type,
            word_count,
        )
        return NormalizedDocument(
            source_id=doc.source_id,
            source_type=doc.source_type,
            title=doc.title,
            url=doc.url,
            clean_content=clean_text,
            word_count=word_count,
            metadata=meta,
        )
