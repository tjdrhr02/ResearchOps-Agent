"""
Document Normalizer.

Tool 실행 결과의 raw dict를 SourceDocument로 정규화하고
필수 metadata 필드를 보완한다.

정규화 규칙:
  - source_id: 없으면 URL 기반 생성
  - source_type: source_priority에서 전달받은 값으로 보완
  - metadata: provider / query / source_type 항상 포함
  - content: 공백 정리
"""
import hashlib
import logging
from typing import Any

from src.domain.models.source_document import SourceDocument

logger = logging.getLogger(__name__)

_REQUIRED_METADATA = {"provider", "query", "source_type"}


class DocumentNormalizer:
    def normalize(
        self,
        raw: Any,
        source_type: str = "",
        query: str = "",
    ) -> SourceDocument | None:
        try:
            if isinstance(raw, SourceDocument):
                return self._enrich_metadata(raw, source_type=source_type, query=query)

            if isinstance(raw, dict):
                doc_dict = dict(raw)
                doc_dict = self._fill_missing_fields(
                    doc_dict,
                    source_type=source_type,
                    query=query,
                )
                doc = SourceDocument(**doc_dict)
                return self._enrich_metadata(doc, source_type=source_type, query=query)

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "normalizer_failed source_type=%s query=%s error=%s",
                source_type,
                query,
                exc,
            )
        return None

    def _fill_missing_fields(
        self,
        doc_dict: dict[str, Any],
        source_type: str,
        query: str,
    ) -> dict[str, Any]:
        if not doc_dict.get("source_id"):
            url = doc_dict.get("url", query)
            doc_dict["source_id"] = hashlib.md5(url.encode()).hexdigest()  # noqa: S324

        if not doc_dict.get("source_type") and source_type:
            doc_dict["source_type"] = source_type

        if not doc_dict.get("title"):
            doc_dict["title"] = f"[{source_type}] {query}"

        if not doc_dict.get("url"):
            doc_dict["url"] = ""

        doc_dict["content"] = (doc_dict.get("content") or "").strip()

        if "metadata" not in doc_dict:
            doc_dict["metadata"] = {}

        return doc_dict

    def _enrich_metadata(
        self,
        doc: SourceDocument,
        source_type: str,
        query: str,
    ) -> SourceDocument:
        enriched = dict(doc.metadata)
        if "query" not in enriched:
            enriched["query"] = query
        if "source_type" not in enriched:
            enriched["source_type"] = source_type or doc.source_type
        if "provider" not in enriched:
            enriched["provider"] = "unknown"

        return SourceDocument(
            source_id=doc.source_id,
            source_type=doc.source_type,
            title=doc.title,
            url=doc.url,
            content=doc.content,
            metadata=enriched,
        )
