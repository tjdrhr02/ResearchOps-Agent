"""
Duplicate Filter.

수집된 SourceDocument에서 중복을 제거한다.

중복 기준 (우선순위 순서):
  1. URL 정규화 후 완전 일치
  2. 제목 정규화 후 완전 일치 (URL이 없는 경우 보완)
"""
import logging
import re

from src.domain.models.source_document import SourceDocument

logger = logging.getLogger(__name__)


def _normalize_url(url: str) -> str:
    url = url.strip().rstrip("/").lower()
    url = re.sub(r"^https?://", "", url)
    return url


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.strip().lower())


class DuplicateFilter:
    def filter(self, docs: list[SourceDocument]) -> list[SourceDocument]:
        seen_urls: set[str] = set()
        seen_titles: set[str] = set()
        unique: list[SourceDocument] = []

        for doc in docs:
            url_key = _normalize_url(doc.url)
            title_key = _normalize_title(doc.title)

            if url_key and url_key in seen_urls:
                logger.debug("dedup_url_skip url=%s", doc.url)
                continue

            if title_key and title_key in seen_titles:
                logger.debug("dedup_title_skip title=%s", doc.title)
                continue

            if url_key:
                seen_urls.add(url_key)
            if title_key:
                seen_titles.add(title_key)
            unique.append(doc)

        removed = len(docs) - len(unique)
        if removed:
            logger.info("dedup_done removed=%s remaining=%s", removed, len(unique))

        return unique
