"""
RelevanceFilter.

검색 결과에서 관련성이 낮은 청크와 중복 청크를 제거한다.

필터링 단계:
  1. min_score 컷  — score < min_score 인 결과 제거
  2. 중복 제거     — Jaccard 유사도 기반으로 내용이 거의 같은 청크 제거
  3. 정렬          — score 내림차순 최종 정렬

Jaccard 유사도:
  두 텍스트를 단어 집합으로 변환 후 |교집합| / |합집합| 계산.
  max_content_overlap(기본 0.85) 이상이면 score가 낮은 쪽을 제거한다.
"""
import logging
import re

from src.domain.models.search_result import SearchResult

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"\w+")


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(t.lower() for t in _TOKEN.findall(text))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class RelevanceFilter:
    """
    Args:
        min_score: 이 값 미만의 결과를 제거한다 (기본 0.0 = 필터 없음)
        max_content_overlap: 두 청크의 Jaccard 유사도가 이 값 이상이면 중복으로 간주 (기본 0.85)
    """

    def __init__(
        self,
        min_score: float = 0.0,
        max_content_overlap: float = 0.85,
    ) -> None:
        self.min_score = min_score
        self.max_content_overlap = max_content_overlap

    def filter(self, results: list[SearchResult]) -> list[SearchResult]:
        """score 필터 → 중복 제거 → 내림차순 정렬 순서로 처리한다."""
        after_score = self._apply_min_score(results)
        after_dedup = self._deduplicate(after_score)
        final = sorted(after_dedup, key=lambda r: r.score, reverse=True)

        logger.info(
            "relevance_filter input=%s after_score=%s after_dedup=%s",
            len(results),
            len(after_score),
            len(final),
        )
        return final

    def _apply_min_score(self, results: list[SearchResult]) -> list[SearchResult]:
        if self.min_score <= 0.0:
            return results
        return [r for r in results if r.score >= self.min_score]

    def _deduplicate(self, results: list[SearchResult]) -> list[SearchResult]:
        """score 높은 순으로 순회하며, 이미 선택된 청크와 내용이 겹치면 제거."""
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
        kept: list[SearchResult] = []
        kept_tokens: list[frozenset[str]] = []

        for result in sorted_results:
            tokens = _tokenize(result.document.content)
            is_duplicate = any(
                _jaccard(tokens, seen) >= self.max_content_overlap
                for seen in kept_tokens
            )
            if not is_duplicate:
                kept.append(result)
                kept_tokens.append(tokens)

        return kept
