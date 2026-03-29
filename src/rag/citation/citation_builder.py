"""
CitationBuilder.

EvidenceChunk 목록에서 구조화된 인용문을 생성한다.

인용 포맷:
  단건: "[1] Title — URL (source_type, score: 0.87)"
  블록: 번호 부여 + 출처별 그룹화

설계 원칙:
  - citation 번호는 호출 순서 기반 (1-indexed)
  - 동일 source_id가 여러 청크로 나뉜 경우 번호는 최초 등장 시 부여
  - build()는 단건 citation 문자열 반환 (EvidenceChunk.citation 필드용)
  - build_reference_list()는 최종 Brief에 붙이는 참고문헌 블록 반환
"""
from dataclasses import dataclass, field

from src.domain.models.evidence_chunk import EvidenceChunk


@dataclass
class CitationBuilder:
    """
    인용 번호와 포맷을 관리한다.
    인스턴스를 재사용하면 번호가 누적되므로, 새 리서치마다 fresh 인스턴스를 사용한다.
    """

    _index: dict[str, int] = field(default_factory=dict)
    _counter: int = 1

    def build(
        self,
        title: str,
        url: str,
        source_type: str,
        source_id: str = "",
        score: float | None = None,
    ) -> str:
        """
        단건 citation 문자열을 반환한다.
        source_id가 동일하면 같은 번호를 재사용한다.

        Example:
            "[1] Attention Is All You Need — https://arxiv.org/abs/1706.03762 (paper, score: 0.91)"
        """
        num = self._get_or_assign(source_id or f"{source_type}:{url}")
        score_part = f", score: {score:.2f}" if score is not None else ""
        return f"[{num}] {title} — {url} ({source_type}{score_part})"

    def build_for_chunks(self, chunks: list[EvidenceChunk]) -> list[EvidenceChunk]:
        """
        청크 목록에 citation 번호를 부여하고 새 EvidenceChunk 목록을 반환한다.
        citation 필드가 None인 청크도 처리한다.
        """
        result: list[EvidenceChunk] = []
        for chunk in chunks:
            num = self._get_or_assign(chunk.source_id)
            citation = chunk.citation or ""
            # 번호가 이미 없으면 앞에 [N] 번호를 붙인다
            if not citation.startswith(f"[{num}]"):
                citation = f"[{num}] {citation}".strip()
            result.append(chunk.model_copy(update={"citation": citation}))
        return result

    def build_reference_list(self, chunks: list[EvidenceChunk]) -> str:
        """
        Brief 하단에 붙이는 참고문헌 블록을 생성한다.

        Example output:
            ## References
            [1] Paper Title — https://... (paper)
            [2] Blog Post — https://... (blog)
        """
        seen: dict[str, str] = {}
        for chunk in chunks:
            if chunk.citation and chunk.source_id not in seen:
                seen[chunk.source_id] = chunk.citation

        if not seen:
            return ""

        lines = ["## References"]
        for citation in seen.values():
            lines.append(citation)
        return "\n".join(lines)

    def reset(self) -> None:
        """번호 카운터를 초기화한다. 새 리서치 세션 시작 시 호출한다."""
        self._index.clear()
        self._counter = 1

    def _get_or_assign(self, key: str) -> int:
        if key not in self._index:
            self._index[key] = self._counter
            self._counter += 1
        return self._index[key]
