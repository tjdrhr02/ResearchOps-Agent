"""
SemanticChunker.

단락 경계를 존중하면서 청킹하고 overlap을 적용한다.

알고리즘:
  1. 빈 줄(\n\n)을 기준으로 단락 분리
  2. 단락을 순서대로 누적해 chunk_size 초과 시 청크 확정
  3. overlap_size만큼 앞 청크 끝 문장을 다음 청크 앞에 붙임
  4. 최소 길이(min_chunk_chars)에 못 미치는 단락은 다음 단락과 병합

청크 메타데이터:
  - chunk_id: "{source_id}:chunk{index}"
  - chunk_index: 순서 번호
  - source_type: 원본 문서 유형
"""
import re
from dataclasses import dataclass, field

from src.domain.models.embedded_document import TextChunk
from src.domain.models.source_document import SourceDocument

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class SemanticChunker:
    chunk_size: int = 500
    overlap_size: int = 80
    min_chunk_chars: int = 50

    async def chunk(self, docs: list[SourceDocument]) -> list[TextChunk]:
        result: list[TextChunk] = []
        for doc in docs:
            result.extend(self._chunk_document(doc))
        return result

    def _chunk_document(self, doc: SourceDocument) -> list[TextChunk]:
        content = (doc.content or doc.title).strip()
        if not content:
            return []

        paragraphs = self._split_paragraphs(content)
        raw_chunks = self._build_chunks(paragraphs)
        chunks: list[TextChunk] = []

        for idx, chunk_text in enumerate(raw_chunks):
            chunks.append(
                TextChunk(
                    chunk_id=f"{doc.source_id}:chunk{idx}",
                    source_id=doc.source_id,
                    source_type=doc.source_type,
                    content=chunk_text,
                    chunk_index=idx,
                    metadata={
                        **doc.metadata,
                        "total_chunks": str(len(raw_chunks)),
                    },
                )
            )
        return chunks

    def _split_paragraphs(self, text: str) -> list[str]:
        raw = re.split(r"\n{2,}", text)
        merged: list[str] = []
        buf = ""
        for para in raw:
            para = para.strip()
            if not para:
                continue
            if len(buf) + len(para) < self.min_chunk_chars:
                buf = f"{buf} {para}".strip()
            else:
                if buf:
                    merged.append(buf)
                buf = para
        if buf:
            merged.append(buf)
        return merged

    def _build_chunks(self, paragraphs: list[str]) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for para in paragraphs:
            if current_len + len(para) > self.chunk_size and current:
                chunks.append("\n\n".join(current))
                # overlap: 현재 청크의 마지막 문장들을 다음 청크 시작에 포함
                overlap_text = self._last_sentences(current[-1], self.overlap_size)
                current = [overlap_text] if overlap_text else []
                current_len = len(overlap_text)

            current.append(para)
            current_len += len(para)

        if current:
            chunks.append("\n\n".join(current))

        return [c for c in chunks if c.strip()]

    def _last_sentences(self, text: str, max_chars: int) -> str:
        sentences = _SENTENCE_END.split(text.strip())
        result = ""
        for sentence in reversed(sentences):
            candidate = f"{sentence} {result}".strip()
            if len(candidate) <= max_chars:
                result = candidate
            else:
                break
        return result.strip()


# 기존 코드와의 호환을 위한 alias
SimpleChunker = SemanticChunker
